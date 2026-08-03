from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage32_dgm_calibration_utils import (
    load_json,
    project_root,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage32_sequence_dgm_calibration_config.json"
    )
    output = config["outputs"]
    rules = config["decision_rules"]

    print("=" * 128)
    print("STAGE 123 - SUMMARIZE DGM CALIBRATION")
    print("=" * 128)

    metrics = pd.read_csv(
        root / output["metrics_table"],
        low_memory=False,
    )
    recommended = load_json(
        root / output["recommended_dgm"]
    )
    manifest = load_json(root / output["manifest"])

    validation = metrics[
        metrics["evaluation_set"] == "calibrated_validation"
    ].copy()
    training = metrics[
        metrics["evaluation_set"] == "calibrated_training"
    ].copy()

    validation_abs = np.abs(
        pd.to_numeric(
            validation["standardized_residual"],
            errors="raise",
        ).to_numpy(dtype=float)
    )
    training_abs = np.abs(
        pd.to_numeric(
            training["standardized_residual"],
            errors="raise",
        ).to_numpy(dtype=float)
    )

    validation_max = float(np.max(validation_abs))
    validation_rms = float(
        np.sqrt(np.mean(validation_abs ** 2))
    )
    training_max = float(np.max(training_abs))
    training_rms = float(
        np.sqrt(np.mean(training_abs ** 2))
    )

    passed = bool(
        validation_max
        <= float(
            rules["maximum_absolute_standardized_residual"]
        )
        and validation_rms
        <= float(
            rules[
                "maximum_root_mean_squared_standardized_residual"
            ]
        )
    )

    final = {
        "status": (
            "STAGE32_DGM_CALIBRATION_ACCEPTED"
            if passed
            else "STAGE32_DGM_CALIBRATION_REQUIRES_REVIEW"
        ),
        "calibration_id": manifest["calibration_id"],
        "calibration_accepted": passed,
        "training_maximum_absolute_standardized_residual": (
            training_max
        ),
        "training_rms_standardized_residual": training_rms,
        "validation_maximum_absolute_standardized_residual": (
            validation_max
        ),
        "validation_rms_standardized_residual": validation_rms,
        "decision_rules": rules,
        "calibrated_parameters": recommended[
            "calibrated_parameters"
        ],
        "validation_metrics": recommended[
            "validation_metrics"
        ],
        "next_action": (
            "Build and lock a revised simulation pilot with null and "
            "beneficial effect regimes and omitted-sequencing balance diagnostics."
            if passed
            else "Review target incompatibility or widen only scientifically "
            "defensible parameter bounds before another calibration."
        ),
        "boundary": config["boundary"],
    }
    write_json(final, root / output["final_summary"])

    figure_path = root / output["figure"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plot = metrics[
        metrics["evaluation_set"].isin(
            ["stage31_default_training", "calibrated_validation"]
        )
    ].copy()
    metric_order = list(
        config["target_tolerances"].keys()
    )
    plot["metric"] = pd.Categorical(
        plot["metric"],
        categories=metric_order,
        ordered=True,
    )
    plot = plot.sort_values(
        ["metric", "evaluation_set"]
    )

    x = np.arange(len(metric_order), dtype=float)
    width = 0.36
    plt.figure(figsize=(11.0, 5.4))

    for offset, label in [
        (-width / 2, "stage31_default_training"),
        (width / 2, "calibrated_validation"),
    ]:
        subset = (
            plot[plot["evaluation_set"] == label]
            .set_index("metric")
            .reindex(metric_order)
        )
        plt.bar(
            x + offset,
            subset["standardized_residual"].to_numpy(dtype=float),
            width=width,
            label=label,
        )

    plt.axhline(0.0, linewidth=1.0)
    plt.axhline(1.0, linewidth=1.0, linestyle="--")
    plt.axhline(-1.0, linewidth=1.0, linestyle="--")
    plt.xticks(
        x,
        metric_order,
        rotation=35,
        ha="right",
    )
    plt.ylabel("Standardized target residual")
    plt.title("Empirical target reproduction before and after DGM calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

    print("Calibration decision")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 32 calibration summary completed. "
        "No Candidate V9/V10 output or manuscript prose was changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
