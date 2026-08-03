from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage28_v10_gmin_utils import (
    dataframe_console,
    load_json,
    project_root,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage28_v10_gmin_sensitivity_config.json"
    )
    output = config["output"]

    print("=" * 128)
    print("STAGE 111 - SUMMARIZE G-MIN SENSITIVITY")
    print("=" * 128)

    summary = pd.read_csv(
        root / output["summary_table"],
        low_memory=False,
    ).sort_values("g_min")

    g_values = pd.to_numeric(
        summary["g_min"],
        errors="raise",
    ).to_numpy(dtype=float)
    estimates = pd.to_numeric(
        summary["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    partition_min = pd.to_numeric(
        summary["partition_min_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    partition_max = pd.to_numeric(
        summary["partition_max_days"],
        errors="raise",
    ).to_numpy(dtype=float)

    all_positive = bool((estimates > 0).all())
    direction_consistent = bool(
        np.sign(estimates).min() == np.sign(estimates).max()
    )
    maximum_absolute_change = float(
        np.max(
            np.abs(
                pd.to_numeric(
                    summary["difference_from_primary_days"],
                    errors="raise",
                ).to_numpy(dtype=float)
            )
        )
    )

    final = {
        "status": "STAGE28_GMIN_SENSITIVITY_COMPLETE",
        "sensitivity_id": load_json(
            root / output["calculation_manifest"]
        )["sensitivity_id"],
        "primary_g_min": config["sensitivity"]["primary_value"],
        "post_hoc_g_min_values": config["sensitivity"][
            "post_hoc_values"
        ],
        "all_point_estimates_positive": all_positive,
        "direction_consistent_across_g_min_values": (
            direction_consistent
        ),
        "maximum_absolute_change_from_primary_days": (
            maximum_absolute_change
        ),
        "results": summary.to_dict("records"),
        "interpretation_boundary": (
            "This is a post hoc point-estimate-only diagnostic "
            "sensitivity. It does not replace the locked G-min=0.10 "
            "primary analysis or its 300-repetition bootstrap interval."
        ),
    }
    write_json(
        final,
        root / output["table_dir"]
        / "s28_111_gmin_sensitivity_final.json",
    )

    figure_path = root / output["figure"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    lower = estimates - partition_min
    upper = partition_max - estimates
    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(
        g_values,
        estimates,
        yerr=np.vstack([lower, upper]),
        fmt="o-",
        capsize=4,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Censoring-survival truncation G-min")
    plt.ylabel("ATO RMST difference (days)")
    plt.title("Candidate V10 censoring-truncation sensitivity")
    plt.xticks(g_values)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

    print("Final G-min sensitivity table")
    print(dataframe_console(summary))
    print("\nFinal diagnostic summary")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 28 summary and figure completed. "
        "No bootstrap or manuscript text was generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
