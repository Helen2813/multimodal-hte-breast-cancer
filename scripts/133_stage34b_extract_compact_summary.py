from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            safe_json(data),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def main() -> int:
    root = Path.cwd().resolve()
    config = load_json(
        root / "stage34b_compact_summary_config.json"
    )
    source = config["source"]
    output = config["output"]
    expected = config["expected"]

    print("=" * 128)
    print("STAGE 133 - EXTRACT COMPACT STAGE 34 CONFIRMATORY SUMMARY")
    print("=" * 128)

    required_paths = {
        name: root / relative
        for name, relative in source.items()
    }
    missing = [
        str(path)
        for path in required_paths.values()
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing Stage 34 output files:\n"
            + "\n".join(missing)
        )

    manifest = load_json(required_paths["manifest"])
    final = load_json(required_paths["final_json"])

    if manifest["simulation_id"] != expected["simulation_id"]:
        raise RuntimeError("Unexpected Stage 34 simulation ID.")
    if final["simulation_id"] != expected["simulation_id"]:
        raise RuntimeError("Stage 34 final JSON ID mismatch.")

    summary = pd.read_csv(
        required_paths["summary_csv"],
        low_memory=False,
        keep_default_na=False,
    )
    if len(summary) != int(expected["summary_rows"]):
        raise RuntimeError(
            f"Expected {expected['summary_rows']} summary rows, "
            f"found {len(summary)}."
        )

    anchors = pd.read_csv(
        required_paths["anchor_checks"],
        low_memory=False,
        keep_default_na=False,
    )
    gates = pd.read_csv(
        required_paths["gates"],
        low_memory=False,
        keep_default_na=False,
    )

    numeric_columns = [
        "mean_primary_truth_days",
        "mean_estimate_days",
        "bias_days",
        "bias_mcse_days",
        "rmse_days",
        "primary_if_coverage",
        "coverage_mcse",
        "positive_ci_exclusion_rate",
        "positive_exclusion_mcse",
        "mean_weighted_chemo_smd",
        "mean_included_covariate_max_abs_weighted_smd",
        "mean_ato_ess_fraction_treated",
        "mean_ato_ess_fraction_control",
    ]
    for column in numeric_columns:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        )

    key = summary[
        (
            summary["sequencing_level"].isin(
                ["none", "empirical"]
            )
        )
        & (
            summary["effect_regime"].isin(
                ["true_zero", "observed_risk_benefit"]
            )
        )
    ].copy()

    key_columns = [
        "scenario_id",
        "sample_size",
        "sequencing_level",
        "effect_regime",
        "method",
        "mean_primary_truth_days",
        "mean_estimate_days",
        "bias_days",
        "bias_mcse_days",
        "rmse_days",
        "primary_if_coverage",
        "coverage_mcse",
        "positive_ci_exclusion_rate",
        "positive_exclusion_mcse",
        "mean_weighted_chemo_smd",
        "mean_included_covariate_max_abs_weighted_smd",
        "mean_ato_ess_fraction_treated",
        "mean_ato_ess_fraction_control",
    ]
    key = key[key_columns].sort_values(
        [
            "effect_regime",
            "sample_size",
            "sequencing_level",
            "method",
        ]
    ).reset_index(drop=True)
    write_csv(key, root / output["compact_csv"])

    valid = summary[
        summary["method"].isin(
            ["adjusted_full", "sequencing_aware"]
        )
    ]
    naive = summary[
        summary["method"] == "naive_full"
    ]

    failed_gate_rows = gates[
        ~(
            as_bool(gates["success_gate"])
            & as_bool(gates["balance_gate"])
            & (
                (~as_bool(gates["valid_method"]))
                | (
                    as_bool(gates["bias_gate"])
                    & as_bool(gates["coverage_gate"])
                )
            )
        )
    ].copy()

    compact = {
        "status": final["status"],
        "simulation_id": final["simulation_id"],
        "confirmatory_result_passed_locked_checks": bool(
            final["confirmatory_result_passed_locked_checks"]
        ),
        "valid_method_checks_passed": bool(
            final["valid_method_checks_passed"]
        ),
        "naive_mechanism_checks_passed": bool(
            final["naive_mechanism_checks_passed"]
        ),
        "empirical_anchor_checks_passed": bool(
            final["empirical_anchor_checks_passed"]
        ),
        "method_runs_expected": int(
            manifest["simulation"]["expected_method_runs"]
        ),
        "worst_valid_method_absolute_bias_days": float(
            valid["bias_days"].abs().max()
        ),
        "minimum_valid_method_coverage": float(
            valid["primary_if_coverage"].min()
        ),
        "maximum_valid_method_included_smd": float(
            valid[
                "mean_included_covariate_max_abs_weighted_smd"
            ].max()
        ),
        "maximum_naive_absolute_weighted_chemo_smd": float(
            naive["mean_weighted_chemo_smd"].abs().max()
        ),
        "maximum_naive_positive_exclusion_rate_under_true_zero": float(
            naive.loc[
                naive["effect_regime"] == "true_zero",
                "positive_ci_exclusion_rate",
            ].max()
        ),
        "naive_mechanism_checks": final[
            "naive_mechanism_checks"
        ],
        "failed_locked_gate_rows": (
            failed_gate_rows.to_dict("records")
        ),
        "empirical_anchor_checks": (
            anchors.to_dict("records")
        ),
        "key_result_rows": key.to_dict("records"),
        "boundary": config["boundary"],
    }
    write_json(compact, root / output["compact_json"])

    print("Stage 34 top-level decision")
    print(
        json.dumps(
            {
                "status": compact["status"],
                "simulation_id": compact["simulation_id"],
                "confirmatory_result_passed_locked_checks": (
                    compact[
                        "confirmatory_result_passed_locked_checks"
                    ]
                ),
                "valid_method_checks_passed": compact[
                    "valid_method_checks_passed"
                ],
                "naive_mechanism_checks_passed": compact[
                    "naive_mechanism_checks_passed"
                ],
                "empirical_anchor_checks_passed": compact[
                    "empirical_anchor_checks_passed"
                ],
                "worst_valid_method_absolute_bias_days": compact[
                    "worst_valid_method_absolute_bias_days"
                ],
                "minimum_valid_method_coverage": compact[
                    "minimum_valid_method_coverage"
                ],
                "maximum_naive_positive_exclusion_rate_under_true_zero": (
                    compact[
                        "maximum_naive_positive_exclusion_rate_under_true_zero"
                    ]
                ),
            },
            indent=2,
        )
    )

    print("\nKey confirmatory result rows")
    print(key.to_string(index=False))

    print("\nEmpirical anchor checks")
    print(anchors.to_string(index=False))

    print("\nFailed locked gate rows")
    if len(failed_gate_rows):
        print(failed_gate_rows.to_string(index=False))
    else:
        print("None")

    print(
        "\nPASS: compact Stage 34 summary extracted without "
        "rerunning any simulation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
