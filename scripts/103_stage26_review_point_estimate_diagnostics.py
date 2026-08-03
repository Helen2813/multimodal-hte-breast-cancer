from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage26_v10_utils import (
    dataframe_console,
    load_json,
    project_root,
    read_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage26_v10_point_estimate_config.json"
    )
    output = config["output"]
    review = config["numerical_review"]

    print("=" * 128)
    print("STAGE 103 - REVIEW CANDIDATE V10 POINT-ESTIMATE DIAGNOSTICS")
    print("=" * 128)

    point = read_csv(
        root / output["point_estimate"]
    ).iloc[0]
    partitions = read_csv(
        root / output["partition_table"]
    )

    effect = float(
        point["aipw_ato_rmst_difference_days"]
    )
    finite = bool(np.isfinite(effect))
    in_bounds = bool(
        -float(config["estimator"]["horizon_days"])
        <= effect
        <= float(config["estimator"]["horizon_days"])
    )
    all_partitions = (
        len(partitions)
        == int(config["expected"]["partitions"])
    )
    all_partition_effects_finite = bool(
        np.isfinite(
            pd.to_numeric(
                partitions["estimate_days"],
                errors="coerce",
            ).to_numpy(dtype=float)
        ).all()
    )

    checks = pd.DataFrame([
        {
            "check": "all locked partitions completed",
            "observed": len(partitions),
            "threshold": config["expected"]["partitions"],
            "pass": all_partitions,
        },
        {
            "check": "all partition effects finite",
            "observed": all_partition_effects_finite,
            "threshold": True,
            "pass": all_partition_effects_finite,
        },
        {
            "check": "aggregated effect finite",
            "observed": finite,
            "threshold": True,
            "pass": finite,
        },
        {
            "check": "aggregated effect inside logical RMST bounds",
            "observed": effect,
            "threshold": (
                f"[-{config['estimator']['horizon_days']}, "
                f"{config['estimator']['horizon_days']}]"
            ),
            "pass": in_bounds,
        },
        {
            "check": "partition MCSE",
            "observed": float(
                point["partition_mcse_effect_days"]
            ),
            "threshold": review[
                "maximum_partition_mcse_days_for_automatic_bootstrap_recommendation"
            ],
            "pass": float(
                point["partition_mcse_effect_days"]
            )
            <= float(review[
                "maximum_partition_mcse_days_for_automatic_bootstrap_recommendation"
            ]),
        },
        {
            "check": "partition effect range",
            "observed": float(
                point["partition_range_effect_days"]
            ),
            "threshold": review[
                "maximum_partition_range_days_for_automatic_bootstrap_recommendation"
            ],
            "pass": float(
                point["partition_range_effect_days"]
            )
            <= float(review[
                "maximum_partition_range_days_for_automatic_bootstrap_recommendation"
            ]),
        },
        {
            "check": "maximum nuisance retry",
            "observed": int(
                point["maximum_nuisance_retry"]
            ),
            "threshold": review[
                "maximum_nuisance_retry_for_automatic_bootstrap_recommendation"
            ],
            "pass": int(
                point["maximum_nuisance_retry"]
            )
            <= int(review[
                "maximum_nuisance_retry_for_automatic_bootstrap_recommendation"
            ]),
        },
        {
            "check": "minimum censoring survival",
            "observed": float(
                point["minimum_G_min_raw"]
            ),
            "threshold": review[
                "minimum_G_min_raw"
            ],
            "pass": float(
                point["minimum_G_min_raw"]
            )
            >= float(review["minimum_G_min_raw"]),
        },
    ])

    hard_checks = checks.iloc[:4]
    numerical_checks = checks.iloc[4:]
    hard_pass = bool(hard_checks["pass"].all())
    numerical_pass = bool(
        numerical_checks["pass"].all()
    )

    if not hard_pass:
        status = (
            "CANDIDATE_V10_POINT_ESTIMATE_INVALID_"
            "DO_NOT_BOOTSTRAP"
        )
        bootstrap_recommendation = False
    elif numerical_pass:
        status = (
            "CANDIDATE_V10_POINT_ESTIMATE_COMPLETE_"
            "BOOTSTRAP_RECOMMENDED_AFTER_HUMAN_REVIEW"
        )
        bootstrap_recommendation = True
    else:
        status = (
            "CANDIDATE_V10_POINT_ESTIMATE_COMPLETE_"
            "NUMERICAL_REVIEW_REQUIRED_BEFORE_BOOTSTRAP"
        )
        bootstrap_recommendation = False

    v9_effect = None
    v9_path = root / config["source"]["v9_point_estimate"]
    if v9_path.exists():
        v9 = read_csv(v9_path)
        candidates = [
            column for column in v9.columns
            if "effect" in column.lower()
            or "difference" in column.lower()
            or "estimate" in column.lower()
        ]
        for column in candidates:
            values = pd.to_numeric(
                v9[column],
                errors="coerce",
            ).dropna()
            if len(values):
                value = float(values.iloc[0])
                if abs(value) <= float(
                    config["estimator"]["horizon_days"]
                ):
                    v9_effect = value
                    break

    decision = {
        "status": status,
        "protocol_id": config["expected"]["protocol_id"],
        "candidate_v10_point_estimate_days": effect,
        "diagnostic_if_interval_days": [
            float(point["diagnostic_if_ci_low_days"]),
            float(point["diagnostic_if_ci_high_days"]),
        ],
        "candidate_v9_locked_point_estimate_days": v9_effect,
        "v10_minus_v9_point_estimate_days": (
            effect - v9_effect
            if v9_effect is not None
            else None
        ),
        "bootstrap_recommended_by_numerical_checks": (
            bootstrap_recommendation
        ),
        "primary_publication_interval_available": False,
        "interpretation": (
            "The Stage 26 effect is a locked observational ATO-RMST "
            "point estimate in the sequencing-aware V10 population. "
            "The influence-function interval is diagnostic. No final "
            "benefit, harm, or statistical-significance claim is "
            "authorized before the fully refitted patient bootstrap."
        ),
        "next_action": (
            "Review this log and, if accepted, run a separate "
            "300-repetition fully refitted Candidate V10 patient bootstrap."
            if hard_pass
            else "Audit the point-estimate implementation before any bootstrap."
        ),
    }
    write_json(decision, root / output["decision"])
    checks.to_csv(
        root
        / output["table_dir"]
        / "s26_103_point_estimate_review_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Point-estimate review checks")
    print(dataframe_console(checks))
    print("\nStage 26 decision")
    print(json.dumps(decision, indent=2))

    if not hard_pass:
        raise RuntimeError(
            "Candidate V10 point estimate failed hard validity checks."
        )

    print(
        "\nPASS: Candidate V10 point-estimate review completed. "
        "Publication inference is not yet available."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
