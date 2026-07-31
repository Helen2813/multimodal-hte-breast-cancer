#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage19_utils import (
    dataframe_console,
    ensure_stage19_dirs,
    load_stage19_config,
    markdown_table,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_stage19_dirs(root)
    cfg = load_stage19_config(root)
    thresholds = cfg["decision_thresholds"]
    tables = root / "results/tables"

    summary = read_csv(tables / "77_stage19_stabilization_summary.csv").iloc[0]
    prefix = read_csv(tables / "77_bootstrap_prefix_effects.csv")
    extension = read_csv(tables / "76_extended_bootstrap_partitions_checkpoint.csv")
    errors = read_csv(tables / "76_extended_bootstrap_partition_errors.csv")

    completed_repetitions = int(
        extension.groupby("bootstrap_repetition")["partition"].nunique().eq(15).sum()
    )
    prefix20 = prefix[prefix["prefix_partitions"].astype(int) == 20]
    explosion_threshold = float(thresholds["numerical_explosion_abs_effect_days"])
    explosion_fraction = float(
        np.mean(
            np.abs(
                pd.to_numeric(prefix20["aggregated_effect_days"], errors="coerce")
            )
            > explosion_threshold
        )
    )

    checks = pd.DataFrame(
        [
            {
                "check": "completed_bootstrap_repetitions_with_20_partitions",
                "observed": completed_repetitions,
                "threshold_or_expected": int(
                    thresholds["minimum_completed_bootstrap_repetitions"]
                ),
                "pass": completed_repetitions
                >= int(thresholds["minimum_completed_bootstrap_repetitions"]),
            },
            {
                "check": "stage19_partition_fit_errors",
                "observed": len(errors),
                "threshold_or_expected": 0,
                "pass": len(errors) == 0,
            },
            {
                "check": "median_absolute_10_to_20_shift_days",
                "observed": float(summary["median_absolute_10_to_20_shift_days"]),
                "threshold_or_expected": float(
                    thresholds["maximum_median_absolute_10_to_20_shift_days"]
                ),
                "pass": float(summary["median_absolute_10_to_20_shift_days"])
                <= float(thresholds["maximum_median_absolute_10_to_20_shift_days"]),
            },
            {
                "check": "p95_absolute_10_to_20_shift_days",
                "observed": float(summary["p95_absolute_10_to_20_shift_days"]),
                "threshold_or_expected": float(
                    thresholds["maximum_p95_absolute_10_to_20_shift_days"]
                ),
                "pass": float(summary["p95_absolute_10_to_20_shift_days"])
                <= float(thresholds["maximum_p95_absolute_10_to_20_shift_days"]),
            },
            {
                "check": "median_mcse_at_20_days",
                "observed": float(summary["median_mcse_at_20_days"]),
                "threshold_or_expected": float(
                    thresholds["maximum_median_mcse_at_20_days"]
                ),
                "pass": float(summary["median_mcse_at_20_days"])
                <= float(thresholds["maximum_median_mcse_at_20_days"]),
            },
            {
                "check": "p95_mcse_at_20_days",
                "observed": float(summary["p95_mcse_at_20_days"]),
                "threshold_or_expected": float(
                    thresholds["maximum_p95_mcse_at_20_days"]
                ),
                "pass": float(summary["p95_mcse_at_20_days"])
                <= float(thresholds["maximum_p95_mcse_at_20_days"]),
            },
            {
                "check": "absolute_15_vs_20_distribution_mean_shift_days",
                "observed": float(
                    summary["absolute_15_vs_20_distribution_mean_shift_days"]
                ),
                "threshold_or_expected": float(
                    thresholds["maximum_15_vs_20_distribution_mean_shift_days"]
                ),
                "pass": float(
                    summary["absolute_15_vs_20_distribution_mean_shift_days"]
                )
                <= float(thresholds["maximum_15_vs_20_distribution_mean_shift_days"]),
            },
            {
                "check": "numerical_explosion_fraction",
                "observed": explosion_fraction,
                "threshold_or_expected": float(
                    thresholds["maximum_numerical_explosion_fraction"]
                ),
                "pass": explosion_fraction
                <= float(thresholds["maximum_numerical_explosion_fraction"]),
            },
        ]
    )

    stabilized = bool(checks["pass"].all())
    if stabilized:
        decision = "INNER_CROSSFIT_STABILIZED_PREPARE_CANDIDATE_V9_PROTOCOL_LOCK"
        authorize = True
        next_step = (
            "Freeze the 20-partition inner repeated-cross-fit estimator, record code/config/input hashes, "
            "and then run the 300-repetition patient bootstrap. The final bootstrap must use the locked "
            "20 partition seeds and may not be modified after distribution inspection."
        )
        claim = (
            "The estimator is computationally stable after repeated nuisance-partition averaging. "
            "Treatment-effect magnitude remains subject to sampling uncertainty and will not be claimed "
            "until the locked full bootstrap is complete."
        )
    else:
        decision = "INNER_CROSSFIT_NOT_STABILIZED_USE_REPEATED_SCORE_IF_INFERENCE"
        authorize = False
        next_step = (
            "Do not scale the refitted patient bootstrap. Lock the 30-partition repeated-score point estimator "
            "and use influence-function or multiplier-bootstrap inference as primary, with the Stage 18 patient "
            "bootstrap reported only as a finite-sample sensitivity analysis."
        )
        claim = (
            "Patient resampling exposes substantial nuisance-partition Monte Carlo uncertainty. "
            "The paper should use a reliability-first estimand and avoid a primary efficacy magnitude claim."
        )

    decision_row = pd.DataFrame(
        [
            {
                "stage19_decision": decision,
                "protocol_status": cfg["protocol_status"],
                "completed_bootstrap_repetitions": completed_repetitions,
                "partitions_per_bootstrap": int(cfg["extension"]["target_total_partitions"]),
                "prefix20_mean_effect_days": float(summary["prefix20_mean_effect_days"]),
                "prefix20_median_effect_days": float(summary["prefix20_median_effect_days"]),
                "prefix20_sd_effect_days": float(summary["prefix20_sd_effect_days"]),
                "prefix20_percentile_ci_low_days": float(
                    summary["prefix20_percentile_ci_low_days"]
                ),
                "prefix20_percentile_ci_high_days": float(
                    summary["prefix20_percentile_ci_high_days"]
                ),
                "prefix20_fraction_positive": float(summary["prefix20_fraction_positive"]),
                "median_absolute_10_to_20_shift_days": float(
                    summary["median_absolute_10_to_20_shift_days"]
                ),
                "p95_absolute_10_to_20_shift_days": float(
                    summary["p95_absolute_10_to_20_shift_days"]
                ),
                "median_mcse_at_20_days": float(summary["median_mcse_at_20_days"]),
                "p95_mcse_at_20_days": float(summary["p95_mcse_at_20_days"]),
                "full_publication_bootstrap_locked": True,
                "full_bootstrap_authorized_after_protocol_lock": authorize,
                "claim_status": claim,
                "recommended_next_step": next_step,
            }
        ]
    )

    write_csv(checks, tables / "78_stage19_decision_checks.csv")
    write_csv(decision_row, tables / "78_stage19_decision.csv")
    report = f"""# Stage 19 decision

**Decision:** `{decision}`  
**Protocol status:** `{cfg['protocol_status']}`  
**Full publication bootstrap currently locked:** `True`  
**Authorized after protocol lock:** `{authorize}`

## Stabilized pilot result

{markdown_table(decision_row)}

## Convergence gates

{markdown_table(checks)}

## Interpretation

{claim}

## Recommended next step

{next_step}
"""
    write_text(report, tables / "78_stage19_decision.md")

    print("=" * 124)
    print("STAGE 78 - STAGE 19 DECISION")
    print("=" * 124)
    print("Decision summary")
    print(dataframe_console(decision_row))
    print("\nConvergence checks")
    print(dataframe_console(checks))
    print("\nFull decision report")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
