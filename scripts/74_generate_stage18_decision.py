#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage18_utils import (
    dataframe_console,
    ensure_stage18_dirs,
    load_stage18_config,
    markdown_table,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_stage18_dirs(root)
    cfg = load_stage18_config(root)
    thresholds = cfg["decision_thresholds"]
    tables = root / "results/tables"

    summary = read_csv(tables / "73_bootstrap_pilot_summary.csv").iloc[0]
    prefix = read_csv(tables / "73_bootstrap_prefix_stability.csv")
    distribution = read_csv(tables / "73_bootstrap_pilot_distribution.csv")

    prefix_shift = float("nan")
    if {20, 30}.issubset(set(prefix["prefix_repetitions"].astype(int))):
        theta20 = float(prefix.loc[prefix["prefix_repetitions"] == 20, "mean_effect_days"].iloc[0])
        theta30 = float(prefix.loc[prefix["prefix_repetitions"] == 30, "mean_effect_days"].iloc[0])
        prefix_shift = abs(theta20 - theta30)

    explosion_threshold = float(thresholds["numerical_explosion_abs_effect_days"])
    explosion_fraction = float(
        np.mean(np.abs(pd.to_numeric(distribution["aggregated_effect_days"], errors="coerce")) > explosion_threshold)
    )

    checks = pd.DataFrame([
        {
            "check": "minimum_successful_repetitions",
            "observed": int(summary["successful_repetitions"]),
            "threshold_or_expected": int(thresholds["minimum_successful_repetitions"]),
            "pass": int(summary["successful_repetitions"]) >= int(thresholds["minimum_successful_repetitions"]),
        },
        {
            "check": "minimum_success_rate",
            "observed": float(summary["success_rate"]),
            "threshold_or_expected": float(thresholds["minimum_success_rate"]),
            "pass": float(summary["success_rate"]) >= float(thresholds["minimum_success_rate"]),
        },
        {
            "check": "median_within_bootstrap_partition_sd_days",
            "observed": float(summary["median_partition_sd_days"]),
            "threshold_or_expected": float(thresholds["maximum_median_within_bootstrap_partition_sd_days"]),
            "pass": float(summary["median_partition_sd_days"]) <= float(thresholds["maximum_median_within_bootstrap_partition_sd_days"]),
        },
        {
            "check": "p95_within_bootstrap_partition_sd_days",
            "observed": float(summary["p95_partition_sd_days"]),
            "threshold_or_expected": float(thresholds["maximum_p95_within_bootstrap_partition_sd_days"]),
            "pass": float(summary["p95_partition_sd_days"]) <= float(thresholds["maximum_p95_within_bootstrap_partition_sd_days"]),
        },
        {
            "check": "prefix_20_vs_30_mean_shift_days",
            "observed": prefix_shift,
            "threshold_or_expected": float(thresholds["maximum_prefix_20_vs_30_mean_shift_days"]),
            "pass": bool(np.isfinite(prefix_shift)) and prefix_shift <= float(thresholds["maximum_prefix_20_vs_30_mean_shift_days"]),
        },
        {
            "check": "numerical_explosion_fraction",
            "observed": explosion_fraction,
            "threshold_or_expected": float(thresholds["maximum_numerical_explosion_fraction"]),
            "pass": explosion_fraction <= float(thresholds["maximum_numerical_explosion_fraction"]),
        },
    ])

    numerical_feasible = bool(checks["pass"].all())
    pilot_ci_low = float(summary["pilot_percentile_ci_low_days"])
    pilot_ci_high = float(summary["pilot_percentile_ci_high_days"])
    fraction_positive = float(summary["fraction_positive"])

    if not numerical_feasible:
        decision = "BOOTSTRAP_PIPELINE_NUMERICALLY_UNSTABLE_DO_NOT_SCALE"
        next_step = (
            "Do not run the 300-repetition publication bootstrap. Inspect failed repetitions, "
            "grouped folds, nuisance-model degeneracy, and influential pseudo-outcomes."
        )
        authorize_after_lock = False
        claim_status = "Reliability-only; no primary treatment-effect magnitude should be locked."
    else:
        decision = "BOOTSTRAP_PIPELINE_FEASIBLE_PROCEED_TO_PROTOCOL_LOCK_AND_FULL_BOOTSTRAP"
        authorize_after_lock = True
        if pilot_ci_low > 0:
            claim_status = (
                "Pilot resampling preserves a positive direction, but the 30-repetition interval "
                "is not final inference."
            )
        elif fraction_positive >= 0.80:
            claim_status = (
                "Positive direction is frequent under resampling, while sampling uncertainty remains "
                "large; use a reliability-first claim until the full bootstrap is complete."
            )
        else:
            claim_status = (
                "Sampling uncertainty materially affects direction; proceed only to quantify uncertainty, "
                "not to make an efficacy claim."
            )
        next_step = (
            "Freeze the CANDIDATE_V8 estimator specification, record hashes, and run the full "
            "300-repetition patient bootstrap with five grouped repeated-cross-fit partitions per repetition. "
            "Do not change the estimator after inspecting the full bootstrap distribution."
        )

    decision_row = pd.DataFrame([
        {
            "stage18_decision": decision,
            "protocol_status": cfg["protocol_status"],
            "successful_repetitions": int(summary["successful_repetitions"]),
            "success_rate": float(summary["success_rate"]),
            "stage17_original_effect_days": float(summary["stage17_original_repeated_crossfit_effect_days"]),
            "bootstrap_pilot_mean_days": float(summary["bootstrap_mean_days"]),
            "bootstrap_pilot_sd_days": float(summary["bootstrap_sd_days"]),
            "bootstrap_pilot_ci_low_days": pilot_ci_low,
            "bootstrap_pilot_ci_high_days": pilot_ci_high,
            "bootstrap_pilot_fraction_positive": fraction_positive,
            "median_partition_sd_days": float(summary["median_partition_sd_days"]),
            "p95_partition_sd_days": float(summary["p95_partition_sd_days"]),
            "prefix_20_vs_30_mean_shift_days": prefix_shift,
            "numerical_explosion_fraction": explosion_fraction,
            "full_publication_bootstrap_still_locked": True,
            "full_bootstrap_authorized_after_protocol_lock": authorize_after_lock,
            "claim_status": claim_status,
            "recommended_next_step": next_step,
        }
    ])

    write_csv(checks, tables / "74_stage18_decision_checks.csv")
    write_csv(decision_row, tables / "74_stage18_decision.csv")
    report = f"""# Stage 18 decision

**Decision:** `{decision}`  
**Protocol status:** `{cfg['protocol_status']}`  
**Full publication bootstrap currently locked:** `True`  
**Authorized after protocol lock:** `{authorize_after_lock}`

## Pilot result

{markdown_table(decision_row)}

## Numerical feasibility checks

{markdown_table(checks)}

## Interpretation

{claim_status}

The 30-repetition percentile interval is a computational pilot and is not the final confidence
interval. Statistical significance is not a pilot feasibility gate.

## Recommended next step

{next_step}
"""
    write_text(report, tables / "74_stage18_decision.md")

    print("=" * 124)
    print("STAGE 74 - STAGE 18 DECISION")
    print("=" * 124)
    print("Decision summary")
    print(dataframe_console(decision_row))
    print("\nFeasibility checks")
    print(dataframe_console(checks))
    print("\nFull decision report")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
