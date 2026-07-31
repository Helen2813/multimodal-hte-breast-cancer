#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage17_utils import (
    dataframe_console,
    ensure_stage17_dirs,
    load_stage17_config,
    markdown_table,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def one_row(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    out = df.loc[mask]
    if len(out) != 1:
        raise RuntimeError(f"Expected one row for {label}, found {len(out)}")
    return out.iloc[0]


def main() -> int:
    root = project_root()
    ensure_stage17_dirs(root)
    cfg = load_stage17_config(root)
    rcfg = cfg["repeated_crossfit"]
    thresholds = cfg["decision_thresholds"]
    design = cfg["design"]
    tables = root / "results/tables"

    summary = read_csv(tables / "69_repeated_estimate_summary.csv")
    primary_repeat = read_csv(tables / "69_primary_repeat_estimates.csv")
    gtrack = read_csv(tables / "69_gmin_track_diagnostics.csv")
    prefix = read_csv(tables / "69_repeated_score_prefix_convergence.csv")
    leave_repeat = read_csv(tables / "69_leave_one_repeat_out_summary.csv")

    primary_track = str(rcfg["primary_propensity_track"])
    primary_g = float(rcfg["primary_g_min"])
    primary_row = one_row(
        summary,
        (summary["propensity_track"] == primary_track)
        & np.isclose(summary["g_min"], primary_g)
        & (summary["prediction_constraint"] == "bounded_0_to_horizon"),
        "primary repeated-estimate summary",
    )
    g_row = one_row(
        gtrack,
        gtrack["propensity_track"] == primary_track,
        "primary G-min diagnostics",
    )

    final_rows = prefix[prefix["prefix_repeats"] == int(rcfg["n_repeats"])].copy()
    final_primary = one_row(
        final_rows,
        final_rows["propensity_track"] == primary_track,
        "final primary repeated-score estimate",
    )
    frozen_final = one_row(
        final_rows,
        final_rows["propensity_track"] == "frozen_stage30",
        "final frozen repeated-score estimate",
    )

    def prefix_value(k: int, col: str = "estimate_days") -> float:
        row = one_row(
            prefix,
            (prefix["propensity_track"] == primary_track)
            & (prefix["prefix_repeats"] == k),
            f"primary prefix {k}",
        )
        return float(row[col])

    theta10 = prefix_value(10)
    theta20 = prefix_value(20)
    theta30 = prefix_value(30)
    shift_10_30 = abs(theta10 - theta30)
    shift_20_30 = abs(theta20 - theta30)
    track_shift = abs(float(final_primary["estimate_days"]) - float(frozen_final["estimate_days"]))

    original_loo = float(design["original_stage16_bounded_loo_spread_days"])
    final_loo = float(final_primary["original_fold_loo_spread_days"])
    loo_reduction = 1.0 - final_loo / original_loo if original_loo > 0 else np.nan

    fraction_positive = float(primary_row["fraction_positive"])
    split_sd = float(primary_row["sd_effect_days"])
    split_range = float(primary_row["effect_range_days"])
    gmin_range = float(g_row["gmin_mean_effect_range_days"])
    gmin_direction_consistent = bool(
        float(g_row["minimum_gmin_mean_effect_days"]) > 0
        or float(g_row["maximum_gmin_mean_effect_days"]) < 0
    )
    completed_repeats = int(primary_row["repetitions"])

    checks = pd.DataFrame(
        [
            {
                "check": "all_prespecified_repeats_complete",
                "observed": completed_repeats,
                "threshold_or_expected": int(rcfg["n_repeats"]),
                "pass": completed_repeats == int(rcfg["n_repeats"]),
            },
            {
                "check": "fraction_positive_repeats",
                "observed": fraction_positive,
                "threshold_or_expected": thresholds["minimum_fraction_positive_repeats"],
                "pass": fraction_positive >= float(thresholds["minimum_fraction_positive_repeats"]),
            },
            {
                "check": "between_split_sd_days",
                "observed": split_sd,
                "threshold_or_expected": thresholds["maximum_between_split_sd_days"],
                "pass": split_sd <= float(thresholds["maximum_between_split_sd_days"]),
            },
            {
                "check": "between_split_range_days",
                "observed": split_range,
                "threshold_or_expected": thresholds["maximum_between_split_range_days"],
                "pass": split_range <= float(thresholds["maximum_between_split_range_days"]),
            },
            {
                "check": "gmin_mean_direction_consistent",
                "observed": gmin_direction_consistent,
                "threshold_or_expected": True,
                "pass": gmin_direction_consistent,
            },
            {
                "check": "gmin_mean_effect_range_days",
                "observed": gmin_range,
                "threshold_or_expected": thresholds["maximum_gmin_mean_range_days"],
                "pass": gmin_range <= float(thresholds["maximum_gmin_mean_range_days"]),
            },
            {
                "check": "prefix_20_vs_30_shift_days",
                "observed": shift_20_30,
                "threshold_or_expected": thresholds["maximum_abs_prefix_20_vs_30_shift_days"],
                "pass": shift_20_30 <= float(thresholds["maximum_abs_prefix_20_vs_30_shift_days"]),
            },
            {
                "check": "prefix_10_vs_30_shift_days",
                "observed": shift_10_30,
                "threshold_or_expected": thresholds["maximum_abs_prefix_10_vs_30_shift_days"],
                "pass": shift_10_30 <= float(thresholds["maximum_abs_prefix_10_vs_30_shift_days"]),
            },
            {
                "check": "frozen_vs_refitted_aggregated_shift_days",
                "observed": track_shift,
                "threshold_or_expected": thresholds["maximum_abs_frozen_vs_refitted_aggregated_shift_days"],
                "pass": track_shift <= float(thresholds["maximum_abs_frozen_vs_refitted_aggregated_shift_days"]),
            },
            {
                "check": "original_fold_loo_spread_reduction_fraction",
                "observed": loo_reduction,
                "threshold_or_expected": thresholds["minimum_original_fold_loo_spread_reduction_fraction"],
                "pass": loo_reduction
                >= float(thresholds["minimum_original_fold_loo_spread_reduction_fraction"]),
            },
        ]
    )

    core_instability = checks[checks["check"].isin(
        [
            "all_prespecified_repeats_complete",
            "fraction_positive_repeats",
            "between_split_sd_days",
            "between_split_range_days",
        ]
    )]
    sensitivity_checks = checks[~checks.index.isin(core_instability.index)]

    if not bool(core_instability["pass"].all()):
        decision = "NUISANCE_SPLIT_RANDOMNESS_MATERIALLY_UNSTABLE_HOLD_PUBLICATION_BOOTSTRAP"
        bootstrap_pilot_allowed = False
        recommended_next_step = (
            "Do not run a patient bootstrap. Reframe Paper A around reliability limits, "
            "influence, censoring-tail sensitivity, and estimand dependence."
        )
    elif not bool(sensitivity_checks["pass"].all()):
        decision = "DIRECTION_STABLE_BUT_CENSORING_OR_COMPOSITION_SENSITIVE_HOLD_PUBLICATION_BOOTSTRAP"
        bootstrap_pilot_allowed = False
        recommended_next_step = (
            "Resolve the failed sensitivity gates or prespecify a narrower robust estimand. "
            "Do not start the full publication bootstrap."
        )
    else:
        decision = "REPEATED_CROSSFIT_STABILIZES_ESTIMATE_PROCEED_TO_BOOTSTRAP_PILOT"
        bootstrap_pilot_allowed = True
        recommended_next_step = (
            "Lock the repeated-cross-fit estimator specification and run a small, checkpointed "
            "patient-level bootstrap pilot before authorizing the full publication bootstrap."
        )

    decision_row = pd.DataFrame(
        [
            {
                "stage17_decision": decision,
                "protocol_status": cfg["protocol_status"],
                "primary_propensity_track": primary_track,
                "primary_g_min": primary_g,
                "completed_repeats": completed_repeats,
                "repeat_mean_effect_days": float(primary_row["mean_effect_days"]),
                "repeat_median_effect_days": float(primary_row["median_effect_days"]),
                "repeat_sd_effect_days": split_sd,
                "repeat_min_effect_days": float(primary_row["min_effect_days"]),
                "repeat_max_effect_days": float(primary_row["max_effect_days"]),
                "repeat_fraction_positive": fraction_positive,
                "aggregated_30_repeat_effect_days": float(final_primary["estimate_days"]),
                "aggregated_30_repeat_if_se_days": float(final_primary["if_se_days"]),
                "aggregated_30_repeat_ci_low_days": float(final_primary["if_ci_low_days"]),
                "aggregated_30_repeat_ci_high_days": float(final_primary["if_ci_high_days"]),
                "frozen_30_repeat_effect_days": float(frozen_final["estimate_days"]),
                "frozen_vs_refitted_shift_days": track_shift,
                "prefix_10_vs_30_shift_days": shift_10_30,
                "prefix_20_vs_30_shift_days": shift_20_30,
                "gmin_mean_effect_range_days": gmin_range,
                "original_stage16_loo_spread_days": original_loo,
                "aggregated_original_fold_loo_spread_days": final_loo,
                "loo_spread_reduction_fraction": loo_reduction,
                "publication_bootstrap_locked": True,
                "bootstrap_pilot_allowed": bootstrap_pilot_allowed,
                "recommended_next_step": recommended_next_step,
                "paper_claim": (
                    "Treatment-effect magnitude is evaluated across time-zero, target-weighting, "
                    "censoring, outcome-nuisance, and repeated cross-fitting choices; no split is "
                    "selected for a favorable estimate."
                ),
            }
        ]
    )

    write_csv(checks, tables / "70_stage17_decision_checks.csv")
    write_csv(decision_row, tables / "70_stage17_decision.csv")

    report = f"""# Stage 17 decision

**Decision:** `{decision}`  
**Protocol status:** `{cfg['protocol_status']}`  
**Full publication bootstrap locked:** `True`  
**Small bootstrap pilot allowed:** `{bootstrap_pilot_allowed}`

## Primary repeated-cross-fit result

{markdown_table(decision_row)}

## Prespecified decision gates

{markdown_table(checks)}

## Interpretation

The repeated-cross-fit distribution quantifies nuisance-partition randomness. The repeated-score
estimator averages patient-level score contributions over every prespecified partition; no favorable
split is selected. The influence-function interval remains diagnostic until a patient-level bootstrap
pilot confirms that nuisance refitting and sampling uncertainty are adequately represented.

## Recommended next step

{recommended_next_step}
"""
    write_text(report, tables / "70_stage17_decision.md")

    print("=" * 124)
    print("STAGE 70 — STAGE 17 DECISION")
    print("=" * 124)
    print("Decision summary")
    print(dataframe_console(decision_row))
    print("\nPrespecified decision checks")
    print(dataframe_console(checks))
    print("\nFull decision report")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
