#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage17_utils import (
    aggregate_loo_by_original_fold,
    aggregate_patient_scores,
    dataframe_console,
    ensure_stage17_dirs,
    load_stage17_config,
    markdown_table,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_stage17_dirs(root)
    cfg = load_stage17_config(root)
    rcfg = cfg["repeated_crossfit"]
    tables = root / "results/tables"
    local = root / "data/derived/stage17"

    estimates_path = tables / "68_repeated_crossfit_estimates_checkpoint.csv"
    scores_path = local / "68_primary_patient_scores_LOCAL_ONLY.csv"
    if not estimates_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Stage 68 checkpoint outputs are missing.")

    estimates = read_csv(estimates_path)
    scores = read_csv(scores_path)
    expected_repeats = int(rcfg["n_repeats"])
    if estimates["repeat"].nunique() != expected_repeats:
        raise RuntimeError(
            f"Expected {expected_repeats} completed repeats, found {estimates['repeat'].nunique()}."
        )

    bounded = estimates[estimates["prediction_constraint"] == "bounded_0_to_horizon"].copy()
    summary = (
        estimates.groupby(
            ["propensity_track", "g_min", "prediction_constraint"], as_index=False
        )
        .agg(
            repetitions=("repeat", "nunique"),
            mean_effect_days=("estimate_days", "mean"),
            median_effect_days=("estimate_days", "median"),
            sd_effect_days=("estimate_days", "std"),
            min_effect_days=("estimate_days", "min"),
            max_effect_days=("estimate_days", "max"),
            fraction_positive=("estimate_days", lambda s: float(np.mean(s > 0))),
            mean_if_se_days=("if_se_days", "mean"),
            median_loo_spread_days=("loo_effect_spread", "median"),
            maximum_loo_spread_days=("loo_effect_spread", "max"),
            median_pseudo_p99=("pseudo_p99", "median"),
            maximum_pseudo_max=("pseudo_max", "max"),
        )
    )
    summary["effect_range_days"] = summary["max_effect_days"] - summary["min_effect_days"]

    primary_g = float(rcfg["primary_g_min"])
    primary = bounded[np.isclose(bounded["g_min"], primary_g)].copy()
    primary_repeat = primary[
        [
            "repeat",
            "nominal_seed",
            "split_seed_used",
            "propensity_track",
            "estimate_days",
            "if_se_days",
            "if_ci_low_days",
            "if_ci_high_days",
            "direct_ato_ipw_effect_days",
            "fold_effect_spread",
            "loo_effect_spread",
            "pseudo_p99",
            "pseudo_max",
            "censor_log_loss",
            "censor_brier",
            "G_min_raw",
            "G_p01_raw",
        ]
    ].sort_values(["propensity_track", "repeat"])

    gmin_summary = (
        bounded.groupby(["propensity_track", "g_min"], as_index=False)
        .agg(
            repetitions=("repeat", "nunique"),
            mean_effect_days=("estimate_days", "mean"),
            median_effect_days=("estimate_days", "median"),
            sd_effect_days=("estimate_days", "std"),
            min_effect_days=("estimate_days", "min"),
            max_effect_days=("estimate_days", "max"),
            fraction_positive=("estimate_days", lambda s: float(np.mean(s > 0))),
            median_loo_spread_days=("loo_effect_spread", "median"),
        )
    )
    gmin_track = (
        gmin_summary.groupby("propensity_track", as_index=False)
        .agg(
            minimum_gmin_mean_effect_days=("mean_effect_days", "min"),
            maximum_gmin_mean_effect_days=("mean_effect_days", "max"),
            minimum_gmin_fraction_positive=("fraction_positive", "min"),
            maximum_gmin_fraction_positive=("fraction_positive", "max"),
        )
    )
    gmin_track["gmin_mean_effect_range_days"] = (
        gmin_track["maximum_gmin_mean_effect_days"]
        - gmin_track["minimum_gmin_mean_effect_days"]
    )
    gmin_track["gmin_mean_direction_consistent"] = (
        (gmin_track["minimum_gmin_mean_effect_days"] > 0)
        | (gmin_track["maximum_gmin_mean_effect_days"] < 0)
    )

    merge_keys = ["repeat", "propensity_track", "g_min"]
    b = estimates[estimates["prediction_constraint"] == "bounded_0_to_horizon"][
        merge_keys + ["estimate_days"]
    ].rename(columns={"estimate_days": "bounded_effect_days"})
    u = estimates[estimates["prediction_constraint"] == "unbounded"][
        merge_keys + ["estimate_days"]
    ].rename(columns={"estimate_days": "unbounded_effect_days"})
    paired = b.merge(u, on=merge_keys, how="inner", validate="one_to_one")
    paired["bounded_minus_unbounded_days"] = (
        paired["bounded_effect_days"] - paired["unbounded_effect_days"]
    )
    paired_summary = (
        paired.groupby(["propensity_track", "g_min"], as_index=False)
        .agg(
            paired_repetitions=("repeat", "nunique"),
            mean_bounding_shift_days=("bounded_minus_unbounded_days", "mean"),
            median_bounding_shift_days=("bounded_minus_unbounded_days", "median"),
            sd_bounding_shift_days=("bounded_minus_unbounded_days", "std"),
            max_abs_bounding_shift_days=(
                "bounded_minus_unbounded_days",
                lambda s: float(np.max(np.abs(s))),
            ),
            sign_agreement=(
                "bounded_effect_days",
                lambda s: float(
                    np.mean(
                        np.sign(s.to_numpy(float))
                        == np.sign(
                            paired.loc[s.index, "unbounded_effect_days"].to_numpy(float)
                        )
                    )
                ),
            ),
        )
    )

    prefix_rows: list[dict] = []
    prefix_loo_rows: list[pd.DataFrame] = []
    top_influence_rows: list[pd.DataFrame] = []
    max_repeat = expected_repeats
    prefixes = sorted({int(v) for v in rcfg["aggregation_prefixes"] if int(v) <= max_repeat})
    for track in rcfg["propensity_tracks"]:
        track_scores = scores[scores["propensity_track"] == track].copy()
        if track_scores["repeat"].nunique() != expected_repeats:
            raise RuntimeError(
                f"Patient-score checkpoint for {track} has "
                f"{track_scores['repeat'].nunique()} repeats, expected {expected_repeats}."
            )
        for k in prefixes:
            subset = track_scores[pd.to_numeric(track_scores["repeat"]) <= k]
            agg = aggregate_patient_scores(subset)
            loo = aggregate_loo_by_original_fold(agg["patient"])
            loo_spread = float(
                loo["aggregated_loo_effect_days"].max()
                - loo["aggregated_loo_effect_days"].min()
            )
            prefix_rows.append(
                {
                    "propensity_track": track,
                    "prefix_repeats": k,
                    "estimate_days": agg["estimate_days"],
                    "if_se_days": agg["if_se_days"],
                    "if_ci_low_days": agg["if_ci_low_days"],
                    "if_ci_high_days": agg["if_ci_high_days"],
                    "original_fold_loo_min_days": float(
                        loo["aggregated_loo_effect_days"].min()
                    ),
                    "original_fold_loo_max_days": float(
                        loo["aggregated_loo_effect_days"].max()
                    ),
                    "original_fold_loo_spread_days": loo_spread,
                }
            )
            loo.insert(0, "prefix_repeats", k)
            loo.insert(0, "propensity_track", track)
            prefix_loo_rows.append(loo)

            if k == max_repeat:
                top = agg["patient"].nlargest(25, "absolute_aggregated_influence").copy()
                top.insert(0, "propensity_track", track)
                top_influence_rows.append(top)

    prefix = pd.DataFrame(prefix_rows)
    prefix_loo = pd.concat(prefix_loo_rows, ignore_index=True)
    top_influence = pd.concat(top_influence_rows, ignore_index=True)

    block_size = int(rcfg["aggregation_block_size"])
    block_rows: list[dict] = []
    for track in rcfg["propensity_tracks"]:
        track_scores = scores[scores["propensity_track"] == track]
        block_id = 0
        for start in range(1, expected_repeats + 1, block_size):
            end = min(start + block_size - 1, expected_repeats)
            block_id += 1
            subset = track_scores[
                (pd.to_numeric(track_scores["repeat"]) >= start)
                & (pd.to_numeric(track_scores["repeat"]) <= end)
            ]
            agg = aggregate_patient_scores(subset)
            block_rows.append(
                {
                    "propensity_track": track,
                    "block_id": block_id,
                    "repeat_start": start,
                    "repeat_end": end,
                    "repeats_in_block": end - start + 1,
                    "estimate_days": agg["estimate_days"],
                    "if_se_days": agg["if_se_days"],
                    "if_ci_low_days": agg["if_ci_low_days"],
                    "if_ci_high_days": agg["if_ci_high_days"],
                }
            )
    blocks = pd.DataFrame(block_rows)

    leave_repeat_rows: list[dict] = []
    for track in rcfg["propensity_tracks"]:
        track_scores = scores[scores["propensity_track"] == track]
        for omitted in range(1, expected_repeats + 1):
            subset = track_scores[pd.to_numeric(track_scores["repeat"]) != omitted]
            agg = aggregate_patient_scores(subset)
            leave_repeat_rows.append(
                {
                    "propensity_track": track,
                    "omitted_repeat": omitted,
                    "estimate_days": agg["estimate_days"],
                }
            )
    leave_repeat = pd.DataFrame(leave_repeat_rows)
    leave_repeat_summary = (
        leave_repeat.groupby("propensity_track", as_index=False)
        .agg(
            min_leave_one_repeat_out_days=("estimate_days", "min"),
            max_leave_one_repeat_out_days=("estimate_days", "max"),
            sd_leave_one_repeat_out_days=("estimate_days", "std"),
        )
    )
    leave_repeat_summary["leave_one_repeat_out_range_days"] = (
        leave_repeat_summary["max_leave_one_repeat_out_days"]
        - leave_repeat_summary["min_leave_one_repeat_out_days"]
    )

    write_csv(summary, tables / "69_repeated_estimate_summary.csv")
    write_csv(primary_repeat, tables / "69_primary_repeat_estimates.csv")
    write_csv(gmin_summary, tables / "69_gmin_sensitivity_summary.csv")
    write_csv(gmin_track, tables / "69_gmin_track_diagnostics.csv")
    write_csv(paired, tables / "69_bounded_unbounded_paired_repeats.csv")
    write_csv(paired_summary, tables / "69_bounded_unbounded_paired_summary.csv")
    write_csv(prefix, tables / "69_repeated_score_prefix_convergence.csv")
    write_csv(prefix_loo, tables / "69_aggregated_original_fold_loo.csv")
    write_csv(blocks, tables / "69_repeated_score_block_stability.csv")
    write_csv(leave_repeat, tables / "69_leave_one_repeat_out_estimates.csv")
    write_csv(leave_repeat_summary, tables / "69_leave_one_repeat_out_summary.csv")
    write_csv(top_influence, tables / "69_aggregated_top_influence_deidentified.csv")

    write_text(
        f"""# Stage 17 repeated-score aggregation

## Repeated-estimate summary

{markdown_table(summary)}

## Primary G-min repeat estimates

{markdown_table(primary_repeat, max_rows=80)}

## G-min sensitivity

{markdown_table(gmin_summary)}

## Prefix convergence of the repeated-score estimator

{markdown_table(prefix)}

## Five-repeat block stability

{markdown_table(blocks)}

## Leave-one-repeat-out stability

{markdown_table(leave_repeat_summary)}

Repeated-score aggregation averages patient-level score numerators and target weights over the
prespecified nuisance partitions before computing one estimator. No split was selected based on
its effect estimate.
""",
        tables / "69_repeated_score_aggregation.md",
    )

    print("=" * 124)
    print("STAGE 69 — REPEATED-SCORE AGGREGATION AND STABILITY SUMMARIES")
    print("=" * 124)
    print("Repeated-estimate summary")
    print(dataframe_console(summary))
    print("\nAll primary G-min=0.10 bounded repeat estimates")
    print(dataframe_console(primary_repeat))
    print("\nG-min sensitivity summary")
    print(dataframe_console(gmin_summary))
    print("\nG-min track diagnostics")
    print(dataframe_console(gmin_track))
    print("\nBounded versus unbounded paired summary")
    print(dataframe_console(paired_summary))
    print("\nRepeated-score prefix convergence")
    print(dataframe_console(prefix))
    print("\nRepeated-score block stability")
    print(dataframe_console(blocks))
    print("\nAggregated leave-one-original-fold results")
    print(dataframe_console(prefix_loo))
    print("\nLeave-one-repeat-out summary")
    print(dataframe_console(leave_repeat_summary))
    print("\nTop aggregated influence rows, deidentified")
    print(dataframe_console(top_influence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
