#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage19_utils import (
    dataframe_console,
    denominator_weighted_effect,
    ensure_stage19_dirs,
    load_stage19_config,
    project_root,
    read_csv,
    write_csv,
)


def main() -> int:
    root = project_root()
    ensure_stage19_dirs(root)
    cfg = load_stage19_config(root)
    ext = cfg["extension"]
    tables = root / "results/tables"

    stage18_parts = read_csv(tables / "72_bootstrap_pilot_partitions_checkpoint.csv")
    stage18_reps = read_csv(tables / "72_bootstrap_pilot_repetitions_checkpoint.csv")
    stage19_parts_path = tables / "76_extended_bootstrap_partitions_checkpoint.csv"
    if not stage19_parts_path.exists():
        raise FileNotFoundError(stage19_parts_path)
    stage19_parts = read_csv(stage19_parts_path)

    all_parts = pd.concat([stage18_parts, stage19_parts], ignore_index=True)
    all_parts["bootstrap_repetition"] = pd.to_numeric(
        all_parts["bootstrap_repetition"], errors="raise"
    ).astype(int)
    all_parts["partition"] = pd.to_numeric(all_parts["partition"], errors="raise").astype(int)
    all_parts = (
        all_parts.sort_values(["bootstrap_repetition", "partition"])
        .drop_duplicates(["bootstrap_repetition", "partition"], keep="last")
        .reset_index(drop=True)
    )

    prefixes = [int(x) for x in ext["aggregation_prefixes"]]
    target_reps = int(ext["bootstrap_repetitions"])
    target_partitions = int(ext["target_total_partitions"])
    prefix_rows = []
    convergence_rows = []

    for repetition in range(1, target_reps + 1):
        group = all_parts[all_parts["bootstrap_repetition"] == repetition].copy()
        observed_partitions = set(group["partition"].astype(int))
        expected = set(range(1, target_partitions + 1))
        missing = sorted(expected - observed_partitions)
        if missing:
            raise RuntimeError(
                f"Bootstrap repetition {repetition} is missing partitions: {missing}"
            )
        estimates_by_prefix = {}
        for prefix in prefixes:
            subset = group[group["partition"] <= prefix].copy()
            effect = denominator_weighted_effect(subset)
            partition_estimates = pd.to_numeric(
                subset["estimate_days"], errors="raise"
            ).to_numpy(float)
            partition_sd = float(np.std(partition_estimates, ddof=1))
            mcse = float(partition_sd / np.sqrt(prefix))
            estimates_by_prefix[prefix] = effect
            prefix_rows.append(
                {
                    "bootstrap_repetition": repetition,
                    "prefix_partitions": prefix,
                    "aggregated_effect_days": effect,
                    "partition_mean_effect_days": float(np.mean(partition_estimates)),
                    "partition_median_effect_days": float(np.median(partition_estimates)),
                    "partition_sd_effect_days": partition_sd,
                    "partition_mcse_days": mcse,
                    "partition_min_effect_days": float(np.min(partition_estimates)),
                    "partition_max_effect_days": float(np.max(partition_estimates)),
                    "minimum_G_min_raw": float(
                        pd.to_numeric(subset["G_min_raw"], errors="raise").min()
                    ),
                    "maximum_pseudo_max": float(
                        pd.to_numeric(subset["pseudo_max"], errors="raise").max()
                    ),
                }
            )
        stage18_stored = float(
            stage18_reps.loc[
                stage18_reps["bootstrap_repetition"].astype(int) == repetition,
                "aggregated_effect_days",
            ].iloc[0]
        )
        convergence_rows.append(
            {
                "bootstrap_repetition": repetition,
                "stage18_stored_5_partition_effect_days": stage18_stored,
                "prefix_5_effect_days": estimates_by_prefix[5],
                "prefix_10_effect_days": estimates_by_prefix[10],
                "prefix_15_effect_days": estimates_by_prefix[15],
                "prefix_20_effect_days": estimates_by_prefix[20],
                "absolute_5_to_10_shift_days": abs(
                    estimates_by_prefix[10] - estimates_by_prefix[5]
                ),
                "absolute_10_to_15_shift_days": abs(
                    estimates_by_prefix[15] - estimates_by_prefix[10]
                ),
                "absolute_15_to_20_shift_days": abs(
                    estimates_by_prefix[20] - estimates_by_prefix[15]
                ),
                "absolute_10_to_20_shift_days": abs(
                    estimates_by_prefix[20] - estimates_by_prefix[10]
                ),
                "signed_5_to_20_shift_days": estimates_by_prefix[20]
                - estimates_by_prefix[5],
            }
        )

    prefix_df = pd.DataFrame(prefix_rows)
    convergence = pd.DataFrame(convergence_rows)

    distribution_rows = []
    for prefix in prefixes:
        values = prefix_df.loc[
            prefix_df["prefix_partitions"] == prefix, "aggregated_effect_days"
        ].to_numpy(float)
        mcse = prefix_df.loc[
            prefix_df["prefix_partitions"] == prefix, "partition_mcse_days"
        ].to_numpy(float)
        distribution_rows.append(
            {
                "prefix_partitions": prefix,
                "bootstrap_repetitions": len(values),
                "mean_effect_days": float(np.mean(values)),
                "median_effect_days": float(np.median(values)),
                "sd_effect_days": float(np.std(values, ddof=1)),
                "percentile_ci_low_days": float(np.quantile(values, 0.025)),
                "percentile_ci_high_days": float(np.quantile(values, 0.975)),
                "fraction_positive": float(np.mean(values > 0)),
                "minimum_effect_days": float(np.min(values)),
                "maximum_effect_days": float(np.max(values)),
                "median_partition_mcse_days": float(np.median(mcse)),
                "p95_partition_mcse_days": float(np.quantile(mcse, 0.95)),
            }
        )
    distribution = pd.DataFrame(distribution_rows)

    paired_summary = pd.DataFrame(
        [
            {
                "metric": "absolute_5_to_10_shift_days",
                "median": float(convergence["absolute_5_to_10_shift_days"].median()),
                "p95": float(convergence["absolute_5_to_10_shift_days"].quantile(0.95)),
                "maximum": float(convergence["absolute_5_to_10_shift_days"].max()),
            },
            {
                "metric": "absolute_10_to_15_shift_days",
                "median": float(convergence["absolute_10_to_15_shift_days"].median()),
                "p95": float(convergence["absolute_10_to_15_shift_days"].quantile(0.95)),
                "maximum": float(convergence["absolute_10_to_15_shift_days"].max()),
            },
            {
                "metric": "absolute_15_to_20_shift_days",
                "median": float(convergence["absolute_15_to_20_shift_days"].median()),
                "p95": float(convergence["absolute_15_to_20_shift_days"].quantile(0.95)),
                "maximum": float(convergence["absolute_15_to_20_shift_days"].max()),
            },
            {
                "metric": "absolute_10_to_20_shift_days",
                "median": float(convergence["absolute_10_to_20_shift_days"].median()),
                "p95": float(convergence["absolute_10_to_20_shift_days"].quantile(0.95)),
                "maximum": float(convergence["absolute_10_to_20_shift_days"].max()),
            },
        ]
    )

    prefix15_mean = float(
        distribution.loc[
            distribution["prefix_partitions"] == 15, "mean_effect_days"
        ].iloc[0]
    )
    prefix20_mean = float(
        distribution.loc[
            distribution["prefix_partitions"] == 20, "mean_effect_days"
        ].iloc[0]
    )
    final_summary = pd.DataFrame(
        [
            {
                "bootstrap_repetitions": target_reps,
                "total_partitions_per_bootstrap": target_partitions,
                "prefix20_mean_effect_days": prefix20_mean,
                "prefix20_median_effect_days": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20, "median_effect_days"
                    ].iloc[0]
                ),
                "prefix20_sd_effect_days": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20, "sd_effect_days"
                    ].iloc[0]
                ),
                "prefix20_percentile_ci_low_days": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20,
                        "percentile_ci_low_days",
                    ].iloc[0]
                ),
                "prefix20_percentile_ci_high_days": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20,
                        "percentile_ci_high_days",
                    ].iloc[0]
                ),
                "prefix20_fraction_positive": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20, "fraction_positive"
                    ].iloc[0]
                ),
                "median_absolute_10_to_20_shift_days": float(
                    convergence["absolute_10_to_20_shift_days"].median()
                ),
                "p95_absolute_10_to_20_shift_days": float(
                    convergence["absolute_10_to_20_shift_days"].quantile(0.95)
                ),
                "median_mcse_at_20_days": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20,
                        "median_partition_mcse_days",
                    ].iloc[0]
                ),
                "p95_mcse_at_20_days": float(
                    distribution.loc[
                        distribution["prefix_partitions"] == 20,
                        "p95_partition_mcse_days",
                    ].iloc[0]
                ),
                "absolute_15_vs_20_distribution_mean_shift_days": abs(
                    prefix20_mean - prefix15_mean
                ),
                "correlation_prefix5_prefix20": float(
                    np.corrcoef(
                        convergence["prefix_5_effect_days"],
                        convergence["prefix_20_effect_days"],
                    )[0, 1]
                ),
                "mean_signed_5_to_20_shift_days": float(
                    convergence["signed_5_to_20_shift_days"].mean()
                ),
                "maximum_absolute_5_to_20_shift_days": float(
                    np.abs(convergence["signed_5_to_20_shift_days"]).max()
                ),
            }
        ]
    )

    write_csv(all_parts, tables / "77_all_20_partition_bootstrap_fits.csv")
    write_csv(prefix_df, tables / "77_bootstrap_prefix_effects.csv")
    write_csv(convergence, tables / "77_bootstrap_inner_crossfit_convergence.csv")
    write_csv(distribution, tables / "77_bootstrap_prefix_distributions.csv")
    write_csv(paired_summary, tables / "77_bootstrap_prefix_shift_summary.csv")
    write_csv(final_summary, tables / "77_stage19_stabilization_summary.csv")

    print("=" * 124)
    print("STAGE 77 - INNER CROSS-FIT CONVERGENCE ON THE SAME 30 BOOTSTRAP SAMPLES")
    print("=" * 124)
    print("Prefix distributions")
    print(dataframe_console(distribution))
    print("\nPaired prefix shifts")
    print(dataframe_console(paired_summary))
    print("\nPer-bootstrap convergence")
    print(dataframe_console(convergence))
    print("\nFinal stabilization summary")
    print(dataframe_console(final_summary))
    print("\nThe 30-sample bootstrap interval remains a pilot interval, not publication inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
