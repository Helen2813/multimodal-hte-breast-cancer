#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage19_utils import (
    dataframe_console,
    denominator_weighted_effect,
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
    tables = root / "results/tables"

    stage18_summary_path = tables / "73_bootstrap_pilot_summary.csv"
    stage18_reps_path = tables / "72_bootstrap_pilot_repetitions_checkpoint.csv"
    stage18_parts_path = tables / "72_bootstrap_pilot_partitions_checkpoint.csv"
    stage18_decision_path = tables / "74_stage18_decision.csv"
    required = [stage18_summary_path, stage18_reps_path, stage18_parts_path, stage18_decision_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 18 inputs:\n" + "\n".join(missing))

    summary = read_csv(stage18_summary_path).iloc[0]
    reps = read_csv(stage18_reps_path)
    parts = read_csv(stage18_parts_path)
    decision18 = read_csv(stage18_decision_path).iloc[0]

    completed = int(reps["bootstrap_repetition"].nunique())
    partition_counts = parts.groupby("bootstrap_repetition")["partition"].nunique()
    required_partitions = set(range(1, int(cfg["extension"]["stage18_existing_partitions"]) + 1))
    bad_partition_reps = []
    reaggregation_rows = []
    for repetition, group in parts.groupby("bootstrap_repetition"):
        observed = set(pd.to_numeric(group["partition"], errors="raise").astype(int))
        if not required_partitions.issubset(observed):
            bad_partition_reps.append(int(repetition))
            continue
        subset = group[group["partition"].astype(int).isin(required_partitions)].copy()
        recomputed = denominator_weighted_effect(subset)
        stored = float(
            reps.loc[
                reps["bootstrap_repetition"].astype(int) == int(repetition),
                "aggregated_effect_days",
            ].iloc[0]
        )
        reaggregation_rows.append(
            {
                "bootstrap_repetition": int(repetition),
                "stored_stage18_effect_days": stored,
                "recomputed_from_partition_summaries_days": recomputed,
                "absolute_difference_days": abs(stored - recomputed),
            }
        )
    reaggregation = pd.DataFrame(reaggregation_rows)
    max_difference = float(reaggregation["absolute_difference_days"].max())

    checks = pd.DataFrame(
        [
            {
                "check": "stage18_successful_repetitions",
                "observed": completed,
                "expected": int(cfg["extension"]["bootstrap_repetitions"]),
                "pass": completed == int(cfg["extension"]["bootstrap_repetitions"]),
            },
            {
                "check": "stage18_failed_repetitions",
                "observed": int(summary["failed_repetitions"]),
                "expected": 0,
                "pass": int(summary["failed_repetitions"]) == 0,
            },
            {
                "check": "stage18_numerical_explosion_fraction",
                "observed": float(decision18["numerical_explosion_fraction"]),
                "expected": 0.0,
                "pass": float(decision18["numerical_explosion_fraction"]) == 0.0,
            },
            {
                "check": "five_partition_rows_present_for_every_repetition",
                "observed": len(bad_partition_reps),
                "expected": 0,
                "pass": len(bad_partition_reps) == 0,
            },
            {
                "check": "partition_summary_reaggregation_matches_stage18",
                "observed": max_difference,
                "expected": float(
                    cfg["decision_thresholds"]["maximum_stage18_reaggregation_difference_days"]
                ),
                "pass": max_difference
                <= float(cfg["decision_thresholds"]["maximum_stage18_reaggregation_difference_days"]),
            },
        ]
    )
    if not bool(checks["pass"].all()):
        print(dataframe_console(checks))
        raise RuntimeError("Stage 19 preflight failed. Do not extend the bootstrap partitions.")

    amendment = pd.DataFrame(
        [
            {
                "stage19_protocol_status": cfg["protocol_status"],
                "stage18_automatic_label": str(decision18["stage18_decision"]),
                "stage18_successful_repetitions": completed,
                "stage18_failed_repetitions": int(summary["failed_repetitions"]),
                "stage18_numerical_explosion_fraction": float(
                    decision18["numerical_explosion_fraction"]
                ),
                "stage18_median_partition_sd_days": float(summary["median_partition_sd_days"]),
                "stage18_p95_partition_sd_days": float(summary["p95_partition_sd_days"]),
                "corrected_interpretation": cfg["interpretation_amendment"][
                    "corrected_interpretation"
                ],
                "new_diagnostic": "Extend the same 30 bootstrap samples from 5 to 20 prespecified nuisance partitions and assess prefix convergence plus Monte Carlo standard error.",
                "publication_bootstrap_locked": True,
            }
        ]
    )

    write_csv(reaggregation, tables / "75_stage18_partition_reaggregation_check.csv")
    write_csv(checks, tables / "75_stage19_preflight_checks.csv")
    write_csv(amendment, tables / "75_stage19_protocol_correction.csv")
    report = f"""# Stage 19 protocol correction

The Stage 18 label is retained as an audit trail and is not overwritten.

The pilot completed all 30 patient bootstrap samples without fit failures or numerical
explosions. Its failed gates were thresholds on the standard deviation across only five
nuisance partitions. That quantity measures finite Monte Carlo noise from nuisance
partitioning; it is not, by itself, a numerical failure of the estimator.

Stage 19 therefore adds prespecified partitions 6 through 20 to the same bootstrap samples.
It does not draw new bootstrap samples and does not start the publication bootstrap.

## Preflight checks

{markdown_table(checks)}

## Amendment

{markdown_table(amendment)}
"""
    write_text(report, tables / "75_stage19_protocol_correction.md")

    print("=" * 124)
    print("STAGE 75 - STAGE 19 PROTOCOL CORRECTION AND PREFLIGHT")
    print("=" * 124)
    print("Stage 18 automatic decision")
    print(str(decision18["stage18_decision"]))
    print("\nCorrected interpretation")
    print(cfg["interpretation_amendment"]["corrected_interpretation"])
    print("\nPreflight checks")
    print(dataframe_console(checks))
    print("\nProtocol amendment")
    print(dataframe_console(amendment))
    print("\nStage 18 is not rerun. No new patient bootstrap samples are drawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
