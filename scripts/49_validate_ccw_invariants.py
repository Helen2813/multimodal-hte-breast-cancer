#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage13_utils import (
    bootstrap_stats_from_summary_or_checkpoint,
    ensure_output_dirs,
    find_point_rows,
    load_config,
    markdown_table,
    numeric,
    project_root,
    read_csv,
    value_from_quantity_table,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_output_dirs(root)
    cfg = load_config(root)
    tables = root / "results" / "tables"
    expected = cfg["expected_counts"]
    _, ccw, _, _ = find_point_rows(root)

    flow_path = tables / "39_ccw_clone_flow.csv"
    flow = read_csv(flow_path) if flow_path.exists() else pd.DataFrame(columns=["quantity", "n"])

    observed = {
        "source_n": numeric(ccw.get("source_n"), expected["source_n"]),
        "excluded_ambiguous_timing": numeric(ccw.get("excluded_ambiguous"), expected["excluded_ambiguous"]),
        "ccw_eligible_patients": numeric(ccw.get("ccw_eligible_n"), expected["ccw_eligible_n"]),
        "observed_initiate_by_180": numeric(ccw.get("early_initiators"), expected["early_initiators"]),
        "observed_no_initiation_by_180": numeric(ccw.get("no_initiation_by_180"), expected["no_initiation_by_180"]),
        "diagnosis_time_clones": value_from_quantity_table(flow, "diagnosis_time_clones"),
        "no_initiation_clone_artificially_censored_at_start": value_from_quantity_table(
            flow, "no_initiation_clone_artificially_censored_at_start"
        ),
        "initiation_clone_artificially_censored_at_day180": value_from_quantity_table(
            flow, "initiation_clone_artificially_censored_at_day180"
        ),
        "natural_event_or_censor_before_day180_noinit": value_from_quantity_table(
            flow, "natural_event_or_censor_before_day180_noinit"
        ),
    }

    checks = [
        {
            "check": "source_partition",
            "observed": observed["ccw_eligible_patients"] + observed["excluded_ambiguous_timing"],
            "expected": observed["source_n"],
        },
        {
            "check": "strategy_partition",
            "observed": observed["observed_initiate_by_180"] + observed["observed_no_initiation_by_180"],
            "expected": observed["ccw_eligible_patients"],
        },
        {
            "check": "two_clones_per_eligible_patient",
            "observed": observed["diagnosis_time_clones"],
            "expected": 2 * observed["ccw_eligible_patients"],
        },
        {
            "check": "noinit_clone_censored_at_early_start",
            "observed": observed["no_initiation_clone_artificially_censored_at_start"],
            "expected": observed["observed_initiate_by_180"],
        },
        {
            "check": "initiation_clone_day180_accounting",
            "observed": observed["initiation_clone_artificially_censored_at_day180"]
            + observed["natural_event_or_censor_before_day180_noinit"],
            "expected": observed["observed_no_initiation_by_180"],
        },
        {
            "check": "ccw_point_weight_p99_below_warning",
            "observed": numeric(ccw.get("weight_p99")),
            "expected": float(cfg["weight_warning_threshold"]),
            "comparison": "<=",
        },
    ]

    for row in checks:
        if row.get("comparison") == "<=":
            row["pass"] = bool(np.isfinite(row["observed"]) and row["observed"] <= row["expected"])
        else:
            row["pass"] = bool(
                np.isfinite(row["observed"])
                and np.isfinite(row["expected"])
                and abs(row["observed"] - row["expected"]) < 1e-8
            )
        row.setdefault("comparison", "==")

    stats = bootstrap_stats_from_summary_or_checkpoint(root, "ccw")
    checks.append(
        {
            "check": "ccw_bootstrap_successful_repetitions",
            "observed": stats.successful_reps,
            "expected": cfg["minimum_reps_for_centering_gate"],
            "comparison": ">=",
            "pass": stats.successful_reps >= cfg["minimum_reps_for_centering_gate"],
        }
    )

    checks_df = pd.DataFrame(checks)
    status = "CCW_INTERNAL_CONSISTENCY_PASSED" if bool(checks_df["pass"].all()) else "CCW_INTERNAL_CONSISTENCY_REVIEW_REQUIRED"
    write_csv(checks_df, tables / "49_ccw_invariant_checks.csv")
    report = f"""# CCW internal-consistency checks

**Status:** `{status}`

{markdown_table(checks_df)}

These checks validate clone-flow arithmetic and gross weight behavior. They do not prove
exchangeability, correct treatment measurement, or correct natural-censoring models.
"""
    write_text(report, tables / "49_ccw_internal_consistency.md")

    print("=" * 112)
    print("STAGE 49 — CCW INTERNAL-CONSISTENCY AUDIT")
    print("=" * 112)
    print(checks_df.to_string(index=False))
    print(f"\nStatus: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
