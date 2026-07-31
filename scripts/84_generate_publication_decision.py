#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage21_utils import (
    dataframe_console,
    ensure_stage21_dirs,
    load_stage21_config,
    markdown_table,
    project_root,
    verify_locked_files,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_stage21_dirs(root)
    verify_locked_files(root)
    cfg = load_stage21_config(root)
    thresholds = cfg["decision_thresholds"]
    tables = root / "results/tables"

    summary = pd.read_csv(tables / "83_publication_bootstrap_summary.csv", low_memory=False)
    convergence = pd.read_csv(tables / "83_publication_bootstrap_convergence.csv", low_memory=False)
    if len(summary) != 1 or len(convergence) != 1:
        raise RuntimeError("Expected one Stage 83 summary row and one convergence row.")
    s = summary.iloc[0]
    c = convergence.iloc[0]

    checks = pd.DataFrame([
        {
            "check": "minimum_successful_repetitions",
            "observed": float(s["successful_repetitions"]),
            "threshold_or_expected": float(thresholds["minimum_successful_repetitions"]),
            "pass": float(s["successful_repetitions"]) >= float(thresholds["minimum_successful_repetitions"]),
        },
        {
            "check": "minimum_success_rate",
            "observed": float(s["success_rate"]),
            "threshold_or_expected": float(thresholds["minimum_success_rate"]),
            "pass": float(s["success_rate"]) >= float(thresholds["minimum_success_rate"]),
        },
        {
            "check": "maximum_abs_prefix_200_vs_300_mean_shift_days",
            "observed": float(c["absolute_prefix_200_vs_300_mean_shift_days"]),
            "threshold_or_expected": float(thresholds["maximum_abs_prefix_200_vs_300_mean_shift_days"]),
            "pass": np.isfinite(float(c["absolute_prefix_200_vs_300_mean_shift_days"])) and float(c["absolute_prefix_200_vs_300_mean_shift_days"]) <= float(thresholds["maximum_abs_prefix_200_vs_300_mean_shift_days"]),
        },
        {
            "check": "maximum_abs_prefix_200_vs_300_lower_endpoint_shift_days",
            "observed": float(c["absolute_prefix_200_vs_300_lower_endpoint_shift_days"]),
            "threshold_or_expected": float(thresholds["maximum_abs_prefix_200_vs_300_lower_endpoint_shift_days"]),
            "pass": np.isfinite(float(c["absolute_prefix_200_vs_300_lower_endpoint_shift_days"])) and float(c["absolute_prefix_200_vs_300_lower_endpoint_shift_days"]) <= float(thresholds["maximum_abs_prefix_200_vs_300_lower_endpoint_shift_days"]),
        },
        {
            "check": "maximum_abs_prefix_200_vs_300_upper_endpoint_shift_days",
            "observed": float(c["absolute_prefix_200_vs_300_upper_endpoint_shift_days"]),
            "threshold_or_expected": float(thresholds["maximum_abs_prefix_200_vs_300_upper_endpoint_shift_days"]),
            "pass": np.isfinite(float(c["absolute_prefix_200_vs_300_upper_endpoint_shift_days"])) and float(c["absolute_prefix_200_vs_300_upper_endpoint_shift_days"]) <= float(thresholds["maximum_abs_prefix_200_vs_300_upper_endpoint_shift_days"]),
        },
        {
            "check": "maximum_numerical_explosion_fraction",
            "observed": float(s["numerical_explosion_fraction"]),
            "threshold_or_expected": float(thresholds["maximum_numerical_explosion_fraction"]),
            "pass": float(s["numerical_explosion_fraction"]) <= float(thresholds["maximum_numerical_explosion_fraction"]),
        },
    ])

    computational_pass = bool(checks["pass"].all())
    low = float(s["percentile_ci_low_days"])
    high = float(s["percentile_ci_high_days"])
    point = float(s["locked_point_estimate_days"])

    if not computational_pass:
        decision = "FULL_BOOTSTRAP_INCOMPLETE_OR_MONTE_CARLO_UNSTABLE"
        claim_status = "No treatment-effect claim should be finalized from this run."
        recommended = "Resolve only documented computational failures without changing the locked estimator; otherwise report the analysis as incomplete."
    elif low > 0:
        decision = "FULL_BOOTSTRAP_COMPLETE_POSITIVE_ATO_RMST_DIRECTION_SUPPORTED"
        claim_status = (
            "The locked analysis supports a positive 730-day post-landmark ATO RMST contrast for hormone-therapy initiation by day 180. "
            "This is not an unconditional efficacy claim and remains conditional on the stated causal and censoring assumptions."
        )
        recommended = "Freeze all final results, generate manuscript tables and figures, and proceed to the prespecified sensitivity analyses without changing the primary estimator."
    elif high < 0:
        decision = "FULL_BOOTSTRAP_COMPLETE_NEGATIVE_ATO_RMST_DIRECTION_SUPPORTED"
        claim_status = (
            "The locked analysis supports a negative 730-day post-landmark ATO RMST contrast. "
            "Interpretation remains conditional on the stated causal and censoring assumptions."
        )
        recommended = "Freeze all final results and revise the clinical interpretation without changing the primary estimator."
    else:
        decision = "FULL_BOOTSTRAP_COMPLETE_DIRECTION_IMPRECISE"
        claim_status = (
            "The locked point estimate is reported with a patient-bootstrap interval that includes zero; the data do not distinguish no contrast from effects within the reported interval."
        )
        recommended = "Freeze the result as an imprecise reliability finding and proceed to prespecified sensitivity analyses; do not search for a more favorable primary specification."

    decision_row = pd.DataFrame([{
        "stage21_decision": decision,
        "protocol_status": "PAPER_A_CANDIDATE_V9_ANALYSIS_COMPLETE",
        "locked_point_estimate_days": point,
        "primary_percentile_ci_low_days": low,
        "primary_percentile_ci_high_days": high,
        "bootstrap_mean_days": float(s["bootstrap_mean_days"]),
        "bootstrap_sd_days": float(s["bootstrap_sd_days"]),
        "fraction_positive": float(s["fraction_positive"]),
        "successful_repetitions": int(s["successful_repetitions"]),
        "computational_gates_passed": computational_pass,
        "claim_status": claim_status,
        "recommended_next_step": recommended,
    }])

    write_csv(checks, tables / "84_publication_bootstrap_decision_checks.csv")
    write_csv(decision_row, tables / "84_publication_bootstrap_decision.csv")

    report = f"""# Candidate V9 final publication-bootstrap decision

**Decision:** `{decision}`  
**Protocol status:** `PAPER_A_CANDIDATE_V9_ANALYSIS_COMPLETE`

## Final locked result

{markdown_table(decision_row)}

## Computational and Monte Carlo checks

{markdown_table(checks)}

## Interpretation

{claim_status}

## Recommended next step

{recommended}
"""
    write_text(report, tables / "84_publication_bootstrap_decision.md")

    print("=" * 128)
    print("STAGE 84 - FINAL PUBLICATION BOOTSTRAP DECISION")
    print("=" * 128)
    print("Decision summary")
    print(dataframe_console(decision_row))
    print("\nPrespecified computational checks")
    print(dataframe_console(checks))
    print("\nFull decision report")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
