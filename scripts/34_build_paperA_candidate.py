from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _common import PROJECT_ROOT, RESULTS_DIR, ensure_dirs
from _stage10_utils import (
    PRIMARY_COHORT,
    PRIMARY_G_MIN,
    PRIMARY_HORIZON,
    PRIMARY_LANDMARK,
    build_primary_model_table,
    load_paperA_inputs,
)


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    paper_dir = PROJECT_ROOT / "paper_A_treatment_effects"
    paper_dir.mkdir(parents=True, exist_ok=True)

    base, W_cols, RNA_cols, metadata = build_primary_model_table()
    inputs = load_paperA_inputs()

    primary_balance = inputs["balance"][
        (inputs["balance"]["cohort"] == PRIMARY_COHORT)
        & (
            inputs["balance"]["landmark_day"]
            == PRIMARY_LANDMARK
        )
    ]
    primary_gate = inputs["gate"][
        (inputs["gate"]["cohort"] == PRIMARY_COHORT)
        & (inputs["gate"]["landmark_day"] == PRIMARY_LANDMARK)
        & (
            inputs["gate"]["horizon_days_post_landmark"]
            == PRIMARY_HORIZON
        )
    ]
    primary_results = inputs["results"][
        (inputs["results"]["cohort"] == PRIMARY_COHORT)
        & (
            inputs["results"]["landmark_day"]
            == PRIMARY_LANDMARK
        )
        & (
            inputs["results"]["horizon_days_post_landmark"]
            == PRIMARY_HORIZON
        )
    ].copy()

    if primary_balance.empty or primary_gate.empty:
        raise ValueError(
            "Primary Paper A balance/gate rows were not found."
        )

    classical_primary = primary_results[
        (primary_results["strategy"] == "classical")
        & (
            primary_results["g_min"].sub(PRIMARY_G_MIN).abs()
            < 1e-9
        )
    ]
    if len(classical_primary) != 1:
        raise ValueError(
            "Expected exactly one classical primary result at Gmin=0.10."
        )

    result = classical_primary.iloc[0]
    balance = primary_balance.iloc[0]
    gate = primary_gate.iloc[0]

    candidate = {
        "status": "CANDIDATE_NOT_LOCKED",
        "paper": "A",
        "working_title": (
            "Reliability of AI-assisted landmark causal survival "
            "estimation in observational breast cancer data"
        ),
        "population": PRIMARY_COHORT,
        "time_zero": (
            "day 180 after diagnosis among patients alive and "
            "uncensored at day 180"
        ),
        "treatment_strategy": (
            "verified treatment initiation from day 0 through day 180 "
            "versus no initiation by day 180"
        ),
        "later_initiation_policy": (
            "patients initiating after day 180 remain in the "
            "no-initiation-by-day-180 strategy; this is not an "
            "ever-versus-never estimand"
        ),
        "primary_estimand": (
            "ATO 730-day post-landmark RMST difference"
        ),
        "primary_censoring_model": "regularized pooled logistic",
        "primary_outcome_model": "ridge regression",
        "primary_censoring_truncation": PRIMARY_G_MIN,
        "ai_role": (
            "boosted censoring and outcome nuisance models are "
            "prespecified sensitivity analyses, not the primary estimator"
        ),
        "primary_effect_days": float(
            result["aipw_ato_rmst_difference_days"]
        ),
        "primary_if_ci_low_days": float(
            result["influence_ci_low_days"]
        ),
        "primary_if_ci_high_days": float(
            result["influence_ci_high_days"]
        ),
        "primary_fixed_boot_ci_low_days": float(
            result["fixed_boot_ci_low_days"]
        ),
        "primary_fixed_boot_ci_high_days": float(
            result["fixed_boot_ci_high_days"]
        ),
        "max_abs_smd_overlap": float(
            balance["max_abs_smd_overlap"]
        ),
        "ess_treated": float(balance["ess_treated"]),
        "ess_control": float(balance["ess_control"]),
        "feasibility_status": str(
            gate["feasibility_status"]
        ),
        "claim": (
            "robustness and uncertainty of an early-initiation signal; "
            "not proof of treatment efficacy"
        ),
        "sensitivity_analyses": [
            "Gmin 0.05",
            "boosted-AI censoring and outcome nuisance models",
            "1095-day post-landmark RMST",
            "365-day landmark",
            "descriptive assessment of later initiators",
        ],
    }

    config_path = (
        paper_dir / "primary_estimand_CANDIDATE.json"
    )
    config_path.write_text(
        json.dumps(candidate, indent=2), encoding="utf-8"
    )

    plan = f"""# Paper A candidate analysis plan

## Status

**CANDIDATE_NOT_LOCKED.** This is the proposed final design after exploratory
source verification, temporal alignment, overlap diagnostics, and censoring
sensitivity analysis.

## Scientific question

Among verified HR-positive/HER2-negative patients alive and uncensored at
day {PRIMARY_LANDMARK} after diagnosis, what is the overlap-population
difference in subsequent restricted mean survival time between patients who
initiated verified hormone therapy by day {PRIMARY_LANDMARK} and patients
who had not initiated by that day?

## Important interpretation

This is an **early-initiation strategy** estimand. Patients initiating after
day {PRIMARY_LANDMARK} remain members of the no-initiation-by-day-{PRIMARY_LANDMARK}
strategy. The analysis is not interpreted as ever-treated versus never-treated
and not as a sustained-treatment per-protocol effect.

## Primary design

- Cohort: `{PRIMARY_COHORT}`
- Eligible patients: {metadata['n']}
- Treated by landmark: {metadata['treated']}
- Not treated by landmark: {metadata['controls']}
- Events after landmark: {metadata['events']}
- Time zero: day {PRIMARY_LANDMARK} after diagnosis
- Primary horizon: {PRIMARY_HORIZON} days after landmark
- Estimand: ATO RMST difference
- Primary propensity: cross-fitted compact clinical plus diagnosis-era model
- Primary censoring nuisance: regularized pooled logistic model
- Primary outcome nuisance: ridge regression
- Primary censoring survival truncation: {PRIMARY_G_MIN:.2f}

## Primary feasibility result

- AIPW-RMST difference: {candidate['primary_effect_days']:.1f} days
- Influence-function diagnostic interval:
  {candidate['primary_if_ci_low_days']:.1f} to
  {candidate['primary_if_ci_high_days']:.1f} days
- Fixed-nuisance bootstrap interval:
  {candidate['primary_fixed_boot_ci_low_days']:.1f} to
  {candidate['primary_fixed_boot_ci_high_days']:.1f} days
- Maximum weighted SMD: {candidate['max_abs_smd_overlap']:.3f}
- ESS treated/control:
  {candidate['ess_treated']:.1f}/{candidate['ess_control']:.1f}

## AI component

Boosted-AI censoring and outcome nuisance models are retained as a
prespecified sensitivity analysis. Their purpose is to assess whether flexible
machine learning improves nuisance prediction without destabilizing causal
weights or effect estimates. The primary estimator remains the more stable
regularized classical nuisance specification.

## Claims

The paper will evaluate reliability, model sensitivity, temporal alignment,
and precision. It will not claim that observational data establish hormone
therapy efficacy.

## Sensitivity analyses

1. censoring truncation at 0.05;
2. boosted-AI nuisance models;
3. 1095-day post-landmark RMST;
4. 365-day landmark;
5. descriptive analysis of later initiators.

## Required before final lock

- professor review of the early-initiation estimand;
- final full-pipeline bootstrap implementation;
- frozen model registry and software versions;
- final figure/table specifications;
- repository hash and git tag.
"""
    plan_path = paper_dir / "analysis_plan_CANDIDATE.md"
    plan_path.write_text(plan, encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                **metadata,
                "primary_effect_days": candidate[
                    "primary_effect_days"
                ],
                "if_ci_low": candidate[
                    "primary_if_ci_low_days"
                ],
                "if_ci_high": candidate[
                    "primary_if_ci_high_days"
                ],
                "max_abs_smd": candidate[
                    "max_abs_smd_overlap"
                ],
                "ess_treated": candidate["ess_treated"],
                "ess_control": candidate["ess_control"],
                "status": candidate["feasibility_status"],
            }
        ]
    )
    summary.to_csv(
        table_dir / "34_paperA_candidate_summary.csv",
        index=False,
    )

    print("=" * 115)
    print("STAGE 34 — PAPER A CANDIDATE DESIGN")
    print("=" * 115)
    print(summary.to_string(index=False))
    print(f"\nCandidate plan: {plan_path}")
    print(f"Candidate config: {config_path}")
    print("\nProtocol remains CANDIDATE_NOT_LOCKED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
