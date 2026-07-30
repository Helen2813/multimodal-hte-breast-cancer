from __future__ import annotations

import json

import pandas as pd

from _common import PROJECT_ROOT, RESULTS_DIR, ensure_dirs, read_table


def markdown_escape(value: object) -> str:
    """Render one Markdown cell without optional third-party packages."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        cell = f"{value:.6g}"
    else:
        cell = str(value)
    return cell.replace("|", r"\|").replace("\n", " ")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a pandas DataFrame as Markdown using no optional packages."""
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for record in frame.itertuples(index=False, name=None):
        body.append(
            "| "
            + " | ".join(markdown_escape(value) for value in record)
            + " |"
        )
    return "\n".join([header, separator, *body])


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    paper_dir = PROJECT_ROOT / "paper_A_treatment_effects"
    paper_dir.mkdir(parents=True, exist_ok=True)

    candidate = read_table(table_dir / "34_paperA_candidate_summary.csv").iloc[0]
    composition = read_table(table_dir / "37_control_strategy_composition.csv")
    later_balance = read_table(table_dir / "37_later_vs_never_balance.csv")
    era = read_table(table_dir / "37_era_interaction_feasibility.csv")
    ccw = read_table(table_dir / "39_ccw_feasibility_decision.csv").iloc[0]

    later_row = composition[
        composition["control_component"] == "later_initiator"
    ].iloc[0]
    never_row = composition[
        composition["control_component"] == "no_recorded_later_initiation"
    ].iloc[0]
    later_auc = float(composition["later_prediction_oof_auc"].iloc[0])
    later_max_smd = float(later_balance["abs_smd"].max())
    era_formal = int(era["formal_interaction_feasible"].max())

    ccw_status = str(ccw["feasibility_status"])
    ccw_plan = (
        "A full clone-censor-weight analysis will be implemented as a diagnosis-time sensitivity analysis."
        if ccw_status.startswith("CCW_SENSITIVITY_FEASIBLE")
        else (
            "Clone-censor-weight was assessed but not promoted to an effect analysis because "
            "artificial-censoring weights or balance were insufficiently stable."
        )
    )

    config = {
        "status": "CANDIDATE_V2_NOT_LOCKED",
        "paper": "A",
        "design_name": "180-day landmark with grace-period treatment strategies",
        "target_population": (
            "verified HR+/HER2-negative patients alive and uncensored at day 180 after diagnosis"
        ),
        "strategies": {
            "early_initiation": (
                "initiate verified hormone therapy from diagnosis through day 180"
            ),
            "no_initiation_by_landmark": (
                "do not initiate through day 180; later initiation is permitted after the grace period"
            ),
        },
        "primary_estimand": (
            "ATO 730-day post-landmark RMST difference among day-180 survivors"
        ),
        "primary_effect_days": float(candidate["primary_effect_days"]),
        "diagnostic_ci": [float(candidate["if_ci_low"]), float(candidate["if_ci_high"])],
        "control_strategy_composition": {
            "later_initiators": int(later_row["n"]),
            "no_recorded_later_initiation": int(never_row["n"]),
            "later_vs_never_max_abs_smd": later_max_smd,
            "later_initiation_prediction_auc": later_auc,
        },
        "era_interaction": (
            "formal_exploratory_interaction"
            if era_formal
            else "descriptive_only_due_to_sparse_event_cells"
        ),
        "ccw_feasibility_status": ccw_status,
        "ccw_plan": ccw_plan,
        "ai_claim": (
            "In this TCGA application, boosted nuisance models produced less stable censoring weights "
            "and wider uncertainty; no general claim about AI methods is made."
        ),
        "identifiability_assumptions": [
            "consistency of recorded initiation timing",
            "conditional exchangeability of early-initiation strategies",
            "positivity in the day-180 survivor population",
            "conditional independent natural censoring",
            "accurate receptor, event, and timing measurement",
            "no interference",
            "selection to the landmark population is part of the estimand",
            "later initiation after day 180 is allowed under the no-initiation-by-day-180 strategy",
        ],
        "required_before_lock": [
            "professor approval",
            "full-pipeline patient bootstrap",
            "CCW sensitivity implementation if feasible",
            "final Table 1, love plot, and flow figure review",
            "clinical contextualization against external evidence",
            "frozen software/model registry and git tag",
        ],
    }
    config_path = paper_dir / "primary_estimand_CANDIDATE_V2.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    plan = f"""# Paper A analysis plan — Candidate V2

## Status

**CANDIDATE_V2_NOT_LOCKED.**

## Design terminology

This is a 180-day **landmark analysis with a treatment-initiation grace period**.
It compares two strategies among patients alive and uncensored at day 180:

1. initiate verified hormone therapy between diagnosis and day 180;
2. do not initiate by day 180.

The second strategy permits treatment initiation after day 180. Therefore, the
analysis does not compare permanently treated and permanently untreated patients.

## Primary estimand

The primary estimand is the overlap-population difference in 730-day post-landmark
restricted mean survival time among day-180 survivors.

Candidate estimate: {config['primary_effect_days']:.1f} days, with the current
diagnostic interval {config['diagnostic_ci'][0]:.1f} to
{config['diagnostic_ci'][1]:.1f} days.

## Control-strategy composition

The no-initiation-by-day-180 strategy contains:

- {int(later_row['n'])} patients with recorded initiation after day 180;
- {int(never_row['n'])} patients with no recorded later initiation.

The maximum absolute baseline SMD between these components is
{later_max_smd:.3f}; the cross-fitted AUC for predicting later initiation is
{later_auc:.3f}. These groups are described to clarify the strategy composition,
not treated as separate causal arms.

## Clone-censor-weight sensitivity

{ccw_plan}

CCW feasibility status: `{ccw_status}`. Stage 39 is a weight/positivity diagnostic
only and does not itself estimate a treatment effect.

## Identifiability assumptions

1. recorded initiation timing is sufficiently accurate;
2. treatment strategies are conditionally exchangeable given baseline W;
3. positivity holds in the day-180 survivor population;
4. natural censoring is conditionally independent;
5. receptor, event, and clinical measurements are valid;
6. no interference occurs;
7. conditioning on survival and observation through day 180 defines the target
   population rather than estimating a diagnosis-time population;
8. later initiation is allowed after the grace period in the comparison strategy.

## Era analysis

Era-by-strategy interaction is `{config['era_interaction']}`. When event cells are
sparse, era-specific results will be descriptive and will not support subgroup-effect
claims.

## AI interpretation

In this application, boosted nuisance models generated more extreme censoring-weight
behaviour and wider uncertainty than the regularized classical specification. This
statement is application-specific; the paper will not generalize that AI methods are
intrinsically less reliable.

## Changes relative to the earlier exploratory analysis

- imputed receptor scores were no longer thresholded as observed labels;
- treatment families were reconstructed from the authoritative source;
- five-year binary mortality was not treated as fully observed;
- ever-treated exposure was replaced by a time-aligned strategy;
- TNBC chemotherapy failed the overlap/balance/ESS gate;
- targeted therapy was too sparse for a primary analysis;
- individual treatment-effect claims were removed;
- the estimand became a landmark RMST contrast in the overlap population.

## Reporting assets

The staged pipeline now creates a cohort flow diagram, Table 1 before and after
overlap weighting, a primary love plot, a control-composition love plot, and complete
console transcripts.

## Before protocol lock

Professor review, full-pipeline bootstrap, the planned CCW sensitivity when feasible,
clinical contextualization, frozen software/model registry, and a git-tagged hash
manifest remain required.
"""
    plan_path = paper_dir / "analysis_plan_CANDIDATE_V2.md"
    plan_path.write_text(plan, encoding="utf-8")

    decision = pd.DataFrame(
        [
            {
                "primary_effect_days": config["primary_effect_days"],
                "later_initiators": int(later_row["n"]),
                "never_recorded_later": int(never_row["n"]),
                "later_vs_never_max_abs_smd": later_max_smd,
                "later_prediction_auc": later_auc,
                "era_interaction": config["era_interaction"],
                "ccw_feasibility_status": ccw_status,
                "protocol_status": config["status"],
                "next_stage": (
                    "full_pipeline_bootstrap_and_ccw_sensitivity"
                    if ccw_status.startswith("CCW_SENSITIVITY_FEASIBLE")
                    else "full_pipeline_bootstrap_landmark_only"
                ),
            }
        ]
    )
    decision.to_csv(table_dir / "40_stage11_design_decision.csv", index=False)

    report = table_dir / "40_stage11_design_decision.md"
    report.write_text(
        "# Stage 11 design decision\n\n" + dataframe_to_markdown(decision) + "\n",
        encoding="utf-8",
    )

    print("=" * 115)
    print("STAGE 40 — UPDATED PAPER A CANDIDATE")
    print("=" * 115)
    print(decision.to_string(index=False))
    print(f"\nUpdated plan: {plan_path}")
    print(f"Updated config: {config_path}")
    print(f"Decision report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
