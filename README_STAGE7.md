# Stage 7 — temporal audit, propensity sensitivity, verified survival baseline, and plan drafts

Copy the package contents into the root of:

```text
multimodal-hte-breast-cancer/
```

Run from the active `.venv`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage7_temporal_design_lock.ps1
```

The PowerShell runner prints every decision table, top residual imbalances,
fold counts, timing coverage, event-definition diagnostics, and output path.
It also saves the entire console transcript automatically to:

```text
results/logs/stage7_YYYYMMDD_HHMMSS.log
```

## Scripts

```text
scripts/20_audit_temporal_era_and_outcomes.py
scripts/21_compare_propensity_strategies.py
scripts/22_verified_survival_baseline.py
scripts/23_create_verified_splits_and_plan_drafts.py
```

## Stage 20

- prints all true treatment timing field names and their nonmissing coverage;
- prints coverage separately by treatment family;
- finds and audits diagnosis year;
- prints diagnosis-era × treatment tables;
- verifies `OS`, `OS.time`, and reconstructs known five-year status;
- reports censoring before five years and disagreement with the legacy binary outcome.

## Stage 21

- compares the prespecified compact propensity strategy with a full baseline
  clinical elastic-net propensity sensitivity;
- excludes standardized receptor scores, outcomes, treatment variables,
  administrative timestamps, follow-up, and post-treatment fields;
- adds verified diagnosis year when sufficiently observed;
- prints tuning choices and the top residual imbalances for every cohort.

## Stage 22

Recomputes the survival baseline on the corrected verified cohorts using both
propensity strategies. This remains a diagnostic weighted Kaplan–Meier analysis,
not the final doubly robust Paper A estimator.

## Stage 23

- regenerates repeated splits after source correction;
- prints every fold's patient, treatment, control, and event counts;
- creates separate draft analysis plans for Paper A and Paper B;
- hashes verified cohort inputs and split assignments.

The generated plans remain `DRAFT_NOT_LOCKED` until Stage 20–23 is reviewed.
