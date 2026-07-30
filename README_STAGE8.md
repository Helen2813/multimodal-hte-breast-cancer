# Stage 8 — first feasibility results for both papers

Copy the package contents into the project root and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage8_two_paper_feasibility.ps1
```

The complete console output is saved automatically to:

```text
results/logs/stage8_YYYYMMDD_HHMMSS.log
```

## What is corrected

Stage 7 evaluated diagnosis-year balance using the old compact weights. Stage 24
actually refits the compact propensity model with diagnosis year included.

## Paper A feasibility

- audited treatment initiation within 90, 180, and 365 days;
- reverse-KM censoring at 2, 3, 4, and 5 years;
- classical and boosted-AI censoring nuisance models;
- cross-fitted IPCW RMST pseudo-outcomes;
- classical and boosted-AI outcome nuisance models;
- exploratory ATO AIPW-RMST estimates at 3 and 5 years;
- influence-function and fixed-nuisance bootstrap diagnostic intervals.

The Stage 26 estimator is a feasibility estimator, not the final confirmatory
implementation.

## Paper B feasibility

On the verified outer hormone cohort:

- clinical versus clinical+RNA;
- linear ridge versus boosted-AI learners;
- prognostic IPCW-RMST prediction;
- prescriptive R-loss;
- three repeated fold assignments for the feasibility pilot;
- a 3-year primary pilot horizon because the 5-year cohort is heavily censored.

The final locked Paper B analysis will return to all five verified repeats after the simulation/power gate.

Only clinical versus RNA is opened at this stage. Other modalities remain closed
until the simulation/power gate.

## Outputs

```text
results/tables/24_compact_era_propensity_summary.csv
results/tables/24_treatment_timing_gate_summary.csv
results/tables/25_horizon_feasibility_gate.csv
results/tables/25_censoring_model_summary.csv
results/tables/25_ipcw_pseudooutcome_summary.csv
results/tables/26_paperA_ai_aipw_feasibility.csv
results/tables/26_paperA_feasibility_gate.csv
results/tables/27_paperB_ai_feasibility_summary.csv
results/tables/28_two_paper_feasibility_report.md
```

Protocol status remains `DRAFT_NOT_LOCKED`.
