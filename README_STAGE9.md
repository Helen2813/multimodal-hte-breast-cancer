# Stage 9 — landmark correction and Paper B power gate

Copy the package contents into the project root and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage9_landmark_and_power.ps1
```

The complete transcript is saved automatically to:

```text
results/logs/stage9_YYYYMMDD_HHMMSS.log
```

## Why Stage 9 is required

Stage 8 still classified treatment as “ever treated.” Because true treatment-start
days are available, that exposure can produce immortal-time bias. Stage 9 replaces
it with diagnosis-anchored landmark strategies.

## Stage 29

Builds 180-day and 365-day landmark cohorts:

- includes only patients alive and uncensored at the landmark;
- treated means verified initiation between day 0 and the landmark;
- patients initiating later remain in the “not treated by landmark” strategy and
  are explicitly counted;
- verified treated patients with missing, negative-only, or extreme start timing
  are excluded from the corresponding landmark analysis.

## Stage 30

Refits compact+era propensity models inside each landmark cohort, creates new
overlap weights, and reports SMD and ESS.

## Stage 31

For 2-year and 3-year post-landmark RMST:

- reverse-KM censoring;
- classical and boosted-AI censoring models;
- IPCW truncation at G=0.05 and G=0.10;
- classical and boosted-AI outcome nuisance models;
- exploratory ATO AIPW-RMST estimates;
- stability across model and truncation choices.

## Stage 32

Uses the best feasible outer-hormone landmark design to:

- rerun clinical versus clinical+RNA R-loss;
- compare ridge and boosted-AI learners;
- run a small power/type-I grid at null, weak, moderate, and strong RNA HTE;
- decide whether Paper B can detect moderate signals or only strong ones.

The simulation power gate is exploratory and does not replace the final
simulation framework.

## Stage 33

Generates the two-paper decision report. Protocol status remains
`DRAFT_NOT_LOCKED`.
