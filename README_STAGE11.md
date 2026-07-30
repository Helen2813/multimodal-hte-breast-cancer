# Stage 11 — Paper A design hardening before full bootstrap

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage11_design_hardening.ps1
```

The complete transcript is written to:

```text
results/logs/stage11_YYYYMMDD_HHMMSS.log
```

## Stage 37

- decomposes the no-initiation-by-day-180 strategy into later initiators and
  patients with no recorded later initiation;
- calculates baseline SMDs and an out-of-fold AUC predicting later initiation;
- assesses whether era-by-strategy interaction has enough events for formal
  estimation or must remain descriptive.

## Stage 38

Creates Table 1, the primary love plot, a control-composition love plot, and a
landmark cohort flow diagram in CSV, LaTeX, PNG, and SVG formats.

## Stage 39

Performs a clone-censor-weight feasibility audit: diagnosis-time clone counts,
artificial-censoring flow, cross-fitted early-initiation probabilities, stabilized
baseline adherence weights, ESS, and balance diagnostics. This is not the final
CCW effect estimator.

## Stage 40

Updates the Paper A candidate plan with precise landmark/grace-period terminology,
control-strategy composition, landmark-specific identifiability assumptions,
application-specific AI wording, deviations from the earlier analysis, and a CCW
sensitivity decision.

Full-pipeline bootstrap starts only after Stage 11 review.
