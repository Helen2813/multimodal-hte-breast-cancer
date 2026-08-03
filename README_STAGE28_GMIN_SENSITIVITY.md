# Paper A Stage 28 — censoring G-min sensitivity

Extract the package into the project root and run:

```powershell
.\run_stage28_candidate_v10_gmin_sensitivity.ps1
```

Stage 28 is a post hoc diagnostic sensitivity analysis.

It:

- verifies the Candidate V10 and Stage 26 locks;
- locks the Stage 28 code and G-min values before computing sensitivity results;
- reproduces the locked primary G-min=0.10 point estimate exactly;
- computes 20-partition point estimates at G-min=0.15 and G-min=0.20;
- keeps the cohort, treatment definition, propensity, overlap score, folds,
  learners, and partition seeds unchanged;
- reports paired partition differences and IPCW pseudo-outcome diagnostics;
- creates one numerical figure.

It does not:

- rerun the 300-repetition publication bootstrap;
- change the primary Candidate V10 analysis;
- create new confidence intervals;
- modify Candidate V9 or Candidate V10;
- generate manuscript prose.

Upload the single Stage 28 log after completion.
