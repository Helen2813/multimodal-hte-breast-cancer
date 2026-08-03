# Paper A Stage 26 — locked Candidate V10 point estimate

Extract into the project root and run:

```powershell
.\run_stage26_candidate_v10_point_estimate.ps1
```

Stage 26:

1. Verifies every file in the Candidate V10 Stage 25C manifest.
2. Hash-locks the Stage 26 calculation code before any effect is computed.
3. Reuses the frozen 271-patient V10 population.
4. Fits the locked full-sample unpenalized logistic propensity once.
5. Runs the exact Candidate V9 cross-fitted censoring and bounded arm-specific
   RidgeCV outcome nuisances over the 20 locked partition seeds.
6. Computes the algebraically stabilized, unclipped ATO-AIPW RMST point estimate.
7. Reports partition variability, diagnostic influence-function uncertainty,
   nuisance diagnostics, and patient influence diagnostics.

Stage 26 does not run the publication bootstrap and does not generate manuscript
text. The diagnostic influence-function interval is not the primary publication
interval.

Upload the single Stage 26 log. A separate Stage 27 package will run the locked
300-repetition patient bootstrap only after the point-estimate diagnostics are
reviewed.
