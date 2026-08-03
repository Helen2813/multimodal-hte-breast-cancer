# Paper A Stage 25B — Candidate V10 balance repair

The original Stage 25 correctly stopped before protocol lock because the frozen
V10 cohort failed only the weighted-balance gate:

- age weighted SMD: 0.186
- diagnosis year weighted SMD: 0.145
- W_T_missing weighted SMD: 0.111

At the same time, overlap and effective sample size were excellent.

Stage 25B does not relax the balance threshold. It performs an outcome-blind
design repair before any V10 effect is calculated.

Run:

```powershell
.\run_stage25b_candidate_v10_balance_repair.ps1
```

The stage:

1. Reuses the frozen 271-patient V10 cohort from Stage 94.
2. Verifies its hashes and the exact Stage 25 failure pattern.
3. Fits an unpenalized full-sample logistic propensity model using the same
   13 baseline variables.
4. Checks ATO balance, propensity overlap, coefficients, and ESS.
5. Creates a Candidate V10 protocol lock only if every repair gate passes.

Censoring and outcome nuisances remain cross-fitted as in Candidate V9.
The repaired propensity will be refitted in every future patient-bootstrap sample.

No RMST effect, outcome regression, or bootstrap is run in Stage 25B.
