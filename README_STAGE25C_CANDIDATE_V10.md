# Paper A Stage 25C — unclipped stabilized overlap design

Stage 25B achieved exact overlap balance but stopped because one patient had a
propensity below 0.01 and two patients were near or above 0.99.

For an ATO estimand, the AIPW score can be written without inverse propensity
terms:

```text
h = e(1-e)

score numerator =
    h(mu1-mu0)
  + A(1-e)(Y-mu1)
  - (1-A)e(Y-mu0)
```

Therefore isolated endpoint propensities do not create exploding treatment
weights. The overlap target naturally downweights them.

Stage 25C:

- reuses the frozen 271-patient Candidate V10 population;
- retains the unpenalized logistic propensity;
- evaluates balance and ESS using the actual unclipped overlap weights;
- replaces pointwise min/max gates with tail-mass and overlap-mass gates;
- runs 300 propensity-only bootstrap resamples to test convergence and design
  stability before any outcome effect is calculated;
- locks the algebraically stabilized unclipped ATO score only if all gates pass.

Run:

```powershell
.\run_stage25c_candidate_v10_unclipped_ato_lock.ps1
```

No RMST effect, censoring model, outcome model, or publication bootstrap is run.
