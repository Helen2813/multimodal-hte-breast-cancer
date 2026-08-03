# Paper A Stage 33 — revised sequencing simulation pilot

Extract into the project root and run:

```powershell
.\run_stage33_sequence_simulation_pilot_v2.ps1
```

The revised pilot uses the accepted Stage 32 data-generating parameters and
evaluates 12 scenarios:

- n = 559 and 1118;
- no, half-empirical, and empirical sequencing;
- true null and observed-risk-matching beneficial effect regimes;
- 100 repetitions per scenario.

It compares the naive full-cohort, sequencing-adjusted full-cohort, and
sequencing-aware estimators.

Major corrections relative to Stage 31:

1. The simulated treatment, chemotherapy, strict-population size, and sparse
   event structure are anchored to the documented V9/V10 margins.
2. Null-effect scenarios test whether sequencing produces a spurious positive
   treatment contrast.
3. The omitted chemotherapy SMD is reported before and after naive overlap
   weighting.
4. The naive estimator is evaluated against both:
   - the intended full-cohort ATO truth; and
   - its implied overlap target based on the marginal propensity that omits
     chemotherapy.
5. Naive total error is decomposed into target-population drift and residual
   omitted-sequencing bias.
6. The sequencing-aware estimator is evaluated only against its own strict
   no-chemotherapy target truth.
7. Independent ascertainability thinning reproduces the strict V10 sample-size
   scale without treatment- or outcome-dependent eligibility.

This remains a pilot. A confirmatory simulation is locked only after the pilot
gates and empirical anchor checks pass.
