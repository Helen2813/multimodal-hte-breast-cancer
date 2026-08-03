# Paper A Stage 31 — focused sequencing simulation pilot

Extract into the project root and run:

```powershell
.\run_stage31_sequencing_simulation_pilot.ps1
```

The pilot uses six scenarios:

- n = 300 and 600;
- no, moderate, and strong chemotherapy-sequencing effects on hormone initiation;
- 50 repetitions per scenario.

It compares:

1. A full-cohort estimator that omits the sequencing indicator.
2. A full-cohort estimator that adjusts for the sequencing indicator.
3. A sequencing-aware estimator restricted to patients without chemotherapy
   by day 180.

The third estimator has a different target population and is evaluated against
its own no-chemotherapy overlap-population truth.

The simulation reports:

- bias and RMSE;
- empirical standard deviation and mean diagnostic IF standard error;
- empirical 95% IF-interval coverage;
- overlap-weighted balance;
- effective sample-size fractions;
- event counts and treated-event counts;
- propensity and IPCW pseudo-outcome diagnostics;
- numerical failure rates.

This is a pilot only. A larger confirmatory simulation will be locked separately
only after the pilot behaviour is reviewed.
