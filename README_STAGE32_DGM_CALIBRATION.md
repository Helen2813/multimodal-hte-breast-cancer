# Paper A Stage 32 — empirically anchored DGM calibration

Extract into the project root and run:

```powershell
.\run_stage32_sequence_dgm_calibration.ps1
```

Stage 32 does not run the final simulation. It first corrects the design
mismatch identified in the Stage 31 pilot.

It calibrates:

- chemotherapy prevalence and its arm-specific imbalance;
- the full-cohort hormone-initiation fraction;
- the no-chemotherapy population fraction;
- the treatment fraction after no-chemotherapy restriction;
- arm-specific event risks in the sequencing-aware population.

The targets come from the documented V9 sequencing audit and frozen V10
counts. Calibration uses a deterministic Sobol training sample and a larger,
independent Sobol validation sample.

The resulting JSON proposes a revised simulation design with:

- sample sizes 559 and 1118;
- no, half-empirical, and empirical sequencing levels;
- null and empirically calibrated beneficial treatment-effect regimes;
- explicit omitted-chemotherapy balance diagnostics;
- at least 500 and preferably 1000 repetitions.

This is a simulation-design calibration only. It does not modify the real-data
analysis and does not generate manuscript prose.
