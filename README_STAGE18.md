# Stage 18: patient-level bootstrap pilot

This stage does not rerun Stages 15-17. It documents a protocol amendment and runs a new
30-repetition patient-level bootstrap pilot.

## Why the pilot is appropriate

Stage 17 passed 9 of 10 prespecified checks. The only failed check was reduction of the
leave-one-original-fold-out spread. That diagnostic deletes about one fifth of the cohort and
therefore measures patient-composition uncertainty, not nuisance-partition randomness.
Patient-level bootstrap resampling is the correct next diagnostic.

## Primary pilot estimator

- ordinary patient bootstrap with replacement;
- 559 draws per repetition;
- duplicate copies of the same original patient are kept in the same nuisance fold;
- five prespecified grouped repeated-cross-fit partitions per bootstrap repetition;
- propensity refitted using the exact Stage 30 specification;
- censoring model refitted;
- IPCW RMST pseudo-outcome with G-min 0.10;
- arm-specific ridge outcome models bounded to 0-730 days;
- repeated patient-score aggregation across the five partitions.

The pilot contains 30 bootstrap repetitions. It is not final inference. The 300-repetition
publication bootstrap remains locked until the pilot is reviewed and the estimator protocol is
frozen.

## Run

```powershell
.\run_stage18_patient_bootstrap_pilot.ps1
```

The runner prints all key results to the terminal and saves one transcript under:

```text
results\logs\stage18_patient_bootstrap_pilot_YYYYMMDD_HHMMSS.log
```

The bootstrap script is checkpointed. If the process is interrupted, rerunning the same runner
continues from completed Stage 18 repetitions without rerunning Stages 15-17.
