# Stage 5 — source verification before HTE modeling

Copy the files into the project root and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage5_source_verification.ps1
```

## Why this stage is mandatory

The current TNBC files contain varying `W_ER`, `W_PR`, and `W_HER2`
values even though TNBC eligibility should fix all three receptors as negative.
This suggests that imputed/standardized receptor scores were treated as binary
clinical labels.

The previous timing audit also counted birth, follow-up, and sample-collection
dates rather than treatment start/end dates.

## Outputs

### Receptor source audit

```text
results/tables/13_receptor_source_summary.csv
results/tables/13_receptor_source_audit.csv
results/tables/13_current_ite_receptor_fields.csv
```

This searches the copied processed files for raw textual or genuinely binary
ER/PR/HER2 fields. It does not automatically rebuild cohorts.

### Treatment reconstruction verification

```text
results/tables/14_treatment_reconstruction_comparison.csv
results/tables/14_patient_treatment_flag_comparison.csv
results/tables/14_treatment_text_examples.csv
results/tables/14_true_treatment_timing_fields.csv
```

The script explicitly prioritizes `treatment_type.treatments.diagnoses` and
compares reconstructed treatment families with `T_hormone`, `T_chemo`,
`T_targeted`, and `T_radiation`.

### Balance sensitivity

```text
results/tables/15_balance_sensitivity_summary.csv
results/tables/15_balance_without_receptor_scores_*.csv
results/tables/15_weights_without_receptor_scores_*.csv
```

This removes the current receptor score columns from the diagnostic propensity
model and compares regularized overlap weights with exponential-tilting
calibration weights. These weights are diagnostic only until verified receptor
labels are used to rebuild eligibility cohorts.
