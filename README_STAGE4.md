# Stage 4 — design checks before HTE models

Copy the files into the root of `multimodal-hte-breast-cancer/`.

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage4_design_checks.ps1
```

## Stage 10: treatment timing audit

Searches all clinical CSV/TSV files for treatment type, treatment start/end,
`days_to_*`, drug, and regimen fields.

Main outputs:

```text
results/tables/10_treatment_file_audit.csv
results/tables/10_treatment_timing_columns.csv
results/tables/10_treatment_timing_by_family.csv
data/derived/manifests/10_patient_treatment_timing_records.csv
```

This determines whether a landmark or target-trial time alignment is possible.

## Stage 11: complete-omics selection audit

Checks whether CNV, mutation, methylation, miRNA, and protein availability masks
are identical and whether complete multi-omics availability differs by treatment,
event, or compact clinical covariates.

Outputs:

```text
results/tables/11_complete_omics_selection_summary.csv
results/tables/11_complete_omics_selection_balance.csv
results/tables/11_modality_availability_jaccard.csv
data/derived/manifests/11_complete_omics_patient_ids.csv
```

## Stage 12: restricted overlap in complete-case cohorts

Uses the intersection of the 2.5th–97.5th percentile propensity regions for
treated and controls, then refits the propensity model and overlap weights.

Outputs:

```text
results/tables/12_restricted_complete_case_summary.csv
results/tables/12_restricted_balance_*.csv
results/tables/12_restricted_weights_*.csv
data/derived/restricted_complete_case/*.csv
```

Do not start modality HTE models until these three checks are reviewed.
