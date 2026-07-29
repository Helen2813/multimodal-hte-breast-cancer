# Stage 3 — modality audit, fixed evaluation splits, and survival baseline

Copy the package contents into the root of:

```text
multimodal-hte-breast-cancer/
```

New files:

```text
scripts/07_audit_modality_availability.py
scripts/08_create_repeated_outer_splits.py
scripts/09_weighted_survival_baseline.py
run_stage3_survival_and_splits.ps1
```

Run from PowerShell in the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage3_survival_and_splits.ps1
```

The runner uses `.venv\Scripts\python.exe` explicitly.

## Outputs

### Stage 07

```text
results/tables/07_modality_availability_summary.csv
results/tables/07_modality_feature_audit.csv
data/derived/manifests/07_patient_modality_availability.csv
```

This checks whether the 51st outer-panel column is an availability/missingness
indicator and reports constant or near-zero features.

### Stage 08

```text
data/derived/nested_splits/08_repeated_outer_fold_assignments.csv
results/tables/08_repeated_outer_split_summary.csv
```

Five repeated outer splits are fixed before any modality comparison. All future
clinical-only and clinical-plus-omics models must use these same test folds.

### Stage 09

```text
results/tables/09_weighted_survival_baseline.csv
results/figures/09_weighted_survival_*.png
```

This reports overlap-weighted, stabilized-IPW, and trimmed weighted
Kaplan–Meier estimates at 1,825 days:

- survival probability difference;
- RMST difference in days;
- patient-level fixed-weight bootstrap intervals.

These are survival-aware arm-level baseline estimates. They are not the final
doubly robust HTE inference.
