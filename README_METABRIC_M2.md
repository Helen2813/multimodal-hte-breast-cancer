# METABRIC Stage M2

Run from the project root:

```powershell
.\run_metabric_stage_m2_harmonization_readiness.ps1
```

This stage does not rerun M1 and does not estimate a treatment effect.

It corrects two M1 audit issues:

- `AGE_AT_DIAGNOSIS` is not a diagnosis-year field.
- The reported 50-sample coverage for sample-oriented `gene_panel` and `rna_cleaned`
  files was only a first-50-row scan, not exact coverage.

M2 then:

- computes exact sample coverage for every modality;
- creates a verified patient-level clinical master;
- numerically validates the cleaned RNA file against raw microarray values;
- builds exact modality-availability registries;
- decides whether METABRIC is ready for a locked multimodal transport-validation protocol.

Single log:

```text
results/logs/metabric_stage_m2_harmonization_readiness_YYYYMMDD_HHMMSS.log
```
