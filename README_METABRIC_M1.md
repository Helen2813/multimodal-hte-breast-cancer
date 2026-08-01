# METABRIC Stage M1: read-only data-and-design audit

The package expects the raw METABRIC files under:

```text
data/raw/metabric/
```

Run from the project root in the active virtual environment:

```powershell
.\run_metabric_stage_m1_data_design_audit.ps1
```

The audit:

- inventories and fingerprints the copied raw files;
- reads the patient and sample clinical tables;
- resolves treatment, receptor, survival, and timing fields;
- reports candidate HR+/HER2- treated/control/event counts;
- inspects RNA, CNA, methylation, mutation, and cleaned-RNA schemas;
- measures sample overlap without loading the large matrices fully into memory;
- decides whether an exact TCGA day-180 replication is supportable;
- does not estimate a treatment effect.

Single terminal log:

```text
results/logs/metabric_stage_m1_data_design_audit_YYYYMMDD_HHMMSS.log
```
