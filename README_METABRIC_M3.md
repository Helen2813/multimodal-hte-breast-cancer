# METABRIC Stage M3: TCGA source and harmonization audit

Run from the project root:

```powershell
.\run_metabric_stage_m3_tcga_source_harmonization_audit.ps1
```

M3 does not rerun M1/M2 and does not fit any outcome or treatment-effect model.

It:

- verifies the M2 transport-readiness decision;
- builds exact METABRIC RNA/CNA/mutation/methylation gene universes;
- scans local `data/` and `results/` files for TCGA clinical and omics source candidates;
- ranks candidates by filename and schema evidence;
- audits gene-symbol versus Ensembl identifiers and direct overlap;
- searches for local annotation and GMT pathway resources;
- creates a non-locked M4 source-selection template;
- prints every candidate and decision check into one terminal log.

Main log:

```text
results/logs/metabric_stage_m3_tcga_source_harmonization_audit_YYYYMMDD_HHMMSS.log
```
