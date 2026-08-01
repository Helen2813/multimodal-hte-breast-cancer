# METABRIC Stage M4

Run from the project root:

```powershell
.\run_metabric_stage_m4_dual_track_harmonization.ps1
```

M4 prepares two scientifically separate tracks:

1. **Fixed TCGA-panel transport**
   - maps selected TCGA RNA/CNA Ensembl IDs to HGNC using current and GRCh37
     Ensembl REST endpoints;
   - caches and hashes every response;
   - builds outcome-blind METABRIC RNA and CNA matrices for mapped, assayed features;
   - audits mutation gene-panel coverage before any absent call is coded as wild-type.

2. **Independent Paper-1 replication in METABRIC**
   - recovers historical `summary_all_results.csv`, selected lists, candidate matrices,
     algorithms, alpha values, and source evidence;
   - writes a nested selection protocol draft;
   - does not execute selection or inspect METABRIC outcome for Track A.

No previous stage is rerun and no model is fitted.

Single log:

```text
results/logs/metabric_stage_m4_dual_track_harmonization_YYYYMMDD_HHMMSS.log
```
