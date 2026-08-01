# METABRIC Stage M3B: corrected canonical TCGA feature bridge

Run from the project root:

```powershell
.\run_metabric_stage_m3b_canonical_feature_bridge.ps1
```

M3 incorrectly ranked merged cohort tables as separate RNA/CNA sources because it treated
many prefixed column names as generic gene symbols and did not strip modality prefixes.

M3B:

- prefers `data/derived/cohorts/master_outer.csv` as the canonical patient-level TCGA schema;
- classifies exact `CLIN_`, `RNA_`, `CNV_`/`CNA_`, `METH_`, `PROT_`, and `MUT_` columns;
- strips prefixes before gene matching;
- separates RNA Ensembl IDs, CNA/mutation symbols, and methylation CpGs;
- searches local files for mapping evidence;
- generates a non-locked M4 bridge template;
- never inspects METABRIC outcomes and fits no model.

Single log:

```text
results/logs/metabric_stage_m3b_canonical_feature_bridge_YYYYMMDD_HHMMSS.log
```
