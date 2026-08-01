# METABRIC M9 analysis-only repair

The original M9 run successfully completed M45-M48 and stopped at M49 because
`read_rows` was not present in `_metabric_m9_utils.py`.

The original M48 output also showed that numeric p-values were accidentally
accepted as gene-like tokens. This patch corrects both issues.

Run:

```powershell
.\run_metabric_stage_m9_resume_analysis_only.ps1
```

It reruns only:

- corrected M48 methylation annotation audit;
- M49 numerical claim table and figures.

It does **not** generate manuscript prose, a manuscript `.tex` file, titles,
Results, Discussion, or any other article text.

Expected outputs:

```text
results/tables/metabric_m9/m49_final_claim_table.csv
results/tables/metabric_m9/m49_m9_numerical_report.json
results/figures/metabric_m9/m49_incremental_c_index_forest.png
results/figures/metabric_m9/m49_chance_adjusted_stability.png
```
