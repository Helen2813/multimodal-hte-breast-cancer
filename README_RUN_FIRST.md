# Initial multimodal HTE data pipeline

Copy the contents of this package into the root of:

```text
multimodal-hte-breast-cancer/
```

Your existing files must remain under:

```text
data/processed/
```

Expected layout:

```text
multimodal-hte-breast-cancer/
├── data/
│   ├── processed/
│   │   ├── 01_Clinical/
│   │   ├── 02_CNV/
│   │   ├── 03_Methylation/
│   │   ├── 04_miRNA/
│   │   ├── 05_Mutation/
│   │   ├── 06_proteins/
│   │   ├── 07_RNA/
│   │   ├── MERGE/
│   │   ├── MERGE_continuous_outer/
│   │   ├── output/
│   │   └── paper1_panels/
│   └── derived/
├── scripts/
├── results/
├── requirements-initial.txt
└── run_initial_pipeline.ps1
```

## Run in Windows PowerShell

Open PowerShell in the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_initial_pipeline.ps1
```

## Or run manually

```powershell
python -m pip install -r requirements-initial.txt
python scripts/00_validate_inputs.py
python scripts/01_audit_processed_data.py
python scripts/02_build_master_tables.py
python scripts/03_create_analysis_cohorts.py
python scripts/04_run_overlap_diagnostics.py
```

Do not continue after a failed step.

## First outputs to review

```text
data/derived/audits/00_input_validation.md
data/derived/audits/01_audit_report.md
data/derived/audits/02_join_summary.csv
results/tables/03_cohort_summary.csv
results/tables/04_overlap_summary.csv
```

The scripts never overwrite anything under `data/processed/`.
