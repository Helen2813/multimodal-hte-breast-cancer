# Stage 2 — compact clinical adjustment and overlap diagnostics

Copy these files into the root of `multimodal-hte-breast-cancer/`.

They add:

```text
scripts/_compact_adjustment.py
scripts/05_build_compact_adjustment.py
scripts/06_run_compact_overlap.py
setup_venv_and_run_00_to_06.ps1
run_stage2_compact_overlap.ps1
```

## Recommended first run

Open PowerShell in the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_venv_and_run_00_to_06.ps1
```

This script:

1. creates `.venv` using Python 3.10 if needed;
2. installs `requirements-initial.txt` inside `.venv`;
3. writes `requirements-lock.txt`;
4. reruns stages `00–04`;
5. runs new stages `05–06`.

It invokes `.venv\Scripts\python.exe` explicitly, so activation is not required.

## Later rerun of only Stage 2

```powershell
.\run_stage2_compact_overlap.ps1
```

## New outputs

```text
data/derived/compact_adjustment/
results/tables/05_compact_adjustment_summary.csv
results/tables/06_compact_overlap_summary.csv
results/tables/06_legacy_vs_compact_overlap.csv
results/tables/06_compact_balance_*.csv
results/tables/06_compact_propensity_*.csv
results/figures/06_compact_overlap_*.png
```

The compact adjustment set is constructed from baseline age, AJCC stage/T/N/M,
lymph-node counts, grade when available, and receptor variables only when they vary
inside the cohort. It excludes treatment, outcomes, omics, tissue location, and staging
system edition.

Propensity scores are out-of-fold and use L2-regularized logistic regression.
The script reports trimming, stabilized IPTW, overlap weighting, effective sample size,
and standardized mean differences.
