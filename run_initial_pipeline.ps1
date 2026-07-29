$ErrorActionPreference = "Stop"

Write-Host "Installing initial requirements..."
python -m pip install -r requirements-initial.txt

Write-Host "`n[1/5] Validating inputs..."
python scripts/00_validate_inputs.py

Write-Host "`n[2/5] Auditing processed data..."
python scripts/01_audit_processed_data.py

Write-Host "`n[3/5] Building master tables..."
python scripts/02_build_master_tables.py

Write-Host "`n[4/5] Creating treatment-specific cohorts..."
python scripts/03_create_analysis_cohorts.py

Write-Host "`n[5/5] Running descriptive overlap diagnostics..."
python scripts/04_run_overlap_diagnostics.py

Write-Host "`nInitial pipeline completed."
Write-Host "Review:"
Write-Host "  data\derived\audits\"
Write-Host "  results\tables\03_cohort_summary.csv"
Write-Host "  results\tables\04_overlap_summary.csv"
