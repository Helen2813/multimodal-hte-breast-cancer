$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw ".venv not found. Run setup_venv_and_run_00_to_06.ps1 first."
}

$SCRIPTS = @(
    "scripts\10_audit_treatment_timing.py",
    "scripts\11_audit_complete_omics_selection.py",
    "scripts\12_refine_complete_case_overlap.py"
)

foreach ($script in $SCRIPTS) {
    $path = Join-Path $ROOT $script
    if (-not (Test-Path $path)) {
        throw "Missing script: $path"
    }
    Write-Host "`nRunning $script ..."
    & $PY $path
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed: $script"
    }
}

Write-Host "`nStages 10-12 completed."
Write-Host "Review:"
Write-Host "  results\tables\10_treatment_timing_by_family.csv"
Write-Host "  results\tables\11_complete_omics_selection_summary.csv"
Write-Host "  results\tables\12_restricted_complete_case_summary.csv"
