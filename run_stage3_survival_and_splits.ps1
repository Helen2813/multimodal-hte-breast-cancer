$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw ".venv not found. Run setup_venv_and_run_00_to_06.ps1 first."
}

$SCRIPTS = @(
    "scripts\07_audit_modality_availability.py",
    "scripts\08_create_repeated_outer_splits.py",
    "scripts\09_weighted_survival_baseline.py"
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

Write-Host "`nStages 07-09 completed."
Write-Host "Review:"
Write-Host "  results\tables\07_modality_availability_summary.csv"
Write-Host "  results\tables\08_repeated_outer_split_summary.csv"
Write-Host "  results\tables\09_weighted_survival_baseline.csv"
