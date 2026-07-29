$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

$SCRIPTS = @(
    "scripts\13_audit_receptor_sources.py",
    "scripts\14_verify_treatment_reconstruction.py",
    "scripts\15_balance_without_receptor_scores.py"
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

Write-Host "`nStages 13-15 completed."
Write-Host "Review:"
Write-Host "  results\tables\13_receptor_source_summary.csv"
Write-Host "  results\tables\13_receptor_source_audit.csv"
Write-Host "  results\tables\14_treatment_reconstruction_comparison.csv"
Write-Host "  results\tables\14_true_treatment_timing_fields.csv"
Write-Host "  results\tables\15_balance_sensitivity_summary.csv"
