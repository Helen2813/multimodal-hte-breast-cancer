$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

$SCRIPTS = @(
    "scripts\16_recover_observed_receptors.py",
    "scripts\17_verify_original_treatments.py",
    "scripts\18_rebuild_verified_cohorts.py",
    "scripts\19_recompute_verified_balance.py"
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

Write-Host "`nStages 16-19 completed."
Write-Host "Review:"
Write-Host "  results\tables\16_receptor_mode_recovery_summary.csv"
Write-Host "  results\tables\17_treatment_verification_summary.csv"
Write-Host "  results\tables\18_verified_cohort_summary.csv"
Write-Host "  results\tables\19_verified_balance_summary.csv"
