$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage16_outcome_augmentation_audit_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 16 transcript: $LOG"
    Write-Host "The publication bootstrap is NOT started."
    Write-Host "All outcome-model variants use the same landmark cohort, folds, propensity scores, censoring model, and IPCW-RMST pseudo-outcome."

    $SCRIPTS = @(
        "scripts\61_stage16_preflight.py",
        "scripts\62_decompose_exact_landmark_aipw.py",
        "scripts\63_outcome_model_robustness.py",
        "scripts\64_fold_and_influence_stability.py",
        "scripts\65_generate_stage16_decision.py"
    )

    foreach ($script in $SCRIPTS) {
        Write-Host ("#" * 120)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 120)
        & $PY (Join-Path $ROOT $script)
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 120)
    Write-Host "STAGE 16 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "results\tables\62_exact_landmark_aipw_decomposition.csv"
    Write-Host "results\tables\63_outcome_model_robustness.csv"
    Write-Host "results\tables\63_outcome_model_calibration.csv"
    Write-Host "results\tables\64_fold_stability_summary.csv"
    Write-Host "results\tables\65_stage16_decision.md"
    Write-Host "Full transcript: $LOG"
}
finally {
    Stop-Transcript
}
