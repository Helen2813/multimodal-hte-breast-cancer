param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    throw "Python virtual environment not found: $PYTHON"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m9_final_inference_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M9 - FINAL INFERENCE, STABILITY CALIBRATION, AND MANUSCRIPT SCAFFOLD"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M8 are NOT rerun."
    Write-Host "M9 does not refit the M7/M8 prediction models."
    Write-Host "M9 runs 2000 paired patient bootstraps of locked OOF predictions."
    Write-Host "The bootstrap is conditional on the fitted repeated models."
    Write-Host "Feature stability is calibrated against random expected overlap."
    Write-Host "Methylation probe-to-gene transport is audited before final wording."
    Write-Host "A new manuscript directory is created; the original ITE paper is not modified."

    $scripts = @(
        "scripts\m45_lock_m9_final_inference_protocol.py",
        "scripts\m46_bootstrap_repeated_oof_predictions.py",
        "scripts\m47_compute_chance_adjusted_stability.py",
        "scripts\m48_audit_methylation_transport.py",
        "scripts\m49_generate_final_claims_and_manuscript.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 124)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 124)
        & $PYTHON $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M9 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m9"
    Write-Host "Figures: results\figures\metabric_m9"
    Write-Host "Manuscript scaffold: paper_cross_cohort_transport"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
