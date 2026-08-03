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
$LOG = Join-Path $LOG_DIR "stage26_candidate_v10_point_estimate_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 26 - LOCKED CANDIDATE V10 POINT ESTIMATE"
    Write-Host ("#" * 128)
    Write-Host "Candidate V9 remains immutable."
    Write-Host "Candidate V10 protocol integrity is checked before calculation."
    Write-Host "Stage 26 calculation code is hashed before the effect is computed."
    Write-Host "The full-sample propensity is not clipped."
    Write-Host "Censoring and outcome nuisances are cross-fitted over 20 locked partitions."
    Write-Host "No patient bootstrap and no manuscript text are generated."

    $scripts = @(
        "scripts\101_stage26_lock_point_estimate_calculation.py",
        "scripts\102_stage26_compute_candidate_v10_point_estimate.py",
        "scripts\103_stage26_review_point_estimate_diagnostics.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 128)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 128)
        & $PYTHON $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 26 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Results: results\tables\stage26_candidate_v10_point_estimate"
    Write-Host "No publication bootstrap was run."
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
