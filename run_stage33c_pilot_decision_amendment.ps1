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
$LOG = Join-Path $LOG_DIR "stage33c_pilot_decision_amendment_$STAMP.log"

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 33C - PILOT DECISION AMENDMENT"
    Write-Host ("#" * 128)
    Write-Host "No simulation is rerun."
    Write-Host "Coverage validity is required only for the two intended valid estimators."
    Write-Host "The naive estimator remains an intentionally misspecified diagnostic comparator."
    & $PYTHON "scripts\129_stage33c_amend_pilot_decision.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 33C failed."
    }
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 33C COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
