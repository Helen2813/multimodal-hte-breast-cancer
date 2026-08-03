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
$LOG = Join-Path $LOG_DIR "stage27b_candidate_v10_interval_repair_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 120)
    Write-Host "PAPER A STAGE 27B - INTERVAL SUMMARY AUDIT AND REPAIR"
    Write-Host ("#" * 120)
    Write-Host "No model or bootstrap estimate is rerun."
    Write-Host "The primary percentile interval must reproduce exactly."
    Write-Host "The locked studentized sensitivity interval is added from existing outputs."

    & $PYTHON "scripts\108_stage27b_repair_studentized_interval.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 27B failed."
    }

    Write-Host ("#" * 120)
    Write-Host "PAPER A STAGE 27B COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Results: results\tables\stage27b_candidate_v10_interval_repair"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
