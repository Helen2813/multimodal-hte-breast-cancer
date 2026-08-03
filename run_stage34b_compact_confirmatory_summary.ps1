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
$LOG = Join-Path $LOG_DIR "stage34b_compact_confirmatory_summary_$STAMP.log"

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 34B - COMPACT CONFIRMATORY SUMMARY"
    Write-Host ("#" * 128)
    Write-Host "No simulation is rerun."
    Write-Host "Existing Stage 34 summary files are read and condensed."
    & $PYTHON "scripts\133_stage34b_extract_compact_summary.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 34B failed."
    }
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 34B COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
