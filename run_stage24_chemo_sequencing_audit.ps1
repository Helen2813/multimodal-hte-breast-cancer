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
$LOG = Join-Path $LOG_DIR "stage24_chemo_sequencing_audit_$STAMP.log"

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 124)
    Write-Host "PAPER A STAGE 24 - CHEMOTHERAPY SEQUENCING AUDIT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages 1-23 are NOT rerun."
    Write-Host "Candidate V9 code, config, manifest, bootstrap, and results are NOT modified."
    Write-Host "No treatment-effect model is fitted."
    Write-Host "This stage diagnoses chemotherapy timing relative to day 180 and hormone initiation."

    & $PYTHON "scripts\stage24_chemo_sequencing_audit.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 24 audit failed."
    }

    Write-Host ("#" * 124)
    Write-Host "PAPER A STAGE 24 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\stage24_chemo_sequencing"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
