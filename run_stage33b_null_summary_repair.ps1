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
$LOG = Join-Path $LOG_DIR "stage33b_null_summary_repair_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 33B - NULL-SCENARIO SUMMARY REPAIR"
    Write-Host ("#" * 128)
    Write-Host "No simulation repetition is rerun."
    Write-Host "The unchanged Stage 33 checkpoint is reparsed with keep_default_na=False."
    Write-Host "This restores rows whose effect_regime literal was 'null'."
    Write-Host "Original Stage 33 outputs remain untouched."
    Write-Host "All 36 scenario-method summaries and pre-specified gates are reconstructed."

    $scripts = @(
        "scripts\127_stage33b_audit_null_parse.py",
        "scripts\128_stage33b_repair_summary.py"
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
    Write-Host "PAPER A STAGE 33B COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage33b_sequence_simulation_summary_repair"
    Write-Host "Figures: results\figures\stage33b_sequence_simulation_summary_repair"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
