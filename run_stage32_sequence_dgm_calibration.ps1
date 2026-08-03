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
$LOG = Join-Path $LOG_DIR "stage32_sequence_dgm_calibration_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 32 - EMPIRICALLY ANCHORED SEQUENCING-DGM CALIBRATION"
    Write-Host ("#" * 128)
    Write-Host "This does not rerun or modify Candidate V9 or Candidate V10."
    Write-Host "Observed V9/V10 margins are used only to calibrate a future simulation regime."
    Write-Host "Treatment/sequence margins are calibrated first; sparse arm-specific event risks second."
    Write-Host "A larger independent Sobol sample validates the calibrated parameters."
    Write-Host "No confirmatory simulation and no manuscript text are generated."

    $scripts = @(
        "scripts\121_stage32_lock_dgm_calibration.py",
        "scripts\122_stage32_run_dgm_calibration.py",
        "scripts\123_stage32_summarize_dgm_calibration.py"
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
    Write-Host "PAPER A STAGE 32 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage32_sequence_dgm_calibration"
    Write-Host "Figure: results\figures\stage32_sequence_dgm_calibration"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
