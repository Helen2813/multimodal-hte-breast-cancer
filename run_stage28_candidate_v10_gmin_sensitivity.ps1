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
$LOG = Join-Path $LOG_DIR "stage28_candidate_v10_gmin_sensitivity_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 28 - CANDIDATE V10 G-MIN SENSITIVITY"
    Write-Host ("#" * 128)
    Write-Host "Primary G-min=0.10 is reproduced as an identity check."
    Write-Host "Post hoc diagnostic point estimates use G-min=0.15 and 0.20."
    Write-Host "The frozen V10 cohort, propensity, score, learners, folds, and seeds are unchanged."
    Write-Host "No patient bootstrap and no manuscript text are generated."

    $scripts = @(
        "scripts\109_stage28_lock_gmin_sensitivity.py",
        "scripts\110_stage28_compute_gmin_sensitivity.py",
        "scripts\111_stage28_summarize_gmin_sensitivity.py"
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
    Write-Host "PAPER A STAGE 28 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage28_candidate_v10_gmin_sensitivity"
    Write-Host "Figure: results\figures\stage28_candidate_v10_gmin_sensitivity"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
