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
$LOG = Join-Path $LOG_DIR "stage31_sequencing_simulation_pilot_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 31 - FOCUSED TREATMENT-SEQUENCING SIMULATION PILOT"
    Write-Host ("#" * 128)
    Write-Host "Six scenarios: n=300/600 crossed with no/moderate/strong sequencing."
    Write-Host "Fifty pilot repetitions per scenario."
    Write-Host "Methods: naive full cohort, sequencing-adjusted full cohort, and sequencing-aware restriction."
    Write-Host "The sequencing-aware method has a different target population."
    Write-Host "The runner checkpoints each method/repetition and resumes automatically."
    Write-Host "No Candidate V9/V10 files and no manuscript prose are modified."

    $scripts = @(
        "scripts\118_stage31_lock_simulation_pilot.py",
        "scripts\119_stage31_run_simulation_pilot.py",
        "scripts\120_stage31_summarize_simulation_pilot.py"
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
    Write-Host "PAPER A STAGE 31 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage31_sequencing_simulation_pilot"
    Write-Host "Figures: results\figures\stage31_sequencing_simulation_pilot"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
