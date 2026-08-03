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
$LOG = Join-Path $LOG_DIR "stage33_sequence_simulation_pilot_v2_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 33 - REVISED EMPIRICALLY ANCHORED SEQUENCING SIMULATION PILOT"
    Write-Host ("#" * 128)
    Write-Host "Twelve scenarios: n=559/1118 x no/half/empirical sequencing x null/benefit."
    Write-Host "One hundred pilot repetitions per scenario."
    Write-Host "Naive bias is decomposed into target drift and residual omitted-sequencing bias."
    Write-Host "Omitted chemotherapy SMD is reported before and after naive overlap weighting."
    Write-Host "Independent timing-ascertainability thinning reproduces the strict V10 sample-size scale."
    Write-Host "The empirically calibrated benefit is an observed-risk-matching scenario, not known causal truth."
    Write-Host "The runner checkpoints each method/repetition and resumes automatically."
    Write-Host "No Candidate V9/V10 file and no manuscript prose is modified."

    $scripts = @(
        "scripts\124_stage33_lock_simulation_pilot_v2.py",
        "scripts\125_stage33_run_simulation_pilot_v2.py",
        "scripts\126_stage33_summarize_simulation_pilot_v2.py"
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
    Write-Host "PAPER A STAGE 33 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage33_sequence_simulation_pilot_v2"
    Write-Host "Figures: results\figures\stage33_sequence_simulation_pilot_v2"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
