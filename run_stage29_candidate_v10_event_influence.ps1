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
$LOG = Join-Path $LOG_DIR "stage29_candidate_v10_event_influence_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 29 - LEAVE-ONE-EVENT-PATIENT-OUT INFLUENCE ANALYSIS"
    Write-Host ("#" * 128)
    Write-Host "The 36 observed event patients are locked before estimation."
    Write-Host "Each deletion refits propensity, censoring, and outcome nuisances over 20 partitions."
    Write-Host "Candidate V9 and the locked Candidate V10 primary analysis remain unchanged."
    Write-Host "No patient bootstrap and no manuscript text are generated."
    Write-Host "The runner checkpoints after every successful deletion and resumes automatically."

    $scripts = @(
        "scripts\112_stage29_lock_event_influence.py",
        "scripts\113_stage29_run_leave_one_event_out.py",
        "scripts\114_stage29_summarize_event_influence.py"
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
    Write-Host "PAPER A STAGE 29 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage29_candidate_v10_event_influence"
    Write-Host "Figure: results\figures\stage29_candidate_v10_event_influence"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
