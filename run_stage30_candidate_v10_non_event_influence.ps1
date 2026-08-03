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
$LOG = Join-Path $LOG_DIR "stage30_candidate_v10_non_event_influence_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 30 - TOP-INFLUENCE NON-EVENT LEAVE-ONE-OUT"
    Write-Host ("#" * 128)
    Write-Host "The top 10 non-event patients are selected from the locked Stage 26 absolute-influence ranking."
    Write-Host "The selected set is locked before any leave-one-out refit."
    Write-Host "Each deletion refits propensity, censoring, and outcome nuisances over 20 partitions."
    Write-Host "Stage 29 event-patient results are combined only at the summary stage."
    Write-Host "No patient bootstrap, new confidence interval, or manuscript text is generated."
    Write-Host "The runner checkpoints after every successful deletion and resumes automatically."

    $scripts = @(
        "scripts\115_stage30_lock_top_non_event_set.py",
        "scripts\116_stage30_run_top_non_event_leave_one_out.py",
        "scripts\117_stage30_summarize_non_event_influence.py"
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
    Write-Host "PAPER A STAGE 30 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage30_candidate_v10_non_event_influence"
    Write-Host "Figure: results\figures\stage30_candidate_v10_non_event_influence"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
