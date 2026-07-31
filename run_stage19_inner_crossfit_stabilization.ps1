$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage19_inner_crossfit_stabilization_$STAMP.log"

if (-not (Test-Path $PY)) {
    throw "Python virtual environment not found: $PY"
}

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 124)
    Write-Host "STAGE 19 - INNER CROSS-FIT STABILIZATION"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages 15 through 18 are NOT rerun."
    Write-Host "No new patient bootstrap samples are drawn."
    Write-Host "The same 30 Stage 18 bootstrap samples are extended from 5 to 20 nuisance partitions."
    Write-Host "The 300-repetition publication bootstrap is NOT started."

    $scripts = @(
        "scripts\75_stage19_protocol_correction.py",
        "scripts\76_extend_bootstrap_partitions.py",
        "scripts\77_assess_inner_crossfit_convergence.py",
        "scripts\78_generate_stage19_decision.py"
    )

    foreach ($script in $scripts) {
        $path = Join-Path $ROOT $script
        if (-not (Test-Path $path)) {
            throw "Missing script: $path"
        }
        Write-Host ("#" * 124)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 124)
        & $PY $path
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 124)
    Write-Host "STAGE 19 COMPLETED"
    Write-Host ("#" * 124)
    $report = Join-Path $ROOT "results\tables\78_stage19_decision.md"
    if (Test-Path $report) {
        Write-Host "FINAL DECISION REPORT"
        Write-Host ("-" * 124)
        Get-Content $report
        Write-Host ("-" * 124)
    }
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
