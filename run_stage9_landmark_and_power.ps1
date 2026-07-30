$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage9_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 9 full console transcript: $LOG"

    $SCRIPTS = @(
        "scripts\29_build_landmark_cohorts.py",
        "scripts\30_landmark_balance.py",
        "scripts\31_landmark_ai_aipw.py",
        "scripts\32_paperB_landmark_power.py",
        "scripts\33_generate_stage9_decision.py"
    )

    foreach ($script in $SCRIPTS) {
        $path = Join-Path $ROOT $script
        if (-not (Test-Path $path)) {
            throw "Missing script: $path"
        }

        Write-Host ""
        Write-Host ("#" * 120)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 120)

        & $PY $path

        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ""
    Write-Host ("#" * 120)
    Write-Host "STAGE 9 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "  results\tables\29_landmark_cohort_summary.csv"
    Write-Host "  results\tables\30_landmark_balance_summary.csv"
    Write-Host "  results\tables\31_landmark_paperA_gate.csv"
    Write-Host "  results\tables\31_landmark_aipw_results.csv"
    Write-Host "  results\tables\32_paperB_landmark_observed_pilot.csv"
    Write-Host "  results\tables\32_paperB_power_grid.csv"
    Write-Host "  results\tables\32_paperB_power_gate.csv"
    Write-Host "  results\tables\33_stage9_two_paper_decision.md"
    Write-Host ""
    Write-Host "Full transcript saved to: $LOG"
}
finally {
    Stop-Transcript
}
