$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

$BALANCE_SUMMARY = Join-Path $ROOT "results\tables\30_landmark_balance_summary.csv"
if (-not (Test-Path $BALANCE_SUMMARY)) {
    throw "Stage 30 output was not found. Run Stage 30 first."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage9_resume_from31_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 30 output found:"
    Write-Host "  $BALANCE_SUMMARY"
    Write-Host "Resuming Stage 9 from Stage 31."
    Write-Host "Designs marked LANDMARK_NOT_READY will be skipped automatically."
    Write-Host "Full transcript: $LOG"

    $SCRIPTS = @(
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
    Write-Host "STAGE 9 RESUME FROM 31 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "  results\tables\31_skipped_not_ready_designs.csv"
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
