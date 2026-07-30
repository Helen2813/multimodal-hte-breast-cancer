$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

foreach ($required in @(
    (Join-Path $ROOT "results\tables\31_landmark_paperA_gate.csv"),
    (Join-Path $ROOT "results\tables\31_landmark_aipw_results.csv")
)) {
    if (-not (Test-Path $required)) {
        throw "Required Stage 31 output not found: $required"
    }
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage9_v5_finish_from32_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 31 outputs found."
    Write-Host "Stage 32 preflight will assemble the exact modeling table."
    Write-Host "Full transcript: $LOG"

    $SCRIPTS = @(
        "scripts\32_preflight_stage9_inputs.py",
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
    Write-Host "STAGE 9 COMPLETED SUCCESSFULLY"
    Write-Host ("#" * 120)
    Write-Host "Review:"
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
