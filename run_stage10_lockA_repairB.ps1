$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage10_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 10 full transcript: $LOG"

    $SCRIPTS = @(
        "scripts\34_stage10_preflight.py",
        "scripts\34_build_paperA_candidate.py",
        "scripts\35_repair_paperB_power.py",
        "scripts\36_generate_stage10_decision.py"
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
    Write-Host "STAGE 10 COMPLETED SUCCESSFULLY"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "  paper_A_treatment_effects\analysis_plan_CANDIDATE.md"
    Write-Host "  paper_A_treatment_effects\primary_estimand_CANDIDATE.json"
    Write-Host "  results\tables\34_paperA_candidate_summary.csv"
    Write-Host "  results\tables\35_paperB_repaired_observed_pilot.csv"
    Write-Host "  results\tables\35_paperB_repaired_power_summary.csv"
    Write-Host "  results\tables\35_paperB_repaired_power_gate.csv"
    Write-Host "  results\tables\36_stage10_decision.md"
    Write-Host ""
    Write-Host "Full transcript saved to: $LOG"
}
finally {
    Stop-Transcript
}
