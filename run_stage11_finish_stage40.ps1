$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage11_finish_stage40_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stages 37-39 will not be rerun."
    Write-Host "Stage 40 full transcript: $LOG"

    $SCRIPTS = @(
        "scripts\40_stage11_finish_preflight.py",
        "scripts\40_update_paperA_candidate.py"
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
    Write-Host "STAGE 11 COMPLETED SUCCESSFULLY"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "  paper_A_treatment_effects\analysis_plan_CANDIDATE_V2.md"
    Write-Host "  paper_A_treatment_effects\primary_estimand_CANDIDATE_V2.json"
    Write-Host "  results\tables\40_stage11_design_decision.csv"
    Write-Host "  results\tables\40_stage11_design_decision.md"
    Write-Host ""
    Write-Host "Full transcript saved to: $LOG"
}
finally {
    Stop-Transcript
}
