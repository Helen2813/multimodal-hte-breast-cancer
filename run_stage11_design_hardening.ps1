$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage11_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 11 full transcript: $LOG"

    $SCRIPTS = @(
        "scripts\37_stage11_preflight.py",
        "scripts\37_audit_control_strategy_and_era.py",
        "scripts\38_generate_reporting_assets.py",
        "scripts\39_ccw_feasibility_audit.py",
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
    Write-Host "  results\tables\37_control_strategy_composition.csv"
    Write-Host "  results\tables\37_later_vs_never_balance.csv"
    Write-Host "  results\tables\37_era_interaction_feasibility.csv"
    Write-Host "  results\tables\38_table1_landmark180.csv"
    Write-Host "  results\tables\38_landmark_flow_counts.csv"
    Write-Host "  results\figures\38_landmark_flow.png"
    Write-Host "  results\figures\38_love_plot_landmark180.png"
    Write-Host "  results\figures\38_later_vs_never_love_plot.png"
    Write-Host "  results\tables\39_ccw_feasibility_decision.csv"
    Write-Host "  paper_A_treatment_effects\analysis_plan_CANDIDATE_V2.md"
    Write-Host "  results\tables\40_stage11_design_decision.md"
    Write-Host ""
    Write-Host "Full transcript saved to: $LOG"
}
finally {
    Stop-Transcript
}
