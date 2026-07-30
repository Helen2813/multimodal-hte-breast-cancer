$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage7_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 7 full console transcript: $LOG"

    $SCRIPTS = @(
        "scripts\20_audit_temporal_era_and_outcomes.py",
        "scripts\21_compare_propensity_strategies.py",
        "scripts\22_verified_survival_baseline.py",
        "scripts\23_create_verified_splits_and_plan_drafts.py"
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
    Write-Host "STAGE 7 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review these files:"
    Write-Host "  results\tables\20_event_definition_summary.csv"
    Write-Host "  results\tables\20_treatment_timing_coverage.csv"
    Write-Host "  results\tables\20_treatment_timing_by_family.csv"
    Write-Host "  results\tables\20_cohort_diagnosis_era_summary.csv"
    Write-Host "  results\tables\21_propensity_strategy_summary.csv"
    Write-Host "  results\tables\22_verified_survival_baseline.csv"
    Write-Host "  results\tables\23_verified_split_summary.csv"
    Write-Host "  paper_A_treatment_effects\analysis_plan_DRAFT.md"
    Write-Host "  paper_B_modality_utility\analysis_plan_DRAFT.md"
    Write-Host ""
    Write-Host "Full transcript saved to: $LOG"
}
finally {
    Stop-Transcript
}
