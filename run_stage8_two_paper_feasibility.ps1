$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path $PY)) {
    throw ".venv not found."
}

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage8_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 8 full console transcript: $LOG"

    $SCRIPTS = @(
        "scripts\24_refit_compact_era_and_timing_gate.py",
        "scripts\25_censoring_and_ipcw_pseudooutcomes.py",
        "scripts\26_paperA_ai_aipw_feasibility.py",
        "scripts\27_paperB_ai_modality_feasibility.py",
        "scripts\28_generate_two_paper_feasibility_report.py"
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
    Write-Host "STAGE 8 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "  results\tables\24_compact_era_propensity_summary.csv"
    Write-Host "  results\tables\24_treatment_timing_gate_summary.csv"
    Write-Host "  results\tables\25_horizon_feasibility_gate.csv"
    Write-Host "  results\tables\25_censoring_model_summary.csv"
    Write-Host "  results\tables\26_paperA_ai_aipw_feasibility.csv"
    Write-Host "  results\tables\26_paperA_feasibility_gate.csv"
    Write-Host "  results\tables\27_paperB_ai_feasibility_summary.csv"
    Write-Host "  results\tables\28_two_paper_feasibility_report.md"
    Write-Host ""
    Write-Host "Full transcript saved to: $LOG"
}
finally {
    Stop-Transcript
}
