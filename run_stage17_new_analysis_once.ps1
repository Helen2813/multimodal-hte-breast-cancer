$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

# This runner is intentionally one-time only. It refuses to mix with any
# previous or partial Stage 17 run.
$existingStage17 = @(
    @(
        "results\tables\68_repeated_crossfit_estimates_checkpoint.csv",
        "results\tables\69_repeated_estimate_summary.csv",
        "results\tables\70_stage17_decision.csv",
        "data\derived\stage17\68_primary_patient_scores_LOCAL_ONLY.csv"
    ) | ForEach-Object {
        Join-Path $ROOT $_
    } | Where-Object {
        Test-Path $_
    }
)

if ($existingStage17.Count -gt 0) {
    Write-Host "Stage 17 output/checkpoint files already exist:" -ForegroundColor Yellow
    $existingStage17 | ForEach-Object { Write-Host "  $_" }
    throw "One-time Stage 17 runner stopped to prevent an accidental repeat or mixed checkpoint run."
}

$env:PYTHONUNBUFFERED = "1"
$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage17_new_analysis_once_$STAMP.log"
$transcriptStarted = $false

try {
    Start-Transcript -Path $LOG -Force
    $transcriptStarted = $true

    Write-Host ("#" * 124)
    Write-Host "STAGE 17 - NEW ANALYSES ONLY"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages 15 and 16 are NOT rerun."
    Write-Host "Stage 66 exact-reconstruction preflight is skipped."
    Write-Host "This run performs only the new Stage 17 analyses:"
    Write-Host "  67 - influence and fold-2 forensics"
    Write-Host "  68 - prespecified nuisance-partition stability experiment"
    Write-Host "  69 - repeated-score aggregation"
    Write-Host "  70 - decision gate"
    Write-Host "The publication bootstrap is NOT started."

    $required = @(
        "scripts\_stage17_utils.py",
        "scripts\67_influence_and_fold2_forensics.py",
        "scripts\68_repeated_crossfit_stability.py",
        "scripts\69_repeated_score_aggregation.py",
        "scripts\70_generate_stage17_decision.py",
        "stage17_config.json",
        "scripts\_stage12_utils.py",
        "scripts\_stage16_utils.py",
        "results\tables\65_stage16_decision.csv"
    )

    foreach ($rel in $required) {
        $path = Join-Path $ROOT $rel
        if (-not (Test-Path $path)) {
            throw "Required Stage 17/Stage 16 input is missing: $rel"
        }
    }

    $scripts = @(
        "scripts\67_influence_and_fold2_forensics.py",
        "scripts\68_repeated_crossfit_stability.py",
        "scripts\69_repeated_score_aggregation.py",
        "scripts\70_generate_stage17_decision.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 124)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 124)
        & $PY (Join-Path $ROOT $script)
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 124)
    Write-Host "STAGE 17 NEW-ANALYSIS RUN COMPLETED"
    Write-Host ("#" * 124)

    $decision = Join-Path $ROOT "results\tables\70_stage17_decision.md"
    if (Test-Path $decision) {
        Write-Host "FINAL DECISION REPORT"
        Write-Host ("-" * 124)
        Get-Content $decision | ForEach-Object { Write-Host $_ }
        Write-Host ("-" * 124)
    }

    Write-Host "Keep this single log file: $LOG"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript
    }
}
