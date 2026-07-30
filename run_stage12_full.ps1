$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOGDIR = Join-Path $ROOT "results\logs"
if (-not (Test-Path $PY)) { throw ".venv not found." }
foreach ($required in @(
    (Join-Path $ROOT "results\tables\42_landmark_bootstrap_summary.csv"),
    (Join-Path $ROOT "results\tables\43_ccw_bootstrap_summary.csv")
)) { if (-not (Test-Path $required)) { throw "Stage 12 pilot output not found: $required" } }
$env:STAGE12_LANDMARK_REPS = "300"
$env:STAGE12_CCW_REPS = "200"
New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage12_full_$STAMP.log"
Start-Transcript -Path $LOG -Force
try {
    Write-Host "Stage 12 FULL transcript: $LOG"
    Write-Host "Resuming existing checkpoints."
    $SCRIPTS = @(
        "scripts\41_stage12_preflight.py",
        "scripts\41_replicate_estimators.py",
        "scripts\42_landmark_full_pipeline_bootstrap.py",
        "scripts\43_ccw_sensitivity_bootstrap.py",
        "scripts\44_generate_stage12_report.py"
    )
    foreach ($script in $SCRIPTS) {
        $path = Join-Path $ROOT $script
        if (-not (Test-Path $path)) { throw "Missing script: $path" }
        Write-Host ""; Write-Host ("#" * 120); Write-Host "RUNNING $script"; Write-Host ("#" * 120)
        & $PY $path
        if ($LASTEXITCODE -ne 0) { throw "Script failed: $script" }
    }
    Write-Host ""; Write-Host ("#" * 120); Write-Host "STAGE 12 FULL INFERENCE COMPLETED SUCCESSFULLY"; Write-Host ("#" * 120)
    Write-Host "Full transcript saved to: $LOG"
} finally { Stop-Transcript }
