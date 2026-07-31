$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$REQUIRED = @(
    "results\tables\56_stage15_preflight_checks.csv",
    "results\tables\53_ccw_trace_candidate_manifest.csv",
    "data\derived\stage14_trace\53_candidate_01.csv",
    "scripts\41_replicate_estimators.py",
    "scripts\_stage12_utils.py"
)

foreach ($relativePath in $REQUIRED) {
    $fullPath = Join-Path $ROOT $relativePath
    if (-not (Test-Path $fullPath)) {
        throw "Required prior-stage input not found: $fullPath"
    }
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage15_resume_exact_rowmap_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 56 output found."
    Write-Host "Recovering the exact runtime CCW row_id -> patient-ID map before Stage 57."
    Write-Host "Full transcript: $LOG"
    Write-Host "The 300/200 publication bootstrap is NOT started."

    $SCRIPTS = @(
        "scripts\60_capture_exact_ccw_row_map.py",
        "scripts\57_common_target_estimator_bridge.py",
        "scripts\58_reestimated_ccw_truncation_bootstrap.py",
        "scripts\59_generate_stage15_decision.py"
    )

    foreach ($script in $SCRIPTS) {
        Write-Host ("#" * 120)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 120)
        & $PY (Join-Path $ROOT $script)
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 120)
    Write-Host "STAGE 15 EXACT-ROWMAP RESUME COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "results\tables\60_exact_runtime_row_map_diagnostics.csv"
    Write-Host "results\tables\57_bridge_diagnostics.csv"
    Write-Host "results\tables\57_common_target_estimator_bridge.csv"
    Write-Host "results\tables\58_reestimated_truncation_bootstrap_summary.csv"
    Write-Host "results\tables\59_stage15_decision.md"
    Write-Host "Full transcript: $LOG"
}
finally {
    Stop-Transcript
}
