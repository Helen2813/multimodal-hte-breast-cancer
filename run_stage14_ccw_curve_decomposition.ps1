$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage14_ccw_curve_decomposition_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 14 transcript: $LOG"
    Write-Host "The publication bootstrap is NOT started."
    Write-Host "Stage 41 is re-run once under a narrow trace only to capture the in-memory CCW clone table."

    $SCRIPTS = @(
        "scripts\51_stage14_preflight.py",
        "scripts\52_audit_bootstrap_weight_instability.py",
        "scripts\53_capture_ccw_analysis_state.py",
        "scripts\54_export_ccw_curves_and_decompose.py",
        "scripts\55_generate_stage14_decision.py"
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
    Write-Host "STAGE 14 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "results\tables\52_ccw_bootstrap_weight_audit.csv"
    Write-Host "results\tables\53_ccw_trace_candidate_manifest.csv"
    Write-Host "results\tables\54_ccw_rmst_decomposition.csv"
    Write-Host "results\tables\54_ccw_fixed_weight_cap_sensitivity.csv"
    Write-Host "results\tables\55_stage14_decision.md"
    Write-Host "Full transcript: $LOG"
}
finally {
    Stop-Transcript
}
