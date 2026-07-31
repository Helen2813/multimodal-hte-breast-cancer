$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage15_target_bridge_and_truncation_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 15 transcript: $LOG"
    Write-Host "The 300/200 publication bootstrap is NOT started."
    Write-Host "This stage runs three checkpointed 30-repetition CCW truncation sensitivities."

    $SCRIPTS = @(
        "scripts\56_stage15_preflight.py",
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
    Write-Host "STAGE 15 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "results\tables\57_common_target_estimator_bridge.csv"
    Write-Host "results\tables\57_bridge_component_differences.csv"
    Write-Host "results\tables\58_reestimated_truncation_bootstrap_summary.csv"
    Write-Host "results\tables\58_reestimated_truncation_paired_summary.csv"
    Write-Host "results\tables\59_stage15_decision.md"
    Write-Host "Full transcript: $LOG"
}
finally {
    Stop-Transcript
}
