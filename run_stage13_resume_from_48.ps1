$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$PREFLIGHT = Join-Path $ROOT "results\tables\45_stage13_preflight.json"
$ESTIMAND = Join-Path $ROOT "results\tables\46_estimand_harmonization_summary.csv"

if (-not (Test-Path $PREFLIGHT)) {
    throw "Stage 45 output not found: $PREFLIGHT"
}
if (-not (Test-Path $ESTIMAND)) {
    throw "Stage 46 output not found: $ESTIMAND"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage13_resume_from48_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 45 and Stage 46 outputs found."
    Write-Host "Resuming Stage 13 from Stage 48."
    Write-Host "Full transcript: $LOG"
    Write-Host "The 300/200 publication bootstrap is NOT started."

    Write-Host ("#" * 120)
    Write-Host "RUNNING scripts\48_extend_centering_pilot.py --target 30"
    Write-Host ("#" * 120)
    & $PY (Join-Path $ROOT "scripts\48_extend_centering_pilot.py") --target 30
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed: scripts\48_extend_centering_pilot.py"
    }

    $SCRIPTS = @(
        "scripts\47_audit_stage12_centering.py",
        "scripts\49_validate_ccw_invariants.py",
        "scripts\50_generate_stage13_decision.py"
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
    Write-Host "STAGE 13 RESUME COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "results\tables\48_empty_error_csv_cleanup.csv"
    Write-Host "results\tables\48_centering_pilot_extension.csv"
    Write-Host "results\tables\47_bootstrap_centering_audit.csv"
    Write-Host "results\tables\49_ccw_invariant_checks.csv"
    Write-Host "results\tables\50_stage13_decision.md"
    Write-Host "Full transcript: $LOG"
}
finally {
    Stop-Transcript
}
