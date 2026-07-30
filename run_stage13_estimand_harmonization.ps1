$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage13_estimand_harmonization_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host "Stage 13 transcript: $LOG"
    Write-Host "This stage does NOT run the 300/200 publication bootstrap."
    Write-Host "It extends only the checkpointed centering pilot to 30/30 when the Stage 12 interface is discoverable."

    $SCRIPTS_BEFORE_EXTENSION = @(
        "scripts\45_stage13_preflight.py",
        "scripts\46_compare_estimands_and_targets.py",
        "scripts\47_audit_stage12_centering.py"
    )

    foreach ($script in $SCRIPTS_BEFORE_EXTENSION) {
        Write-Host ("#" * 120)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 120)
        & $PY (Join-Path $ROOT $script)
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 120)
    Write-Host "RUNNING scripts\48_extend_centering_pilot.py --target 30"
    Write-Host ("#" * 120)
    & $PY (Join-Path $ROOT "scripts\48_extend_centering_pilot.py") --target 30
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed: scripts\48_extend_centering_pilot.py"
    }

    $SCRIPTS_AFTER_EXTENSION = @(
        "scripts\47_audit_stage12_centering.py",
        "scripts\49_validate_ccw_invariants.py",
        "scripts\50_generate_stage13_decision.py"
    )

    foreach ($script in $SCRIPTS_AFTER_EXTENSION) {
        Write-Host ("#" * 120)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 120)
        & $PY (Join-Path $ROOT $script)
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 120)
    Write-Host "STAGE 13 COMPLETED"
    Write-Host ("#" * 120)
    Write-Host "Review:"
    Write-Host "results\tables\46_estimand_harmonization.md"
    Write-Host "results\tables\47_bootstrap_centering_audit.csv"
    Write-Host "results\tables\49_ccw_invariant_checks.csv"
    Write-Host "results\tables\50_stage13_decision.md"
    Write-Host "Full transcript: $LOG"
}
finally {
    Stop-Transcript
}
