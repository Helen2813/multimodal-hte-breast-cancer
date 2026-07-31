$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    throw "Python virtual environment was not found: $PY"
}

$LOGDIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR "stage18_patient_bootstrap_pilot_$STAMP.log"

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 124)
    Write-Host "STAGE 18 - PATIENT-LEVEL BOOTSTRAP PILOT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages 15, 16, and 17 are NOT rerun."
    Write-Host "This run performs only new Stage 18 analyses:"
    Write-Host "  71 - protocol amendment and preflight"
    Write-Host "  72 - 30-repetition patient bootstrap pilot"
    Write-Host "  73 - pilot summary and composition diagnostics"
    Write-Host "  74 - decision gate"
    Write-Host "The 300-repetition publication bootstrap is NOT started."

    $scripts = @(
        "scripts\71_stage18_protocol_amendment.py",
        "scripts\72_patient_bootstrap_pilot.py",
        "scripts\73_summarize_bootstrap_pilot.py",
        "scripts\74_generate_stage18_decision.py"
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
    Write-Host "STAGE 18 COMPLETED"
    Write-Host ("#" * 124)
    $report = Join-Path $ROOT "results\tables\74_stage18_decision.md"
    if (Test-Path $report) {
        Write-Host "FINAL DECISION REPORT"
        Write-Host ("-" * 124)
        Get-Content $report
        Write-Host ("-" * 124)
    }
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
