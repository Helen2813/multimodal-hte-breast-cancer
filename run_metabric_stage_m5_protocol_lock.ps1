param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    throw "Python virtual environment not found: $PYTHON"
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m5_protocol_lock_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M5 - RECIPE CORRECTION, PANEL RECOVERY, AND DUAL-TRACK PROTOCOL LOCK"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M4 are NOT rerun."
    Write-Host "No survival, feature-selection, HTE, or treatment-effect model is fitted."
    Write-Host "The official cBioPortal gene-panel API is queried once and cached."
    Write-Host "All protocol gates and final feature counts are printed to this log."

    $scripts = @(
        "scripts\m26_correct_paper1_recipe_registry.py",
        "scripts\m27_recover_official_metabric_173_panel.py",
        "scripts\m28_strict_fixed_panel_transportability_qc.py",
        "scripts\m29_endpoint_and_cohort_alignment_audit.py",
        "scripts\m30_lock_metabric_dual_track_protocol.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 124)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 124)
        & $PYTHON $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M5 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m5"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
