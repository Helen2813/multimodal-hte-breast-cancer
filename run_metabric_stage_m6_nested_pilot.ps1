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
$LOG = Join-Path $LOG_DIR "metabric_stage_m6_nested_pilot_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M6 - HISTORICAL RECONSTRUCTION AND DUAL-TRACK NESTED PILOT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M5 are NOT rerun."
    Write-Host "M5 protocol hashes are verified before modeling."
    Write-Host "Track A uses the fixed TCGA panel and outcome-blind mapping."
    Write-Host "Track B performs one prespecified five-fold nested pilot."
    Write-Host "All supervised screening and IAMB selection occur inside training folds."
    Write-Host "This is a pilot, not the final publication analysis."

    $scripts = @(
        "scripts\m31_verify_lock_and_discover_history.py",
        "scripts\m32_validate_reconstructed_iamb.py",
        "scripts\m33_run_track_a_external_transport_pilot.py",
        "scripts\m34_run_track_b_nested_pilot.py",
        "scripts\m35_generate_m6_pilot_decision.py"
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
    Write-Host "METABRIC STAGE M6 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m6"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
