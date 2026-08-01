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
$LOG = Join-Path $LOG_DIR "metabric_stage_m7_full_core_analysis_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M7 - FULL TRACK A AND FULL RECONSTRUCTED TRACK B"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M6 are NOT rerun."
    Write-Host "M7 locks the full core protocol before scaling."
    Write-Host "Track A runs 1000 paired external patient bootstraps."
    Write-Host "Track B runs 20 repeats x 5 outer folds with checkpoints."
    Write-Host "Track B remains labelled reconstructed, not historical-IAMB exact."
    Write-Host "An interrupted M38 run resumes from completed repeat/fold checkpoints."

    $scripts = @(
        "scripts\m36_lock_m7_full_core_analysis.py",
        "scripts\m37_run_track_a_full_external_validation.py",
        "scripts\m38_run_track_b_full_repeated_nested.py",
        "scripts\m39_generate_m7_full_report.py"
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
    Write-Host "METABRIC STAGE M7 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m7"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
