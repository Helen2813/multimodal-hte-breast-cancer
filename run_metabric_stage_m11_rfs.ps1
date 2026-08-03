param(
    [int]$Bootstrap = 2000,
    [int]$CheckpointEvery = 100
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = $PSScriptRoot
Set-Location $ROOT

$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOG_DIR = Join-Path $ROOT "results\logs"
$SCRIPTS = @(
    "scripts\m51_lock_rfs_sensitivity_protocol.py",
    "scripts\m52_run_track_b_full_repeated_nested_rfs.py",
    "scripts\m53_run_modality_specific_repeated_nested_rfs.py"
)

foreach ($relative in $SCRIPTS) {
    $path = Join-Path $ROOT $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required script not found: $path"
    }
}

New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m11_rfs_$STAMP.log"
$transcriptStarted = $false

try {
    Start-Transcript -Path $LOG -Force
    $transcriptStarted = $true

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M11 - RECURRENCE-FREE-SURVIVAL SENSITIVITY"
    Write-Host ("#" * 124)
    Write-Host "Primary OS analyses are NOT rerun."
    Write-Host "Track A is NOT rerun."
    Write-Host "Track B RFS uses the locked OS analysis settings and seeds."
    Write-Host "All output is written to results\tables\metabric_m11_rfs."
    Write-Host "Partial fold and bootstrap checkpoints are resumed automatically."
    Write-Host ("#" * 124)

    foreach ($relative in $SCRIPTS) {
        $path = Join-Path $ROOT $relative
        Write-Host ("=" * 124)
        Write-Host "RUNNING $relative"
        Write-Host ("=" * 124)
        & $PY $path
        if ($LASTEXITCODE -ne 0) {
            throw "$relative returned exit code $LASTEXITCODE. Keep the transcript and checkpoints."
        }
    }

    $BOOTSTRAP_SCRIPT = Join-Path $ROOT "scripts\m54_rfs_paired_patient_bootstrap.py"
    Write-Host ("=" * 124)
    Write-Host "RUNNING scripts\m54_rfs_paired_patient_bootstrap.py"
    Write-Host ("=" * 124)
    & $PY $BOOTSTRAP_SCRIPT --bootstrap $Bootstrap --checkpoint-every $CheckpointEvery
    if ($LASTEXITCODE -ne 0) {
        throw "RFS bootstrap returned exit code $LASTEXITCODE. Keep the transcript and checkpoints."
    }

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M11 RFS COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Upload this single log file:"
    Write-Host "  $LOG"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript
    }
}
