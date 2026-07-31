$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location $ROOT

$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    throw "Project virtual environment not found: $PY"
}

$LOCK = Join-Path $ROOT "paper_A_treatment_effects\PROTOCOL_LOCKED_CANDIDATE_V9.txt"
if (Test-Path $LOCK) {
    throw "Candidate V9 is already locked. This runner refuses to repeat or overwrite the protocol lock."
}

$LOGDIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR ("stage20_candidate_v9_protocol_lock_" + $STAMP + ".log")

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 124)
    Write-Host "STAGE 20 - CANDIDATE V9 PROTOCOL LOCK"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages 15 through 19 are NOT rerun."
    Write-Host "The 300-repetition publication bootstrap is NOT started."
    Write-Host "This run performs only:"
    Write-Host "  79 - calculate the final 20-partition point estimator"
    Write-Host "  80 - write the Candidate V9 final protocol and hash manifest"
    Write-Host "  81 - verify every locked hash and semantic setting"

    $scripts = @(
        "scripts\79_final_20_partition_point_estimate.py",
        "scripts\80_create_candidate_v9_protocol_lock.py",
        "scripts\81_verify_candidate_v9_protocol_lock.py"
    )
    foreach ($script in $scripts) {
        Write-Host ("#" * 124)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 124)
        & $PY $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 124)
    Write-Host "STAGE 20 COMPLETED - CANDIDATE V9 LOCKED"
    Write-Host ("#" * 124)
    Get-Content "results\tables\81_candidate_v9_lock_integrity_summary.csv"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
