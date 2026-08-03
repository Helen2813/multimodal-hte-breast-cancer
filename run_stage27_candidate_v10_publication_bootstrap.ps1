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
$LOG = Join-Path $LOG_DIR "stage27_candidate_v10_publication_bootstrap_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 27 - CANDIDATE V10 FULL PUBLICATION BOOTSTRAP"
    Write-Host ("#" * 128)
    Write-Host "Candidate V9 and the Candidate V10 protocol remain immutable."
    Write-Host "Stage 27 verifies an identity resample before bootstrapping."
    Write-Host "All 300 patient bootstrap samples refit propensity, censoring, and outcome nuisances."
    Write-Host "All copies of a source patient remain in one cross-fitting fold."
    Write-Host "The runner checkpoints after every repetition and resumes automatically."
    Write-Host "No manuscript text is generated."

    $scripts = @(
        "scripts\104_stage27_lock_publication_bootstrap.py",
        "scripts\105_stage27_identity_reproduction.py",
        "scripts\106_stage27_run_publication_bootstrap.py",
        "scripts\107_stage27_summarize_publication_bootstrap.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 128)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 128)
        & $PYTHON $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 27 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Results: results\tables\stage27_candidate_v10_bootstrap"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
