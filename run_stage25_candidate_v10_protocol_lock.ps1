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
$LOG = Join-Path $LOG_DIR "stage25_candidate_v10_protocol_lock_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 25 - CANDIDATE V10 COHORT AND PROTOCOL LOCK"
    Write-Host ("#" * 128)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Candidate V9 is verified and remains immutable."
    Write-Host "Stage 25 reproduces the Stage 24 sequencing counts."
    Write-Host "Stage 25 excludes unascertainable chemotherapy start timing."
    Write-Host "Stage 25 evaluates propensity overlap and ATO balance."
    Write-Host "No Candidate V10 treatment-effect estimate is computed."

    $scripts = @(
        "scripts\94_stage25_build_candidate_v10_cohort.py",
        "scripts\95_stage25_v10_pre_effect_gates.py",
        "scripts\96_stage25_lock_candidate_v10_protocol.py"
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
    Write-Host "PAPER A STAGE 25 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Results: results\tables\stage25_candidate_v10"
    Write-Host "Derived cohort: data\derived\candidate_v10"
    Write-Host "Protocol: paper_A_treatment_effects\candidate_v10"
    Write-Host "No treatment-effect estimate was computed."
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
