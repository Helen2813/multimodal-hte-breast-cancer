param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    throw "Python virtual environment not found: $PYTHON"
}

$REQUIRED = @(
    "data\derived\candidate_v10\outer_hormone_hrpos_her2neg_landmark180_no_documented_chemo_by180_ascertainable.csv",
    "data\derived\candidate_v10\outer_hormone_hrpos_her2neg_landmark180_no_documented_chemo_by180_ascertainable_compact.csv",
    "results\tables\stage25_candidate_v10\s25_94_v10_cohort_summary.json",
    "results\tables\stage25_candidate_v10\s25_95_pre_effect_gates.csv",
    "results\tables\stage25_candidate_v10\s25_95_ato_balance.csv",
    "results\tables\stage25_candidate_v10\s25_95_pre_effect_summary.json"
)
foreach ($relative in $REQUIRED) {
    $path = Join-Path $ROOT $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required Stage 25 output missing: $relative"
    }
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage25b_candidate_v10_balance_repair_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 25B - OUTCOME-BLIND CANDIDATE V10 BALANCE REPAIR AND LOCK"
    Write-Host ("#" * 128)
    Write-Host "Stage 94 is NOT rerun. The frozen 271-patient V10 cohort is reused."
    Write-Host "Candidate V9 remains immutable."
    Write-Host "No RMST effect, survival outcome model, or bootstrap is run."
    Write-Host "The repair uses treatment and the locked 13 baseline variables only."

    $scripts = @(
        "scripts\97_stage25b_v10_balance_repair.py",
        "scripts\98_stage25b_lock_candidate_v10_protocol.py"
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
    Write-Host "PAPER A STAGE 25B COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Results: results\tables\stage25b_candidate_v10_balance_repair"
    Write-Host "Protocol: paper_A_treatment_effects\candidate_v10"
    Write-Host "No treatment-effect estimate was computed."
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
