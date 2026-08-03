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
    "results\tables\stage25b_candidate_v10_balance_repair\s25b_97_balance_repair_summary.json",
    "results\tables\stage25b_candidate_v10_balance_repair\s25b_97_balance_repair_gates.csv"
)
foreach ($relative in $REQUIRED) {
    $path = Join-Path $ROOT $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input missing: $relative"
    }
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage25c_candidate_v10_unclipped_ato_lock_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 25C - UNCLIPPED STABILIZED ATO DESIGN AND LOCK"
    Write-Host ("#" * 128)
    Write-Host "The frozen 271-patient V10 population is reused."
    Write-Host "Candidate V9 remains immutable."
    Write-Host "No RMST effect, censoring model, outcome model, or publication bootstrap is run."
    Write-Host "Stage 25C validates the actual algebraically stabilized overlap score."
    Write-Host "A 300-resample propensity-only feasibility audit is performed."

    $scripts = @(
        "scripts\99_stage25c_validate_unclipped_ato_design.py",
        "scripts\100_stage25c_lock_candidate_v10_protocol.py"
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
    Write-Host "PAPER A STAGE 25C COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Results: results\tables\stage25c_candidate_v10_unclipped_ato"
    Write-Host "Protocol: paper_A_treatment_effects\candidate_v10"
    Write-Host "No Candidate V10 treatment effect was computed."
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
