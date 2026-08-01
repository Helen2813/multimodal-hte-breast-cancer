param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    throw "Python virtual environment not found: $PYTHON"
}

$M46 = Join-Path $ROOT "results\tables\metabric_m9\m46_oof_patient_bootstrap_summary.csv"
$M47 = Join-Path $ROOT "results\tables\metabric_m9\m47_chance_adjusted_stability_summary.csv"
if (-not (Test-Path $M46) -or -not (Test-Path $M47)) {
    throw "M46/M47 outputs not found. The original M9 run must complete M45-M47 first."
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m9_resume_analysis_only_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M9 - RESUME M48/M49, NUMERICAL OUTPUTS ONLY"
    Write-Host ("#" * 124)
    Write-Host "M45, M46, and M47 are NOT rerun."
    Write-Host "M48 corrects the methylation annotation audit."
    Write-Host "M49 creates numerical tables and figures only."
    Write-Host "No manuscript prose, LaTeX manuscript, title, Results, or Discussion is generated."

    $scripts = @(
        "scripts\m48_audit_methylation_transport.py",
        "scripts\m49_generate_final_claims_and_manuscript.py"
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
    Write-Host "METABRIC STAGE M9 ANALYSIS-ONLY RESUME COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m9"
    Write-Host "Figures: results\figures\metabric_m9"
    Write-Host "No manuscript text was generated."
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
