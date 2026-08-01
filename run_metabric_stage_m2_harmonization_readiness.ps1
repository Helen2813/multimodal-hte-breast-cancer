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
$LOG = Join-Path $LOG_DIR "metabric_stage_m2_harmonization_readiness_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M2 - CORRECTED COVERAGE, PROVENANCE, AND TRANSPORT READINESS"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stage M1 is NOT rerun."
    Write-Host "No treatment-effect estimator is run."
    Write-Host "Raw METABRIC files are read-only."
    Write-Host "Patient/sample identifiers are never printed to the terminal."

    $scripts = @(
        "scripts\m05_correct_exact_omics_coverage.py",
        "scripts\m06_build_verified_clinical_master.py",
        "scripts\m07_validate_cleaned_rna_provenance.py",
        "scripts\m08_build_modality_availability_registry.py",
        "scripts\m09_generate_transport_readiness_decision.py"
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
    Write-Host "METABRIC STAGE M2 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m2"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
