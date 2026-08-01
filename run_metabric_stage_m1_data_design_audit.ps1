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
$LOG = Join-Path $LOG_DIR "metabric_stage_m1_data_design_audit_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M1 - READ-ONLY DATA AND DESIGN AUDIT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Raw directory: data\raw\metabric"
    Write-Host "No treatment-effect estimator is run."
    Write-Host "No raw file is modified or copied."
    Write-Host "Large omics matrices are inspected by streaming, not loaded fully into memory."

    $scripts = @(
        "scripts\m01_inventory_and_preflight.py",
        "scripts\m02_clinical_and_timing_audit.py",
        "scripts\m03_omics_overlap_audit.py",
        "scripts\m04_generate_design_decision.py"
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
    Write-Host "METABRIC STAGE M1 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m1"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
