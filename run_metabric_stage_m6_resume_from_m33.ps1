param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    throw "Python virtual environment not found: $PYTHON"
}

$M32 = Join-Path $ROOT "results\tables\metabric_m6\m32_engine_decision.json"
if (-not (Test-Path $M32)) {
    throw "M32 output not found. The original M6 run must complete M31 and M32 before this resume runner."
}

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m6_resume_from_m33_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M6 - RESUME FROM M33 WITH ROBUST ALIGNMENT AND MEMORY-SAFE TRACK B"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "M31 and M32 are NOT rerun."
    Write-Host "The M5 protocol is not modified."
    Write-Host "M33 fixes the OS.time merge collision and audits TCGA ID overlap."
    Write-Host "M34 uses all 173 panel-aware mutation genes, not only GPS2."
    Write-Host "M34 keeps RNA and CNA matrices separate and screens them in chunks."
    Write-Host "Pilot bootstrap comparisons use paired patient resamples."

    $scripts = @(
        "scripts\m33_run_track_a_external_transport_pilot.py",
        "scripts\m34_run_track_b_nested_pilot.py",
        "scripts\m35_generate_m6_pilot_decision.py"
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
    Write-Host "METABRIC STAGE M6 RESUME COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m6"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
