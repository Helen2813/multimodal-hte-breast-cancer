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
$LOG = Join-Path $LOG_DIR "metabric_stage_m3b_canonical_feature_bridge_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M3B - CANONICAL TCGA FEATURE BRIDGE"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M3 are NOT rerun."
    Write-Host "This stage corrects M3 source ranking by using exact modality prefixes in the canonical TCGA patient table."
    Write-Host "No treatment-effect, survival, HTE, or modality-utility model is fitted."
    Write-Host "No METABRIC outcome is inspected."

    $scripts = @(
        "scripts\m15_resolve_canonical_tcga_feature_schema.py",
        "scripts\m16_build_prefix_aware_direct_bridge.py",
        "scripts\m17_find_mapping_and_provenance_evidence.py",
        "scripts\m18_generate_corrected_bridge_decision.py"
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
    Write-Host "METABRIC STAGE M3B COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m3b"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
