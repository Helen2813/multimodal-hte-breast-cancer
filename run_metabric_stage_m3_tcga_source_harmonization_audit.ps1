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
$LOG = Join-Path $LOG_DIR "metabric_stage_m3_tcga_source_harmonization_audit_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M3 - TCGA SOURCE AND HARMONIZATION AUDIT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1 and M2 are NOT rerun."
    Write-Host "No treatment-effect, survival, HTE, or modality-utility model is fitted."
    Write-Host "No top-ranked TCGA source is silently accepted."
    Write-Host "Everything needed for the M4 source lock is printed to this single log."

    $scripts = @(
        "scripts\m10_verify_m2_and_metabric_gene_universes.py",
        "scripts\m11_discover_tcga_harmonization_sources.py",
        "scripts\m12_audit_candidate_identifiers_and_overlap.py",
        "scripts\m13_generate_m4_source_selection_template.py",
        "scripts\m14_generate_m3_decision.py"
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
    Write-Host "METABRIC STAGE M3 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m3"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
