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
$LOG = Join-Path $LOG_DIR "metabric_stage_m8_modality_concordance_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M8 - MODALITY-SPECIFIC, GENE, AND PATHWAY CONCORDANCE"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M7 are NOT rerun."
    Write-Host "M8 is not designed to rescue negative predictive results."
    Write-Host "Four modality-specific analyses use 10 repeats x 5 folds."
    Write-Host "Every supervised step remains inside the outer training fold."
    Write-Host "Gene concordance uses modality-specific assayable denominators."
    Write-Host "Reactome GMT is downloaded once, cached, and hashed."
    Write-Host "M41 checkpoints after every modality/repeat/fold."

    $scripts = @(
        "scripts\m40_lock_m8_modality_concordance_protocol.py",
        "scripts\m41_run_modality_specific_repeated_nested.py",
        "scripts\m42_compute_gene_level_concordance.py",
        "scripts\m43_compute_reactome_pathway_concordance.py",
        "scripts\m44_generate_m8_publication_assets.py"
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
    Write-Host "METABRIC STAGE M8 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m8"
    Write-Host "Figures: results\figures\metabric_m8"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
