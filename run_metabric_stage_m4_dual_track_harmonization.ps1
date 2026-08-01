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
$LOG = Join-Path $LOG_DIR "metabric_stage_m4_dual_track_harmonization_$STAMP.log"

Start-Transcript -Path $LOG -Force

try {
    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M4 - DUAL-TRACK HARMONIZATION AND PAPER-1 REPLICATION PREPARATION"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PYTHON"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages M1-M3B are NOT rerun."
    Write-Host "No treatment-effect, survival, HTE, or modality-utility model is fitted."
    Write-Host "Track A mapping is outcome-blind."
    Write-Host "Track B Paper-1 replication is prepared but not yet executed."
    Write-Host "Ensembl REST mapping requires internet once; every response is cached and hashed."

    $scripts = @(
        "scripts\m20_dual_track_preflight.py",
        "scripts\m21_map_selected_ensembl_to_hgnc.py",
        "scripts\m22_build_outcome_blind_fixed_panel_matrices.py",
        "scripts\m23_audit_mutation_panel_coverage.py",
        "scripts\m24_recover_paper1_feature_selection_recipe.py",
        "scripts\m25_generate_dual_track_protocol_decision.py"
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
    Write-Host "METABRIC STAGE M4 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Results: results\tables\metabric_m4"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
