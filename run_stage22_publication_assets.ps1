$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw "Virtual environment Python not found: $PY"
}

$env:PYTHONUNBUFFERED = "1"
$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage22_publication_assets_$STAMP.log"
$transcriptStarted = $false

try {
    Start-Transcript -Path $LOG -Force
    $transcriptStarted = $true

    Write-Host ("#" * 124)
    Write-Host "STAGE 22 - PUBLICATION TABLES, FIGURES, AND MANUSCRIPT SNIPPETS"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "Stages 15 through 21 are NOT rerun."
    Write-Host "No treatment-effect estimator or bootstrap is refitted."
    Write-Host "Locked Candidate V9 files are read-only inputs."
    Write-Host "The working manuscript is NOT overwritten."

    $required = @(
        "stage22_config.json",
        "scripts\_stage22_utils.py",
        "scripts\85_stage22_preflight.py",
        "scripts\86_generate_publication_tables.py",
        "scripts\87_generate_publication_figures.py",
        "scripts\88_generate_manuscript_snippets.py",
        "scripts\89_generate_stage22_report.py",
        "results\tables\79_candidate_v9_final_point_estimate.csv",
        "results\tables\82_publication_bootstrap_repetitions_checkpoint.csv",
        "results\tables\82_publication_bootstrap_partitions_checkpoint.csv",
        "results\tables\82_publication_bootstrap_errors.csv",
        "results\tables\83_publication_bootstrap_summary.csv",
        "results\tables\84_publication_bootstrap_decision.csv",
        "paper_A_treatment_effects\analysis_plan_FINAL.md",
        "data\derived\manifests\80_candidate_v9_protocol_lock_manifest.json"
    )

    foreach ($rel in $required) {
        $path = Join-Path $ROOT $rel
        if (-not (Test-Path $path)) {
            throw "Required Stage 22 input is missing: $rel"
        }
    }

    $scripts = @(
        "scripts\85_stage22_preflight.py",
        "scripts\86_generate_publication_tables.py",
        "scripts\87_generate_publication_figures.py",
        "scripts\88_generate_manuscript_snippets.py",
        "scripts\89_generate_stage22_report.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 124)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 124)
        & $PY (Join-Path $ROOT $script)
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 124)
    Write-Host "STAGE 22 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Publication assets:"
    Write-Host (Join-Path $ROOT "paper_A_treatment_effects\publication_assets_candidate_v9")
    Write-Host "Keep this single log file: $LOG"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript
    }
}
