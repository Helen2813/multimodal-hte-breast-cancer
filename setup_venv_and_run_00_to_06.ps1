$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$VENV = Join-Path $ROOT ".venv"
$PY = Join-Path $VENV "Scripts\python.exe"

Write-Host "Project root: $ROOT"

if (-not (Test-Path $PY)) {
    Write-Host "`nCreating Python 3.10 virtual environment..."
    py -3.10 -m venv $VENV
}

Write-Host "`nInstalling/updating packages inside .venv..."
& $PY -m pip install --upgrade pip
& $PY -m pip install -r (Join-Path $ROOT "requirements-initial.txt")
& $PY -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $ROOT "requirements-lock.txt")

Write-Host "`nPython used:"
& $PY -c "import sys; print(sys.executable)"

$SCRIPTS = @(
    "scripts\00_validate_inputs.py",
    "scripts\01_audit_processed_data.py",
    "scripts\02_build_master_tables.py",
    "scripts\03_create_analysis_cohorts.py",
    "scripts\04_run_overlap_diagnostics.py",
    "scripts\05_build_compact_adjustment.py",
    "scripts\06_run_compact_overlap.py"
)

foreach ($script in $SCRIPTS) {
    $path = Join-Path $ROOT $script
    if (-not (Test-Path $path)) {
        throw "Missing script: $path"
    }
    Write-Host "`nRunning $script ..."
    & $PY $path
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed: $script"
    }
}

Write-Host "`nAll stages 00-06 completed inside .venv."
Write-Host "Review these files:"
Write-Host "  results\tables\05_compact_adjustment_summary.csv"
Write-Host "  results\tables\06_compact_overlap_summary.csv"
Write-Host "  results\tables\06_legacy_vs_compact_overlap.csv"
