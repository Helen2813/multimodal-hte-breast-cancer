$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    throw ".venv not found. Run setup_venv_and_run_00_to_06.ps1 first."
}

& $PY (Join-Path $ROOT "scripts\05_build_compact_adjustment.py")
if ($LASTEXITCODE -ne 0) { throw "05_build_compact_adjustment.py failed." }

& $PY (Join-Path $ROOT "scripts\06_run_compact_overlap.py")
if ($LASTEXITCODE -ne 0) { throw "06_run_compact_overlap.py failed." }

Write-Host "`nStage 2 completed."
