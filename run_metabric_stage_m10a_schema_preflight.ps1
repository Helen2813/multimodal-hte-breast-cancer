param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = $PSScriptRoot
Set-Location $ROOT

$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$SCRIPT = Join-Path $ROOT "scripts\m50_metabric_m10a_schema_preflight.py"
$OUT_DIR = Join-Path $ROOT "results\tables\metabric_m10a"
$LOG_DIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path -LiteralPath $PY -PathType Leaf)) {
    throw "Virtual environment Python not found: $PY"
}
if (-not (Test-Path -LiteralPath $SCRIPT -PathType Leaf)) {
    throw "M10A script not found: $SCRIPT"
}

if (Test-Path -LiteralPath $OUT_DIR -PathType Container) {
    $existing = @(Get-ChildItem -LiteralPath $OUT_DIR -File -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0 -and -not $Force) {
        Write-Host "Existing M10A outputs:" -ForegroundColor Yellow
        $existing | ForEach-Object { Write-Host "  $($_.FullName)" }
        throw "M10A outputs already exist. Use -Force only for an intentional repeat of this read-only preflight."
    }
    if ($Force) {
        Remove-Item -LiteralPath $OUT_DIR -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m10a_schema_preflight_$STAMP.log"
$transcriptStarted = $false

try {
    Start-Transcript -Path $LOG -Force
    $transcriptStarted = $true

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M10A - READ-ONLY NPI AND CALIBRATION SCHEMA PREFLIGHT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "M1-M9 are NOT rerun."
    Write-Host "No model is fitted."
    Write-Host "No feature selection is repeated."
    Write-Host "No patient identifier values are printed."
    Write-Host ("#" * 124)

    & $PY $SCRIPT
    if ($LASTEXITCODE -ne 0) {
        throw "M10A schema preflight returned exit code $LASTEXITCODE. Keep the log; it contains the missing schema checks."
    }

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M10A COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Upload this single log file:"
    Write-Host "  $LOG"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript
    }
}
