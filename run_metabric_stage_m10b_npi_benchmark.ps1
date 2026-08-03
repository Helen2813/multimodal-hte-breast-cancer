param(
    [int]$Bootstrap = 2000,
    [int]$CheckpointEvery = 100
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = $PSScriptRoot
Set-Location $ROOT

$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$SCRIPT = Join-Path $ROOT "scripts\m51_metabric_m10b_npi_benchmark.py"
$LOG_DIR = Join-Path $ROOT "results\logs"

if (-not (Test-Path -LiteralPath $PY -PathType Leaf)) {
    throw "Virtual environment Python not found: $PY"
}
if (-not (Test-Path -LiteralPath $SCRIPT -PathType Leaf)) {
    throw "M10B script not found: $SCRIPT"
}

New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "metabric_stage_m10b_npi_benchmark_$STAMP.log"
$transcriptStarted = $false

try {
    Start-Transcript -Path $LOG -Force
    $transcriptStarted = $true

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M10B - LOCKED OOF NPI BENCHMARK AND PUBLICATION FIGURES"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "Bootstrap repetitions: $Bootstrap"
    Write-Host "M1-M9 are NOT rerun."
    Write-Host "No feature selection is repeated."
    Write-Host "No model is refitted."
    Write-Host "Partial bootstrap checkpoints are resumed automatically."
    Write-Host ("#" * 124)

    & $PY $SCRIPT --bootstrap $Bootstrap --checkpoint-every $CheckpointEvery
    if ($LASTEXITCODE -ne 0) {
        throw "M10B returned exit code $LASTEXITCODE. Keep the transcript and any checkpoint files."
    }

    Write-Host ("#" * 124)
    Write-Host "METABRIC STAGE M10B COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Upload this single log file:"
    Write-Host "  $LOG"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript
    }
}
