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
$LOG = Join-Path $LOG_DIR "stage34_confirmatory_sequence_simulation_$STAMP.log"

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 34 - INDEPENDENT CONFIRMATORY SEQUENCING SIMULATION"
    Write-Host ("#" * 128)
    Write-Host "Twelve scenarios, 500 independent repetitions each, three methods."
    Write-Host "Pilot repetitions are not reused."
    Write-Host "Checkpointing is enabled after every method-run."
    Write-Host "This run may take several hours."
    Write-Host "No Candidate V9/V10 file and no manuscript prose is modified."

    $scripts = @(
        "scripts\130_stage34_lock_confirmatory_simulation.py",
        "scripts\131_stage34_run_confirmatory_simulation.py",
        "scripts\132_stage34_summarize_confirmatory_simulation.py"
    )

    foreach ($script in $scripts) {
        Write-Host ("#" * 128)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 128)
        & $PYTHON $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 128)
    Write-Host "PAPER A STAGE 34 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "Tables: results\tables\stage34_confirmatory_sequence_simulation"
    Write-Host "Figures: results\figures\stage34_confirmatory_sequence_simulation"
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
