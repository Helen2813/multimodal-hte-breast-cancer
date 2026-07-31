$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location $ROOT

$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    throw "Project virtual environment not found: $PY"
}

$LOCK = Join-Path $ROOT "paper_A_treatment_effects\PROTOCOL_LOCKED_CANDIDATE_V9.txt"
if (-not (Test-Path $LOCK)) {
    throw "Candidate V9 protocol is not locked. Run run_stage20_candidate_v9_protocol_lock.ps1 first."
}

$FINAL = Join-Path $ROOT "results\tables\84_publication_bootstrap_decision.csv"
if (Test-Path $FINAL) {
    throw "Stage 21 final decision already exists. This runner refuses to repeat a completed publication bootstrap."
}

$LOGDIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOGDIR ("stage21_publication_bootstrap_" + $STAMP + ".log")

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 128)
    Write-Host "STAGE 21 - LOCKED 300-REPETITION PUBLICATION BOOTSTRAP"
    Write-Host ("#" * 128)
    Write-Host "Project root: $ROOT"
    Write-Host "Python: $PY"
    Write-Host "Transcript: $LOG"
    Write-Host "The Candidate V9 lock is verified before fitting."
    Write-Host "Completed nuisance partitions and repetitions are skipped on resume."
    Write-Host "No locked estimator setting may be changed."

    & $PY "scripts\82_full_publication_bootstrap.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed: scripts\82_full_publication_bootstrap.py"
    }

    & $PY -c "import json,pandas as pd,sys,pathlib; root=pathlib.Path('.'); cfg=json.loads((root/'stage21_config.json').read_text()); rp=root/'results/tables/82_publication_bootstrap_repetitions_checkpoint.csv'; ep=root/'results/tables/82_publication_bootstrap_errors.csv'; r=pd.read_csv(rp) if rp.exists() else pd.DataFrame(); e=pd.read_csv(ep) if ep.exists() else pd.DataFrame(); success=set(pd.to_numeric(r.get('bootstrap_repetition',pd.Series(dtype=float)),errors='coerce').dropna().astype(int)); failed=set(pd.to_numeric(e.get('bootstrap_repetition',pd.Series(dtype=float)),errors='coerce').dropna().astype(int))-success; target=int(cfg['full_bootstrap']['n_repetitions']); minimum=int(cfg['decision_thresholds']['minimum_successful_repetitions']); attempted=len(success|failed); print(f'Successful publication-bootstrap repetitions: {len(success)}/{target}; attempted: {attempted}/{target}; persistent failed repetitions: {len(failed)}'); sys.exit(0 if attempted>=target and len(success)>=minimum else 3)"
    if ($LASTEXITCODE -eq 3) {
        Write-Host "The publication bootstrap has not reached its locked completion threshold. Rerun this same runner to resume from checkpoints."
        Write-Host "Keep this partial log file: $LOG"
        return
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not verify publication-bootstrap completion."
    }

    $scripts = @(
        "scripts\83_summarize_publication_bootstrap.py",
        "scripts\84_generate_publication_decision.py"
    )
    foreach ($script in $scripts) {
        Write-Host ("#" * 128)
        Write-Host "RUNNING $script"
        Write-Host ("#" * 128)
        & $PY $script
        if ($LASTEXITCODE -ne 0) {
            throw "Script failed: $script"
        }
    }

    Write-Host ("#" * 128)
    Write-Host "STAGE 21 COMPLETED"
    Write-Host ("#" * 128)
    Write-Host "FINAL DECISION REPORT"
    Write-Host ("-" * 128)
    Get-Content "results\tables\84_publication_bootstrap_decision.md"
    Write-Host ("-" * 128)
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
