# Stage 9 v4 — robust finish from Stage 32

This patch fixes the third late-stage error.

## Root cause

Stage 31 created the G=0.10 pseudo-outcome file with the suffix:

```text
g01
```

Stage 32 expected:

```text
g010
```

The numerical results were already saved correctly; only the filename convention
differed.

## Changes

- Stage 32 discovers the pseudo-outcome file that actually exists.
- It supports both old and new G-min naming conventions.
- A standalone preflight validates every input and patient-ID set before simulation.
- Simulation results are checkpointed after every scenario.
- The resume runner starts at Stage 32; Stages 29–31 are not rerun.

## Run

Copy the patch contents into the project root, replacing Stage 32, then run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage9_finish_from_32.ps1
```

The transcript is saved to:

```text
results/logs/stage9_finish_from32_YYYYMMDD_HHMMSS.log
```
