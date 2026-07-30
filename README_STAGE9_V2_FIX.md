# Stage 9 v2 correction

The first Stage 9 package stopped at Stage 30 because of a naming mismatch:

```text
effective_sample_size
```

was imported while the shared utility exposed the same function as:

```text
ess
```

This version adds a backward-compatible alias and a resume runner.

## Current state

Stage 29 completed successfully. Its landmark cohort files should be preserved.

## Copy

Copy all contents of this package into the project root and allow replacement of
the existing Stage 9 scripts.

## Resume without rerunning Stage 29

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage9_resume_from_30.ps1
```

The new transcript will be saved to:

```text
results/logs/stage9_resume_from30_YYYYMMDD_HHMMSS.log
```

The resume runner checks that:

```text
results/tables/29_landmark_cohort_summary.csv
```

already exists before it starts.
