# Stage 13 v2 resume patch

## What failed

Stage 45, Stage 46, and the initial Stage 47 audit completed correctly.

Stage 48 failed before extending the landmark checkpoint because the existing
Stage 42 bootstrap script used:

```python
pd.read_csv(errors_path) if errors_path.exists() else pd.DataFrame()
```

The error file existed but contained no header or columns. Pandas therefore raised:

```text
pandas.errors.EmptyDataError: No columns to parse from file
```

This is a resume-state bookkeeping bug. It does not invalidate the 5 landmark
bootstrap repetitions, 3 CCW repetitions, point estimates, checkpoints, or Stage 13
estimand audit.

## Patch contents

Copy these files into the project root and allow replacement of
`scripts\48_extend_centering_pilot.py`:

```text
scripts\
  48_extend_centering_pilot.py
run_stage13_resume_from_48.ps1
README_STAGE13_V2_FIX.md
```

The replacement Stage 48:

1. inspects the Stage 42 and Stage 43 source files for error/failure CSV names;
2. also searches `results\tables` for Stage 42/43 error-like CSVs;
3. backs up only zero-byte, whitespace-only, or `EmptyDataError` CSVs;
4. removes those empty placeholders before resume;
5. preserves every nonempty error log;
6. retries once only when a failed attempt created another structurally empty error CSV;
7. resumes from the existing checkpoint instead of restarting completed bootstrap repetitions.

Backups are stored under:

```text
results\logs\stage13_empty_csv_backups\
```

## Run

Do not rerun Stage 45-47.

From the active `.venv`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage13_resume_from_48.ps1
```

The runner continues with:

```text
Stage 48 v2
Stage 47 centering audit again
Stage 49 CCW invariants
Stage 50 Stage 13 decision
```

It still does not start the 300/200 publication bootstrap.

## Expected first diagnostic

`results\tables\48_empty_error_csv_cleanup.csv` should show one row similar to:

```text
action = BACKED_UP_AND_REMOVED
```

for the empty Stage 42 error CSV. The completed 5/3 checkpoint rows remain intact,
and the scripts should extend them toward 30/30.
