# Stage 9 v5 — exact model-table preflight

## Root cause fixed

Both the landmark cohort and compact adjustment table contain `diagnosis_year`.
A normal pandas merge therefore created:

```text
diagnosis_year_x
diagnosis_year_y
```

while Stage 32 requested `diagnosis_year`.

## Architectural correction

Stage 32 now uses one canonical builder for both preflight and analysis:

1. compact clinical/era features are taken only from the compact table;
2. names that overlap are removed from the cohort side before merging;
3. RNA features remain sourced from the landmark cohort;
4. duplicate columns and `_x`/`_y` suffixes are forbidden;
5. the exact clinical and clinical+RNA matrices are selected during preflight;
6. the same already-tested builder is used in the real run.

## Run

Copy the patch into the project root and allow replacement of the Stage 32
and preflight scripts.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage9_v5_finish_from_32.ps1
```

Stages 29–31 are not rerun.
