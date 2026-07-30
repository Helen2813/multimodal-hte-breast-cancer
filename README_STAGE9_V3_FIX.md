# Stage 9 v3 — skip non-ready landmark designs

Stage 30 completed and classified both TNBC landmark designs as:

```text
LANDMARK_NOT_READY
```

The previous Stage 31 nevertheless attempted arm-specific outcome modeling for
TNBC landmark 365, where one training fold contained fewer than 20 controls.

This patch changes Stage 31 so that every design marked `LANDMARK_NOT_READY`
is skipped before censoring/outcome/AIPW modeling. The reason is printed in
the console and saved to:

```text
results/tables/31_skipped_not_ready_designs.csv
```

## Copy

Copy all contents of this patch into the project root and allow replacement of:

```text
scripts/31_landmark_ai_aipw.py
```

## Resume

Stage 29 and Stage 30 do not need to be rerun.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage9_resume_from_31.ps1
```

The complete transcript will be written to:

```text
results/logs/stage9_resume_from31_YYYYMMDD_HHMMSS.log
```
