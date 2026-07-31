# Stage 15 v2 — patient-ID bridge patch

## Cause of the error

The Stage 14 CCW clone table stores a source-row linkage field (`row_id`). The selected
559-patient landmark table does not necessarily retain that field. The original Stage 15 code
searched all columns with roughly 500–700 unique values and therefore allowed an omics feature
such as `RNA_ENSG...` to be treated as an ID candidate.

This patch:

1. strongly prioritizes actual identifier names;
2. excludes omics, outcome, propensity, and weight columns from normal ID detection;
3. first attempts a direct patient-ID match;
4. when the clone table only has `row_id`, uses the captured 594-row Stage 14 source table as a
   deterministic bridge:

```text
CCW clone row_id
  -> source-cohort row_id
  -> source patient ID
  -> landmark patient ID
```

No positional row matching is used.

## Install

Copy the package contents into the project root and allow replacement of:

```text
scripts\_stage15_utils.py
scripts\57_common_target_estimator_bridge.py
```

A new runner is added:

```text
run_stage15_resume_from_57.ps1
```

## Run

In the same active PowerShell/virtual-environment session:

```powershell
.\run_stage15_resume_from_57.ps1
```

Stage 56 does not need to be repeated. The runner continues with Stages 57–59.

## PowerShell execution policy

`Set-ExecutionPolicy -Scope Process Bypass` applies only to the current PowerShell process.

- In the same terminal window/session: do not repeat it.
- After opening a new PowerShell terminal: run it again.
- Activating or deactivating `.venv` does not reset the process-scoped policy.
