# Stage 15 v3 — exact runtime row-map patch

The previous fix still guessed that one of the saved Stage 14 CSVs retained a usable crosswalk.
That assumption was wrong.

This patch does not guess IDs from saved feature tables. It re-runs the already verified Stage 41
point estimator once under a narrow trace and captures the exact DataFrame passed into
`_clone_long_rows`, while its original index and patient identifier are still present.

The mapping is accepted only when `row_id` is verified as one of:

1. a named key in the runtime input;
2. the runtime input index;
3. the exact complete integer range `0..n-1`, permitting a validated positional mapping.

The resulting clone table must contain 594 unique CCW patients and cover all 559 landmark
patients before Stage 57 is allowed to run.

## Install

Copy the package contents into the project root and replace:

```text
scripts\57_common_target_estimator_bridge.py
```

Add:

```text
scripts\60_capture_exact_ccw_row_map.py
run_stage15_resume_exact_rowmap.ps1
README_STAGE15_V3_EXACT_ROWMAP.md
```

No replacement of `_stage15_utils.py` is required.

## Run

In the same PowerShell terminal:

```powershell
.\run_stage15_resume_exact_rowmap.ps1
```

Do not rerun Stage 56.
