# Stage 19 - Inner cross-fit stabilization

Stage 19 does not rerun Stages 15-18 and does not draw new patient bootstrap samples.
It reuses the exact 30 Stage 18 bootstrap seeds and adds nuisance partitions 6-20.

The Stage 18 automatic label is retained in the audit trail. Stage 19 corrects its
interpretation: all 30 bootstrap samples completed without failures or numerical
explosions. The issue was Monte Carlo variability from averaging only five nuisance
partitions inside each bootstrap sample.

## Run

From the project root with the virtual environment available:

```powershell
.\run_stage19_inner_crossfit_stabilization.ps1
```

In a new PowerShell window, first run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
.\run_stage19_inner_crossfit_stabilization.ps1
```

## New analyses

- Stage 75: protocol correction and exact reaggregation preflight.
- Stage 76: add partitions 6-20 to the same 30 bootstrap samples.
- Stage 77: compare 5, 10, 15, and 20 partition aggregates and estimate inner Monte Carlo error.
- Stage 78: decide whether a locked 20-partition publication bootstrap is computationally justified.

## Main log

`results/logs/stage19_inner_crossfit_stabilization_YYYYMMDD_HHMMSS.log`

The publication bootstrap remains locked throughout Stage 19.
