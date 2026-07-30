# Stage 12 — final Paper A inference

Stage 12 uses one resumable checkpoint for a short pilot and the full run.

## Pilot first

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage12_pilot.ps1
```

The pilot must reproduce the candidate landmark estimate exactly, fit one
CCW point estimate, and complete five landmark plus three CCW bootstrap
repetitions.

## Full run after pilot review

```powershell
.\run_stage12_full.ps1
```

It resumes the same checkpoints and continues to 300 landmark and 200 CCW
bootstrap repetitions.

Every landmark repetition resamples patients, creates grouped folds so duplicate
bootstrap copies cannot leak, and refits propensity, censoring, IPCW RMST, and
arm-specific outcome models.

The diagnosis-time CCW sensitivity clones each patient to both grace-period
strategies, applies artificial censoring, cross-fits artificial- and natural-
censoring hazard models, creates stabilized time-varying clone weights, and
estimates weighted RMST through day 910 from diagnosis.
