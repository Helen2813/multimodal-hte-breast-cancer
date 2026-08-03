# Paper A Stage 27B — interval-summary repair

Stage 27 completed all 300 locked, fully refitted patient-bootstrap repetitions.

The primary percentile interval is valid and unchanged. The Stage 27 lock,
however, specified a studentized sensitivity interval, while Stage 107 reported
a normal interval instead.

Run:

```powershell
.\run_stage27b_candidate_v10_interval_repair.ps1
```

Stage 27B does not rerun any model or bootstrap estimate. It exactly reproduces
the primary percentile interval and computes the locked studentized sensitivity
interval from the existing bootstrap estimates and their replicate diagnostic
influence-function standard errors.
