# Paper A Stage 33C — pilot decision amendment

Run after the completed Stage 33B repair:

```powershell
.\run_stage33c_pilot_decision_amendment.ps1
```

This stage reruns no simulation.

It documents why loss of coverage for `naive_full` is not a pilot-validity
failure: that estimator intentionally omits chemotherapy sequencing and is
included to demonstrate misspecification. Bias and coverage gates apply to
`adjusted_full` and `sequencing_aware`. The naive comparator is checked for
numerical success, included-variable balance, absence of bias without
sequencing, and the expected sequencing-induced distortion.

The amendment is locked before the independent confirmatory simulation.
