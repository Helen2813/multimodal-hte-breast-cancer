# Paper A Stage 25 — Candidate V10 protocol amendment

Extract this package into the project root and run:

```powershell
.\run_stage25_candidate_v10_protocol_lock.ps1
```

Stage 25:

- verifies the complete Candidate V9 lock;
- reproduces the Stage 24 broad no-chemotherapy-by-day-180 counts;
- constructs a stricter Candidate V10 population;
- excludes patients whose chemotherapy start timing is unascertainable;
- does not exclude patients merely because chemotherapy starts after day 180;
- preserves the 13-variable Candidate V9 compact adjustment set;
- runs repeated cross-fitted propensity diagnostics;
- checks ATO balance and effective sample size;
- creates a cryptographic Candidate V10 lock before any V10 effect is computed.

It does **not**:

- modify Candidate V9;
- compute a Candidate V10 treatment-effect estimate;
- run a new bootstrap;
- add an ever-chemotherapy variable to the adjustment set;
- condition on chemotherapy received after day 180;
- generate manuscript text.

Upload the single Stage 25 log after it finishes. The next package will reuse the
exact Candidate V9 estimator for the locked Candidate V10 point estimate. A new
publication bootstrap will be authorized only after the point-estimate diagnostics
are reviewed.

The package also contains a Stage 26 point-estimate runner. Its code is hashed into
the Stage 25 lock before any V10 effect is seen. Do not run Stage 26 until the Stage
25 log has been reviewed.

Later command, only after review:

```powershell
.\run_stage26_candidate_v10_point_estimate.ps1
```
