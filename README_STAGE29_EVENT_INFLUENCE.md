# Paper A Stage 29 — leave-one-event-patient-out influence analysis

Extract into the project root and run:

```powershell
.\run_stage29_candidate_v10_event_influence.ps1
```

Stage 29:

1. Verifies the frozen Candidate V10 and Stage 26 locks.
2. Locks the complete set of 36 observed event patients before influence
   estimates are calculated.
3. Exactly reproduces the Stage 26 primary point estimate.
4. Omits each event patient once.
5. Refits the full-sample unpenalized propensity and all cross-fitted
   censoring/outcome nuisance models over the same 20 partitions.
6. Reports the change from the locked +67.914-day primary estimate.
7. Summarizes deletions separately for the 9 early-hormone events and the
   27 control events.
8. Checkpoints after each deletion and resumes automatically.
9. Writes patient identifiers only to LOCAL_ONLY event-set and checkpoint files.

This is a post hoc influence diagnostic. It does not change the primary
estimand, population, estimator, or bootstrap interval, and it does not
generate manuscript prose.
