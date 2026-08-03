# Paper A Stage 30 — top-influence non-event leave-one-out

Extract into the project root and run:

```powershell
.\run_stage30_candidate_v10_non_event_influence.ps1
```

Stage 30 complements the completed all-event leave-one-out analysis.

It:

1. Verifies the frozen Candidate V10 and Stage 26 locks.
2. Loads the locked Stage 26 patient-level influence scores.
3. Selects the 10 non-event patients with the largest absolute influence.
4. Locks that selected set before any deletion estimate is computed.
5. Reproduces the Stage 26 primary point estimate.
6. Omits each selected non-event patient once.
7. Refits the full-sample propensity and all cross-fitted censoring/outcome
   nuisance models over the same 20 partitions.
8. Combines the summary with the 36 completed event-patient deletions.
9. Stores patient identifiers only in LOCAL_ONLY files.
10. Checkpoints after each deletion and resumes automatically.

This is a post hoc targeted diagnostic. It does not alter the primary analysis,
create a new confidence interval, or generate manuscript prose.
