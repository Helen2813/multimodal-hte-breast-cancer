# Paper A Stage 27 — Candidate V10 publication bootstrap

Extract into the project root and run:

```powershell
.\run_stage27_candidate_v10_publication_bootstrap.ps1
```

Stage 27:

1. Verifies the Candidate V10 lock and Stage 26 point estimate.
2. Locks all Stage 27 code before bootstrap execution.
3. Runs an identity resample and requires exact reproduction of the Stage 26
   point estimate.
4. Runs 300 ordinary patient bootstrap samples.
5. Refits the full-sample unpenalized propensity in every bootstrap sample.
6. Refits all cross-fitted censoring and bounded outcome nuisance models over
   the 20 locked partitions in every bootstrap sample.
7. Keeps all duplicate copies of the same source patient in one fold.
8. Checkpoints after every repetition and resumes automatically.
9. Reports the percentile interval as primary, with basic and normal intervals
   as sensitivity summaries.

No bootstrap repetition is removed because of its effect direction or because
a descriptive propensity-tail diagnostic is unfavourable. Numerical failures
are retained in an error table and prevent final interval generation until all
300 locked repetitions complete.

No manuscript prose is generated.

The repetition count, bootstrap base seed, sampling rule, and interval specification are read from the already locked Candidate V10 estimator specification; they are not newly chosen by Stage 27.
