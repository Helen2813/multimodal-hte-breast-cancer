# Stages 20-21: Candidate V9 protocol lock and publication bootstrap

## Run now

In the active project `.venv`:

```powershell
.\run_stage20_candidate_v9_protocol_lock.ps1
```

Stage 20 does not rerun Stages 15-19 and does not start the 300-repetition bootstrap. It:

1. calculates the final original-cohort estimator with the 20 nuisance-partition seeds approved by Stage 19;
2. writes the final Paper A analysis plan, estimand registry, model registry, and bootstrap registry;
3. hashes critical data, code, configurations, decisions, and final protocol files;
4. verifies the lock and prints all hashes to the terminal and one transcript.

The lock is write-once. The runner refuses to overwrite an existing Candidate V9 lock.

## Run only after Stage 20 passes and its log has been reviewed

```powershell
.\run_stage21_publication_bootstrap.ps1
```

Stage 21 runs the locked 300-repetition ordinary patient bootstrap. Every repetition refits propensity, censoring, and bounded ridge outcome nuisances across the same 20 locked partition seeds. All copies of one bootstrap patient remain in one nuisance fold.

The runner checkpoints every nuisance partition and repetition. If interrupted, rerun the same command. Completed work is skipped. The runner refuses to repeat after the final Stage 84 decision exists.

## Primary inference

- Point estimand: 730-day post-landmark ATO RMST difference, initiation by day 180 minus no initiation by day 180.
- Primary interval: 95% percentile patient-bootstrap interval.
- Sensitivities: basic and studentized bootstrap intervals.
- The analysis remains observational and conditional on consistency, conditional exchangeability, positivity, conditional independent censoring, and source validity.

## Logging

Both runners print all required diagnostics to the terminal and preserve a single transcript under `results/logs/`.
