# Stage 10 — Paper A candidate design and repaired Paper B power diagnostic

Run from the active project environment:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage10_lockA_repairB.ps1
```

The complete console transcript is written to:

```text
results/logs/stage10_YYYYMMDD_HHMMSS.log
```

## Paper A

Stage 34 converts the successful 180-day landmark / 730-day post-landmark
analysis into a candidate protocol:

- early initiation by day 180 versus no initiation by day 180;
- ATO RMST difference;
- classical regularized nuisance models as primary;
- boosted-AI nuisance models as sensitivity;
- 365-day landmark and 1095-day horizon as sensitivities;
- reliability/robustness claim rather than efficacy claim.

The protocol remains `CANDIDATE_NOT_LOCKED`.

## Paper B

Stage 35 repairs the previous power diagnostic:

- nested RNA imputation, scaling, and PCA inside each training fold;
- paired patient-level OOF R-loss difference rather than a t-test over only
  five fold averages;
- true CATE available in simulations, allowing PEHE assessment;
- null and RNA HTE signals of 50, 100, 150, and 200 RMST days SD;
- ridge and boosted-AI learners;
- checkpoint after every simulation.

The observed real-data result is not called a null unless the repaired
simulation demonstrates adequate power and controlled false-positive rate.
