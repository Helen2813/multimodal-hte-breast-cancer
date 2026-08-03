# Paper A Stage 34 — independent confirmatory simulation

Run Stage 33C first. Then extract this package into the project root and run:

```powershell
.\run_stage34_confirmatory_sequence_simulation.ps1
```

Stage 34 is independent of the pilot:

- 12 locked scenarios;
- 500 new repetitions per scenario;
- 3 methods;
- 18,000 expected method-runs;
- new seed range;
- pilot repetitions are not reused;
- robust checkpoint parsing avoids the prior `null` token problem;
- effect labels are `true_zero` and `observed_risk_benefit`.

The run may take several hours. Re-running the same PowerShell command resumes
from the checkpoint.

Validity gates apply to `adjusted_full` and `sequencing_aware`. `naive_full`
is an intentionally misspecified diagnostic comparator and is evaluated using
locked mechanism checks rather than a coverage-validity requirement.
