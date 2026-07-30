# Stage 13 - estimand harmonization and bootstrap centering

## Why this stage is required

Stage 12 v2 reproduced the landmark estimate exactly and completed the pilot bootstraps.
However, the primary 180-day landmark estimate was positive while the diagnosis-time
clone-censor-weight (CCW) point estimate was negative. The two analyses do not currently
estimate the same parameter:

- the landmark analysis conditions on survival and observation to day 180 and targets an
  overlap population from day 180 onward;
- the CCW analysis begins at diagnosis and targets dynamic treatment strategies through
  day 910.

Stage 13 prevents a methodologically invalid response such as choosing the positive estimate,
pooling the two estimates, or immediately launching the 300/200 bootstrap.

## Files to copy

Copy the contents of this package into the project root:

```text
C:\Users\olegk\Desktop\multimodal-hte-breast-cancer
```

It adds:

```text
stage13_config.json
scripts\
  _stage13_utils.py
  45_stage13_preflight.py
  46_compare_estimands_and_targets.py
  47_audit_stage12_centering.py
  48_extend_centering_pilot.py
  49_validate_ccw_invariants.py
  50_generate_stage13_decision.py
run_stage13_estimand_harmonization.ps1
README_STAGE13.md
```

No file under `data\processed` is modified.

## Run

From the active project virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage13_estimand_harmonization.ps1
```

The runner:

1. verifies Stage 11/12 inputs and exact point estimates;
2. maps the two estimands and checks whether they are numerically comparable;
3. audits the 5/3 pilot bootstrap centering;
4. discovers the existing Stage 12 repetition environment variables;
5. extends the checkpointed pilot to 30 landmark and 30 CCW repetitions;
6. reruns the centering audit;
7. checks CCW clone-flow and weight invariants;
8. writes the Stage 13 decision and Candidate V3 amendment.

It does **not** launch the planned 300 landmark / 200 CCW bootstrap.

## Main outputs

```text
results\tables\45_stage13_preflight.md
results\tables\46_estimand_harmonization.md
results\tables\47_bootstrap_centering_audit.csv
results\tables\48_centering_pilot_extension.csv
results\tables\49_ccw_invariant_checks.csv
results\tables\50_stage13_decision.md
paper_A_treatment_effects\analysis_plan_CANDIDATE_V3.md
paper_A_treatment_effects\primary_estimand_CANDIDATE_V3.json
results\logs\stage13_estimand_harmonization_*.log
```

## Decision rules

Possible gates:

- `HOLD_FULL_BOOTSTRAP_AND_DEBUG_CCW`
- `HOLD_FULL_BOOTSTRAP_PENDING_CENTERING_PILOT`
- `HOLD_FULL_BOOTSTRAP_DUE_TO_CENTERING_CONCERN`
- `CENTERING_PASSED_EXPORT_CCW_CURVES_BEFORE_FULL_BOOTSTRAP`
- `PROCEED_FULL_BOOTSTRAP_AS_DESIGN_SENSITIVITY_STUDY`

Even the last gate does not convert Paper A into an efficacy paper. Landmark and CCW remain
separate, design-sensitive estimands.
