# Stage 14 — CCW curve decomposition and bootstrap-weight stability

## Why this stage follows Stage 13

Stage 13 completed 30 checkpointed repetitions for both analyses:

- landmark point estimate: 28.77 days;
- landmark bootstrap mean: 46.12 days, percentile interval -28.24 to 119.71;
- CCW point estimate: -8.80 days;
- CCW bootstrap mean: -7.11 days, percentile interval -28.05 to 7.09.

The centering checks passed their prespecified numerical gate, and all CCW clone-flow invariants
passed. However:

1. the landmark and CCW analyses remain different estimands;
2. the CCW survival curves were not saved;
3. the maximum CCW clone weight reached 25 in the bootstrap and exceeded 10 in 56.7% of
   repetitions;
4. the landmark centering statistic was 2.39, close to the 2.5 threshold.

Therefore the 300/200 publication bootstrap remains locked.

## Install

Copy the package contents into:

```text
C:\Users\olegk\Desktop\multimodal-hte-breast-cancer
```

Added files:

```text
stage14_config.json
scripts\
  _stage14_utils.py
  51_stage14_preflight.py
  52_audit_bootstrap_weight_instability.py
  53_capture_ccw_analysis_state.py
  54_export_ccw_curves_and_decompose.py
  55_generate_stage14_decision.py
run_stage14_ccw_curve_decomposition.ps1
README_STAGE14.md
```

## Run

Inside the active `.venv`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage14_ccw_curve_decomposition.ps1
```

## What Stage 53 does

The original Stage 41 code is not edited. It is executed once under a narrow Python trace.
Only return frames from:

```text
scripts\41_replicate_estimators.py
scripts\_stage12_utils.py
```

and existing helper modules are inspected. Clone-like DataFrames are copied into:

```text
data\derived\stage14_trace\
```

The hashes of the existing Stage 41 output tables are checked before and after the traced rerun.

## Main outputs

```text
results\tables\52_ccw_bootstrap_weight_audit.csv
results\figures\52_ccw_estimate_vs_weight_max.png
results\tables\53_ccw_trace_candidate_manifest.csv
results\tables\53_stage41_output_hash_check.csv
results\tables\54_ccw_weighted_survival_curves.csv
results\tables\54_ccw_curve_replication_checks.csv
results\tables\54_ccw_rmst_decomposition.csv
results\tables\54_ccw_fixed_weight_cap_sensitivity.csv
results\figures\54_ccw_weighted_survival_curves.png
results\tables\55_stage14_decision.md
paper_A_treatment_effects\analysis_plan_CANDIDATE_V4.md
paper_A_treatment_effects\primary_estimand_CANDIDATE_V4.json
```

## Important interpretation

The conditional day-180-to-day-910 CCW contrast is a decomposition of the diagnosis-time
strategy curves. It is not the landmark ATO effect because eligibility conditioning and target
weighting remain different.

The fixed-weight cap analysis does not replace a full-pipeline bootstrap in which the adherence
and censoring models are re-estimated within every repetition.
