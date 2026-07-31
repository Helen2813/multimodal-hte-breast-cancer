# Stage 15 — common-target estimator bridge and re-estimated CCW truncation

## Why this stage is required

Stage 14 reproduced the diagnosis-time clone-censor-weight curves essentially exactly. The total
CCW RMST difference was `-8.801` days, while the pre-landmark component was approximately zero and
the conditional post-day-180 difference was `-6.730` days. Therefore, the disagreement with the
`+28.773`-day landmark ATO AIPW result is not explained by the grace period alone.

Stage 14 also found that maximum clone weights exceeded 10 in 56.7% of bootstrap repetitions,
although the fixed full-data caps remained negative. Stage 15 separates two questions:

1. Is the sign disagreement mainly due to target-population weighting, outcome augmentation, or
   clone/adherence weighting?
2. Does the CCW result remain stable when nuisance models are re-estimated in every bootstrap
   sample and final clone weights are truncated?

## Install

Copy the package contents into:

```text
C:\Users\olegk\Desktop\multimodal-hte-breast-cancer
```

It adds:

```text
stage15_config.json
scripts\
  _stage15_utils.py
  56_stage15_preflight.py
  57_common_target_estimator_bridge.py
  58_reestimated_ccw_truncation_bootstrap.py
  59_generate_stage15_decision.py
run_stage15_target_bridge_and_truncation.ps1
README_STAGE15.md
```

## Run

Inside the active `.venv`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage15_target_bridge_and_truncation.ps1
```

Stages 1–14 are not repeated. The 300/200 publication bootstrap is not started.

## Stage 57 — common-target bridge

On the same post-day-180 horizon the code compares:

```text
unweighted landmark KM
frozen-overlap-weighted landmark KM
landmark overlap AIPW
original conditional CCW
conditional CCW multiplied by the frozen landmark overlap weight
```

Possible classifications:

```text
ATO_TARGET_WEIGHTING_RECONCILES_DIRECTION
OUTCOME_AUGMENTATION_IS_PRIMARY_SIGN_BRIDGE
CCW_ADHERENCE_OR_CENSORING_MODEL_REMAINS_PRIMARY_DIFFERENCE
```

This bridge is diagnostic; it is not a new primary estimator.

## Stage 58 — re-estimated truncation bootstrap

The original Stage 43 script is reused without editing it. During each bootstrap repetition:

1. the original bootstrap sample is constructed;
2. the original nuisance/adherence procedure is re-estimated;
3. the clone-level table is captured when `ccw_estimate` returns;
4. final clone weights are truncated;
5. weighted survival and RMST are recalculated;
6. the capped result is returned to the unchanged Stage 43 checkpoint loop.

Three paired 30-repetition runs are created:

```text
cap at 5
cap at 10
cap at each bootstrap sample's empirical 99th percentile
```

The original Stage 43 files are backed up and restored after every run.

## Main outputs

```text
results\tables\57_common_target_estimator_bridge.csv
results\tables\57_bridge_component_differences.csv
results\tables\57_bridge_diagnostics.csv
results\tables\58_reestimated_truncation_bootstrap_summary.csv
results\tables\58_reestimated_truncation_paired_repetitions.csv
results\tables\58_reestimated_truncation_paired_summary.csv
results\tables\59_stage15_decision.md
paper_A_treatment_effects\analysis_plan_CANDIDATE_V5.md
paper_A_treatment_effects\primary_estimand_CANDIDATE_V5.json
```

No post-hoc selection of the positive landmark estimate is allowed.
