# Stage 16 — exact outcome-augmentation and nuisance-model audit

## Why this stage is different

The uploaded review bundle confirmed the exact Stage 12 implementation:

1. censoring is estimated with a cross-fitted pooled logistic discrete-time model;
2. the observed outcome is converted to a 730-day IPCW-RMST pseudo-outcome;
3. separate cross-fitted `RidgeCV` outcome models estimate `mu0` and `mu1`;
4. the final effect uses the ATO AIPW score.

Therefore the difference between:

```text
landmark overlap-weighted KM = +6.91 days
landmark overlap AIPW       = +28.77 days
```

cannot safely be labelled “outcome augmentation” without reconstructing the exact AIPW
pseudo-outcome and score. Stage 16 does that first.

## Install

Copy the package contents into:

```text
C:\Users\olegk\Desktop\multimodal-hte-breast-cancer
```

Added files:

```text
stage16_config.json
scripts\
  _stage16_utils.py
  61_stage16_preflight.py
  62_decompose_exact_landmark_aipw.py
  63_outcome_model_robustness.py
  64_fold_and_influence_stability.py
  65_generate_stage16_decision.py
run_stage16_outcome_augmentation_audit.ps1
README_STAGE16.md
```

## Run

In the current active PowerShell and `.venv` session:

```powershell
.\run_stage16_outcome_augmentation_audit.ps1
```

`Set-ExecutionPolicy -Scope Process Bypass` does not need to be repeated in the same
PowerShell window.

## Fixed outcome-model registry

All models use identical:

- 559 landmark patients;
- repeat-1 folds;
- frozen Stage 30 propensity scores;
- discrete-time censoring model;
- IPCW-RMST pseudo-outcome;
- ATO AIPW formula.

The prespecified models are:

```text
arm_mean
arm_ridge_unbounded
arm_ridge_bounded
pooled_interaction_ridge_bounded
arm_hist_gradient_boosting_bounded
```

They are sensitivity analyses, not candidates from which the most favorable effect may be selected.

## Main outputs

```text
results\tables\62_exact_landmark_aipw_decomposition.csv
results\tables\62_exact_ridge_calibration.csv
results\tables\63_outcome_model_robustness.csv
results\tables\63_outcome_model_calibration.csv
results\tables\64_fold_specific_effects.csv
results\tables\64_leave_one_fold_out_effects.csv
results\tables\64_fold_stability_summary.csv
results\tables\65_stage16_decision.md
results\logs\stage16_outcome_augmentation_audit_*.log
```

Local-only, de-identified patient-level diagnostics are written under:

```text
data\derived\stage16\
```

Do not commit those files.

## Possible decisions

```text
DEBUG_EXACT_LANDMARK_RECONSTRUCTION
OUTCOME_AUGMENTATION_SIGN_UNSTABLE_HOLD_PUBLICATION_BOOTSTRAP
POSITIVE_DIRECTION_BUT_OUTCOME_MODEL_DEPENDENT_HOLD_PUBLICATION_BOOTSTRAP
OUTCOME_AUGMENTATION_ROBUST_PROCEED_TO_SHARED_NUISANCE_BOOTSTRAP_PILOT
```

The 300/200 publication bootstrap is not started by this stage.
