# Stage 16 decision report

**Decision:** `POSITIVE_DIRECTION_BUT_OUTCOME_MODEL_DEPENDENT_HOLD_PUBLICATION_BOOTSTRAP`

**Protocol status:** `CANDIDATE_V6_NOT_LOCKED`

## Exact AIPW decomposition

| model | estimate_days | direct_ato_ipw_treated_mean_days | direct_ato_ipw_control_mean_days | direct_ato_ipw_effect_days | plugin_component_days | treated_residual_component_days | control_residual_component_days | total_residual_augmentation_days | aipw_minus_direct_ato_ipw_days | ato_denominator | exact_stage12_estimate_days | replication_difference_days | horizon_days | n | treated | control | events | pseudo_mean | pseudo_sd | pseudo_p99 | pseudo_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_ridge_unbounded_exact_stage12 | 28.7732 | 680.251 | 653.74 | 26.5112 | 15.7296 | 4.45052 | 8.59309 | 13.0436 | 2.26198 | 113.178 | 28.7732 | 0 | 730 | 559 | 194 | 365 | 50 | 664.849 | 323.962 | 1569.54 | 2264.14 |

## Fixed outcome-model registry

| model | estimate_days | direct_ato_ipw_treated_mean_days | direct_ato_ipw_control_mean_days | direct_ato_ipw_effect_days | plugin_component_days | treated_residual_component_days | control_residual_component_days | total_residual_augmentation_days | aipw_minus_direct_ato_ipw_days | ato_denominator | mu0_min | mu0_max | mu1_min | mu1_max | fraction_mu0_outside_0_horizon | fraction_mu1_outside_0_horizon | difference_from_exact_ridge_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_mean | 25.974 | 680.251 | 653.74 | 26.5112 | 6.04183 | 11.2534 | 8.67882 | 19.9322 | -0.537146 | 113.178 | 643.133 | 673.085 | 645.742 | 695.154 | 0 | 0 | -2.79913 |
| arm_ridge_unbounded | 28.7732 | 680.251 | 653.74 | 26.5112 | 15.7296 | 4.45052 | 8.59309 | 13.0436 | 2.26198 | 113.178 | 288.145 | 798.935 | 446.985 | 881.986 | 0.0751342 | 0.187835 | 0 |
| arm_ridge_bounded | 29.8905 | 680.251 | 653.74 | 26.5112 | 13.971 | 9.82724 | 6.0922 | 15.9194 | 3.37929 | 113.178 | 288.145 | 730 | 446.985 | 730 | 0 | 0 | 1.1173 |
| pooled_interaction_ridge_bounded | 31.8991 | 680.251 | 653.74 | 26.5112 | 4.16287 | 17.9896 | 9.74666 | 27.7362 | 5.38791 | 113.178 | 410.351 | 730 | 459.73 | 730 | 0 | 0 | 3.12593 |
| arm_hist_gradient_boosting_bounded | 27.9158 | 680.251 | 653.74 | 26.5112 | 2.36747 | 37.3882 | -11.8399 | 25.5483 | 1.40461 | 113.178 | 248.548 | 730 | 269.397 | 730 | 0 | 0 | -0.857371 |

## Fold stability

| model | fold_effect_min | fold_effect_max | fold_effect_spread | fold_fraction_positive | loo_effect_min | loo_effect_max | loo_effect_spread | loo_fraction_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_mean | -187.635 | 139.561 | 327.196 | 0.6 | -4.22982 | 76.5897 | 80.8196 | 0.8 |
| arm_ridge_unbounded | -205.352 | 155.702 | 361.054 | 0.6 | -4.97828 | 84.2503 | 89.2286 | 0.8 |
| arm_ridge_bounded | -203.971 | 154.066 | 358.036 | 0.6 | -3.12893 | 85.305 | 88.434 | 0.8 |
| pooled_interaction_ridge_bounded | -201.6 | 158.098 | 359.699 | 0.6 | -1.65846 | 87.228 | 88.8864 | 0.8 |
| arm_hist_gradient_boosting_bounded | -183.427 | 149.231 | 332.658 | 0.6 | -4.34302 | 77.9946 | 82.3376 | 0.8 |

## Interpretation rules

- The Stage 12 estimator is decomposed on its exact IPCW-RMST pseudo-outcome. This avoids
  attributing the whole difference between AIPW and a separately constructed Kaplan-Meier
  estimator to outcome augmentation.
- Bounded and nonlinear outcome models use the same patients, folds, propensity scores,
  censoring model, pseudo-outcomes, and ATO score.
- No model is selected based on a favorable effect estimate.
- Patient-level influence diagnostics contain no patient identifier and remain local-only.
- The 300/200 publication bootstrap remains locked until the outcome-nuisance gate is resolved.
