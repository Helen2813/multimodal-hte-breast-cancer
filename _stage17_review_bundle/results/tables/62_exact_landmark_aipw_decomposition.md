# Exact Stage 12 landmark AIPW decomposition

| model | estimate_days | direct_ato_ipw_treated_mean_days | direct_ato_ipw_control_mean_days | direct_ato_ipw_effect_days | plugin_component_days | treated_residual_component_days | control_residual_component_days | total_residual_augmentation_days | aipw_minus_direct_ato_ipw_days | ato_denominator | exact_stage12_estimate_days | replication_difference_days | horizon_days | n | treated | control | events | pseudo_mean | pseudo_sd | pseudo_p99 | pseudo_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_ridge_unbounded_exact_stage12 | 28.7732 | 680.251 | 653.74 | 26.5112 | 15.7296 | 4.45052 | 8.59309 | 13.0436 | 2.26198 | 113.178 | 28.7732 | 0 | 730 | 559 | 194 | 365 | 50 | 664.849 | 323.962 | 1569.54 | 2264.14 |

## Factual out-of-fold prediction diagnostics

| model | arm | n | factual_mse | factual_mae | factual_bias_observed_minus_predicted | ato_weighted_bias_observed_minus_predicted | ato_weighted_mse | prediction_min | prediction_p01 | prediction_median | prediction_p99 | prediction_max | fraction_prediction_below_zero | fraction_prediction_above_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_ridge_unbounded | all | 559 | 109949 | 253.472 | -1.0438 | -1.0438 | 109949 | 288.145 | 535.934 | 666.856 | 791.031 | 881.986 | 0 | 0.101968 |
| arm_ridge_unbounded | 0 | 365 | 99743 | 248.713 | -0.72073 | -8.50529 | 111401 | 288.145 | 538.196 | 664.521 | 784.328 | 798.935 | 0 | 0.0876712 |
| arm_ridge_unbounded | 1 | 194 | 129150 | 262.426 | -1.65164 | 4.3768 | 119519 | 446.985 | 534.845 | 669.634 | 797.469 | 881.986 | 0 | 0.128866 |

The `direct_ato_ipw_effect_days` uses the same IPCW-RMST pseudo-outcome and frozen propensity
scores as the AIPW estimator but removes outcome augmentation. It is a cleaner bridge than
comparing AIPW directly with a separately constructed weighted Kaplan-Meier curve.

Patient-level component rows are written without patient identifiers under
`data/derived/stage16` and must not be committed.
