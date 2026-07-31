# Outcome-model robustness

## AIPW estimates and components

| model | estimate_days | direct_ato_ipw_treated_mean_days | direct_ato_ipw_control_mean_days | direct_ato_ipw_effect_days | plugin_component_days | treated_residual_component_days | control_residual_component_days | total_residual_augmentation_days | aipw_minus_direct_ato_ipw_days | ato_denominator | mu0_min | mu0_max | mu1_min | mu1_max | fraction_mu0_outside_0_horizon | fraction_mu1_outside_0_horizon | difference_from_exact_ridge_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_mean | 25.974 | 680.251 | 653.74 | 26.5112 | 6.04183 | 11.2534 | 8.67882 | 19.9322 | -0.537146 | 113.178 | 643.133 | 673.085 | 645.742 | 695.154 | 0 | 0 | -2.79913 |
| arm_ridge_unbounded | 28.7732 | 680.251 | 653.74 | 26.5112 | 15.7296 | 4.45052 | 8.59309 | 13.0436 | 2.26198 | 113.178 | 288.145 | 798.935 | 446.985 | 881.986 | 0.0751342 | 0.187835 | 0 |
| arm_ridge_bounded | 29.8905 | 680.251 | 653.74 | 26.5112 | 13.971 | 9.82724 | 6.0922 | 15.9194 | 3.37929 | 113.178 | 288.145 | 730 | 446.985 | 730 | 0 | 0 | 1.1173 |
| pooled_interaction_ridge_bounded | 31.8991 | 680.251 | 653.74 | 26.5112 | 4.16287 | 17.9896 | 9.74666 | 27.7362 | 5.38791 | 113.178 | 410.351 | 730 | 459.73 | 730 | 0 | 0 | 3.12593 |
| arm_hist_gradient_boosting_bounded | 27.9158 | 680.251 | 653.74 | 26.5112 | 2.36747 | 37.3882 | -11.8399 | 25.5483 | 1.40461 | 113.178 | 248.548 | 730 | 269.397 | 730 | 0 | 0 | -0.857371 |

## Factual out-of-fold calibration

| model | arm | n | factual_mse | factual_mae | factual_bias_observed_minus_predicted | ato_weighted_bias_observed_minus_predicted | ato_weighted_mse | prediction_min | prediction_p01 | prediction_median | prediction_p99 | prediction_max | fraction_prediction_below_zero | fraction_prediction_above_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_mean | all | 559 | 106378 | 258.122 | -0.024849 | -0.024849 | 106378 | 643.133 | 643.133 | 668.686 | 695.154 | 695.154 | 0 | 0 |
| arm_mean | 0 | 365 | 96573.1 | 253.278 | -1.24588e-15 | -8.59014 | 107608 | 643.133 | 643.133 | 668.686 | 673.085 | 673.085 | 0 | 0 |
| arm_mean | 1 | 194 | 124827 | 267.237 | -0.0716009 | 11.067 | 115577 | 645.742 | 645.742 | 667.478 | 695.154 | 695.154 | 0 | 0 |
| arm_ridge_unbounded | all | 559 | 109949 | 253.472 | -1.0438 | -1.0438 | 109949 | 288.145 | 535.934 | 666.856 | 791.031 | 881.986 | 0 | 0.101968 |
| arm_ridge_unbounded | 0 | 365 | 99743 | 248.713 | -0.72073 | -8.50529 | 111401 | 288.145 | 538.196 | 664.521 | 784.328 | 798.935 | 0 | 0.0876712 |
| arm_ridge_unbounded | 1 | 194 | 129150 | 262.426 | -1.65164 | 4.3768 | 119519 | 446.985 | 534.845 | 669.634 | 797.469 | 881.986 | 0 | 0.128866 |
| arm_ridge_bounded | all | 559 | 109463 | 252.771 | 2.04449 | 2.04449 | 109463 | 288.145 | 535.934 | 666.856 | 730 | 730 | 0 | 0 |
| arm_ridge_bounded | 0 | 365 | 99186.4 | 247.735 | 1.90554 | -6.02995 | 110669 | 288.145 | 538.196 | 664.521 | 730 | 730 | 0 | 0 |
| arm_ridge_bounded | 1 | 194 | 128799 | 262.247 | 2.30591 | 9.66445 | 119093 | 446.985 | 534.845 | 669.634 | 730 | 730 | 0 | 0 |
| pooled_interaction_ridge_bounded | all | 559 | 107899 | 249.897 | 2.67276 | 2.67276 | 107899 | 410.351 | 542.521 | 662.747 | 730 | 730 | 0 | 0 |
| pooled_interaction_ridge_bounded | 0 | 365 | 97460.4 | 244.444 | -1.63214 | -9.64707 | 108357 | 410.351 | 560.441 | 663.811 | 730 | 730 | 0 | 0 |
| pooled_interaction_ridge_bounded | 1 | 194 | 127537 | 260.157 | 10.7722 | 17.6916 | 118256 | 459.73 | 520.826 | 660.395 | 730 | 730 | 0 | 0 |
| arm_hist_gradient_boosting_bounded | all | 559 | 101702 | 230.862 | 22.0644 | 22.0644 | 101702 | 248.548 | 314.699 | 703.141 | 730 | 730 | 0 | 0 |
| arm_hist_gradient_boosting_bounded | 0 | 365 | 92097.5 | 225.823 | 16.8397 | 11.7189 | 99287.7 | 248.548 | 321.867 | 700.212 | 730 | 730 | 0 | 0 |
| arm_hist_gradient_boosting_bounded | 1 | 194 | 119773 | 240.344 | 31.8943 | 36.7689 | 109537 | 292.06 | 307.303 | 703.804 | 730 | 730 | 0 | 0 |

These models are a fixed robustness registry. They must not be ranked or selected according to
which one produces the most favorable treatment effect.
