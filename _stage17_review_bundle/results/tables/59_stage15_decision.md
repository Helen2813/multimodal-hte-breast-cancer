# Stage 15 decision report

**Decision:** `ADHERENCE_OR_CENSORING_MODEL_DRIVES_SIGN_DISAGREEMENT_REQUIRE_BRIDGE_ESTIMATOR`

**Protocol status:** `CANDIDATE_V5_NOT_LOCKED`

## Common-target estimator bridge

| analysis | target | adjustment | rmst_effect_days |
| --- | --- | --- | --- |
| landmark_unweighted_km | observed landmark cohort | none | -1.04074 |
| landmark_overlap_weighted_km | landmark ATO | frozen overlap weights | 6.90641 |
| landmark_overlap_aipw | landmark ATO | overlap weights plus outcome augmentation | 28.7732 |
| ccw_conditional_post180_original | diagnosis-time CCW survivors at day 180 | clone/adherence weights | -6.72989 |
| ccw_conditional_post180_multiplied_by_landmark_ato | landmark ATO bridge | clone/adherence weights x frozen overlap weights | -0.0659435 |

## Re-estimated CCW truncation bootstrap

| strategy | successful_reps | bootstrap_mean_days | bootstrap_median_days | bootstrap_sd_days | percentile_ci_low_days | percentile_ci_high_days | fraction_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_uncapped | 30 | -7.1054 | -5.63643 | 10.4311 | -28.0531 | 7.08593 | 0.3 |
| cap_5 | 30 | -7.00349 | -5.61309 | 10.0554 | -25.4073 | 6.6794 | 0.3 |
| cap_10 | 30 | -7.44053 | -5.62649 | 10.7471 | -29.3249 | 6.81416 | 0.3 |
| cap_empirical_p99 | 30 | -3.7929 | -2.83204 | 7.70298 | -19.4633 | 6.57061 | 0.333333 |

## Paired truncation comparisons

| strategy | paired_repetitions | mean_shift_days | median_shift_days | sd_shift_days | sign_agreement | truncation_status |
| --- | --- | --- | --- | --- | --- | --- |
| cap_10 | 30 | -0.335131 | -2.26819e-13 | 0.785978 | 1 | DIRECTION_ROBUST_TO_REESTIMATED_TRUNCATION |
| cap_5 | 30 | 0.101908 | 0.000471487 | 1.62652 | 1 | DIRECTION_ROBUST_TO_REESTIMATED_TRUNCATION |
| cap_empirical_p99 | 30 | 3.3125 | 2.45421 | 3.75829 | 0.966667 | DIRECTION_ROBUST_TO_REESTIMATED_TRUNCATION |

The full publication bootstrap remains locked. No post-hoc selection of the positive landmark
estimate is permitted.
