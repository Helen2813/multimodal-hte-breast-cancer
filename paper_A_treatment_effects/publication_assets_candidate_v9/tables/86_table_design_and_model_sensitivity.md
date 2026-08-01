| estimand_family | analysis | estimate_days | interval_low_days | interval_high_days | target_or_note |
| --- | --- | --- | --- | --- | --- |
| Design bridge | landmark_unweighted_km | -1.04074 |  |  | observed landmark cohort \| none |
| Design bridge | landmark_overlap_weighted_km | 6.90641 |  |  | landmark ATO \| frozen overlap weights |
| Design bridge | landmark_overlap_aipw | 28.7732 |  |  | landmark ATO \| overlap weights plus outcome augmentation |
| Design bridge | ccw_conditional_post180_original | -6.72989 |  |  | diagnosis-time CCW survivors at day 180 \| clone/adherence weights |
| Design bridge | ccw_conditional_post180_multiplied_by_landmark_ato | -0.0659435 |  |  | landmark ATO bridge \| clone/adherence weights x frozen overlap weights |
| Landmark ATO, same IPCW pseudo-outcome | Direct ATO-IPW | 26.5112 |  |  | Anchor without outcome augmentation |
| Landmark ATO-AIPW outcome-model sensitivity | arm_mean | 25.974 |  |  | Same cohort, folds, propensity, censoring, and pseudo-outcome |
| Landmark ATO-AIPW outcome-model sensitivity | arm_ridge_unbounded | 28.7732 |  |  | Same cohort, folds, propensity, censoring, and pseudo-outcome |
| Landmark ATO-AIPW outcome-model sensitivity | arm_ridge_bounded | 29.8905 |  |  | Same cohort, folds, propensity, censoring, and pseudo-outcome |
| Landmark ATO-AIPW outcome-model sensitivity | pooled_interaction_ridge_bounded | 31.8991 |  |  | Same cohort, folds, propensity, censoring, and pseudo-outcome |
| Landmark ATO-AIPW outcome-model sensitivity | arm_hist_gradient_boosting_bounded | 27.9158 |  |  | Same cohort, folds, propensity, censoring, and pseudo-outcome |
| Locked primary analysis | Candidate V9, 20-partition repeated-score ATO-AIPW | 22.9513 | -4.17379 | 91.0096 | Primary 95% patient-bootstrap percentile interval |
| Diagnosis-time CCW sensitivity (not directly comparable) | original_uncapped | -7.1054 | -28.0531 | 7.08593 | Different time zero and adherence/censoring estimand |
| Diagnosis-time CCW sensitivity (not directly comparable) | cap_5 | -7.00349 | -25.4073 | 6.6794 | Different time zero and adherence/censoring estimand |
| Diagnosis-time CCW sensitivity (not directly comparable) | cap_10 | -7.44053 | -29.3249 | 6.81416 | Different time zero and adherence/censoring estimand |
| Diagnosis-time CCW sensitivity (not directly comparable) | cap_empirical_p99 | -3.7929 | -19.4633 | 6.57061 | Different time zero and adherence/censoring estimand |
