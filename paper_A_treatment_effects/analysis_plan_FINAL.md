# Paper A final locked analysis plan: Candidate V9

**Protocol ID:** `PAPER_A_CANDIDATE_V9_9F1E8E368C92EC02`  
**Status:** `PAPER_A_CANDIDATE_V9_LOCKED`  
**Disclosure:** Prospectively locked final analysis following exploratory protocol development on the same data.

## Scientific question

Among verified HR-positive/HER2-negative patients who are alive and eligible at the day-180 landmark, what is the overlap-population difference in subsequent 730-day restricted mean survival time between hormone-therapy initiation during days 0-180 and no initiation by day 180?

## Primary estimand

- Population: verified HR-positive/HER2-negative day-180 landmark survivors.
- Treatment contrast: hormone therapy initiation during days 0-180 versus no initiation by day 180.
- Target population: treatment-overlap population (ATO).
- Outcome: 730-day post-landmark RMST.
- Effect scale: treated minus control, in days.
- Locked point estimate before publication bootstrap: 22.951284 days.
- Influence-function interval at the locked point estimate is diagnostic; the primary final interval is the patient-bootstrap percentile interval.

## Locked estimator

- Five-fold cross-fitting.
- Twenty fixed nuisance partitions with seeds: 18101, 18102, 18103, 18104, 18105, 18106, 18107, 18108, 18109, 18110, 18111, 18112, 18113, 18114, 18115, 18116, 18117, 18118, 18119, 18120.
- Propensity: cross-fitted refitted exact Stage 30 regularized specification.
- Censoring: cross-fitted discrete-time regularized logistic censoring model.
- Outcome nuisance: cross-fitted arm-specific ridge regression on the IPCW-RMST pseudo-outcome, bounded to [0, 730] days.
- Censoring survival floor: G-min = 0.10.
- Discrete-time interval: 90 days.
- Patient-level repeated-score aggregation across all 20 partitions.

## Locked publication bootstrap

- 300 ordinary patient bootstrap repetitions.
- Bootstrap base seed: 21000.
- Every bootstrap copy of the same patient remains in one nuisance fold.
- All propensity, censoring, and outcome nuisance models are refitted within every bootstrap partition.
- Primary interval: 95% percentile patient-bootstrap interval.
- Sensitivity intervals: 95% basic patient-bootstrap interval, 95% studentized patient-bootstrap interval.
- No change to treatment definition, cohort, horizon, G-min, learners, partitions, seeds, bounding, or interval method is permitted after inspection of the publication-bootstrap distribution.

## Identification assumptions

1. Consistency of the recorded treatment strategy.
2. Conditional exchangeability after the locked baseline adjustment set.
3. Positivity in the overlap population.
4. Conditional independent censoring given treatment and locked baseline covariates.
5. Valid source classification of receptor status, treatment, event, and baseline fields.

## Interpretation boundary

The primary result is an observational ATO RMST contrast, not an unconditional efficacy estimate and not a formal randomized-trial result. Treatment initiation timing is reconstructed from recorded clinical fields, and residual confounding remains possible.

## Stage 19 stabilization evidence

| stage19_decision | protocol_status | completed_bootstrap_repetitions | partitions_per_bootstrap | prefix20_mean_effect_days | prefix20_median_effect_days | prefix20_sd_effect_days | prefix20_percentile_ci_low_days | prefix20_percentile_ci_high_days | prefix20_fraction_positive | median_absolute_10_to_20_shift_days | p95_absolute_10_to_20_shift_days | median_mcse_at_20_days | p95_mcse_at_20_days | full_publication_bootstrap_locked | full_bootstrap_authorized_after_protocol_lock | claim_status | recommended_next_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INNER_CROSSFIT_STABILIZED_PREPARE_CANDIDATE_V9_PROTOCOL_LOCK | CANDIDATE_V9_INNER_CROSSFIT_STABILIZATION_NOT_LOCKED | 30 | 20 | 39.5082 | 41.758 | 21.476 | 10.9722 | 79.8504 | 1 | 2.90149 | 10.8467 | 4.23679 | 6.79297 | True | True | The estimator is computationally stable after repeated nuisance-partition averaging. Treatment-effect magnitude remains subject to sampling uncertainty and will not be claimed until the locked full bootstrap is complete. | Freeze the 20-partition inner repeated-cross-fit estimator, record code/config/input hashes, and then run the 300-repetition patient bootstrap. The final bootstrap must use the locked 20 partition seeds and may not be modified after distribution inspection. |

## Candidate V9 prefix convergence on the original cohort

| prefix_partitions | estimate_days | if_se_days | if_ci_low_days | if_ci_high_days | partition_mean_days | partition_sd_days | partition_mcse_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 25.1004 | 31.0754 | -35.8072 | 86.0081 | 25.0736 | 5.10296 | 2.28211 |
| 10 | 24.8225 | 30.9668 | -35.8723 | 85.5174 | 24.8058 | 6.01763 | 1.90294 |
| 15 | 22.798 | 31.0678 | -38.095 | 83.6909 | 22.7972 | 7.75216 | 2.0016 |
| 20 | 22.9513 | 31.1396 | -38.0822 | 83.9848 | 22.9408 | 7.58452 | 1.69595 |
