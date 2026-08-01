# Stage 22 publication-asset report

## Status

`PUBLICATION_ASSETS_GENERATED_FROM_LOCKED_CANDIDATE_V9`

- Locked manifest unchanged after asset generation: **True**
- Generated files: **36**
- Potential stale manuscript claims flagged: **0**

## Locked primary result

| Population | N | Treated | Control | Events | Estimand | Point estimate (days) | 95% percentile CI | Bootstrap positive | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Verified HR+/HER2- day-180 landmark survivors | 559 | 194 | 365 | 50 | ATO difference in 730-day post-landmark RMST | 22.9513 | -4.17 to 91.01 | 0.956667 | Positive direction, statistically imprecise; interval includes zero |

## Interval sensitivity

| Interval | Low (days) | High (days) | Includes zero | Role |
| --- | --- | --- | --- | --- |
| Percentile patient bootstrap (primary) | -4.17379 | 91.0096 | True | Primary inference |
| Basic patient bootstrap | -45.107 | 50.0764 | True | Sensitivity |
| Studentized patient bootstrap | -23.6369 | 47.9554 | True | Sensitivity |
| Influence-function diagnostic | -38.0822 | 83.9848 | True | Diagnostic only |

## Bootstrap convergence

| prefix_repetitions | mean_effect_days | median_effect_days | sd_effect_days | percentile_ci_low_days | percentile_ci_high_days | basic_ci_low_days | basic_ci_high_days | fraction_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 33.595 | 32.157 | 21.3101 | -4.2799 | 85.8161 | -39.9135 | 50.1825 | 0.94 |
| 100 | 35.2718 | 33.2943 | 21.7671 | -4.17379 | 82.8193 | -36.9168 | 50.0764 | 0.96 |
| 200 | 36.1701 | 33.3428 | 22.687 | -3.22174 | 81.8108 | -35.9082 | 49.1243 | 0.96 |
| 300 | 37.6369 | 36.2558 | 23.0294 | -4.17379 | 91.0096 | -45.107 | 50.0764 | 0.956667 |

## Main generated figures

- `figures/87_bootstrap_distribution.png` and `.svg`
- `figures/87_bootstrap_ecdf.png` and `.svg`
- `figures/87_bootstrap_prefix_convergence.png` and `.svg`
- `figures/87_inner_partition_mcse.png` and `.svg` when the MCSE field is available
- `figures/87_landmark_sensitivity_forest.png` and `.svg`

## Manuscript text

- Methods, Results, Discussion, Conclusion, and abstract-result snippets are under `manuscript_snippets/`.
- The original manuscript was not overwritten.
- Review `audit/88_stale_claim_audit.csv` before integrating the snippets.

## Interpretation boundary

The locked point estimate is positive, but the prespecified primary patient-bootstrap interval includes zero. The final manuscript must report statistical imprecision and must not convert the high fraction of positive bootstrap repetitions into a significance claim.

## Next scientific step

Complete manuscript integration and then run a METABRIC data-and-design audit. Do not copy the TCGA day-180 estimand unless METABRIC contains compatible treatment-initiation timing.
