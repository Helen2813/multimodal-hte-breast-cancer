# Paper A analysis plan — DRAFT, not yet registered

## Working title

Source-verified and survival-aware treatment-effect estimation in observational breast-cancer data

## Status

This is a prospectively locked-plan draft following exploratory protocol development.
It is not a retrospective preregistration. Do not label it final until the Stage 20–23
tables are reviewed and the final model code is frozen.

## Primary population

Verified outer HR-positive/HER2-negative cohort.

## Primary treatment

Verified hormone-therapy indicator reconstructed from the original `clinical.tsv`.

## Primary estimand

Five-year restricted mean survival time difference in the clinical-overlap population
(ATO estimand), supplemented by the five-year survival-probability difference.

## Primary adjustment

Prespecified compact baseline clinical adjustment set, with verified diagnosis year
added when adequately observed.

## Sensitivity adjustment

Full baseline clinical elastic-net propensity model with cross-fitting and overlap
weighting. Post-treatment, outcome, administrative, and standardized receptor-score
variables are excluded.

## Secondary population

Verified outer TNBC chemotherapy cohort only if Stage 21 classifies it as at least
`EXPLORATORY_ONLY`. It cannot support a confirmatory causal claim when common support,
control ESS, or residual balance are inadequate.

## Survival analysis

The final confirmatory estimator must be censoring-aware and doubly robust. Weighted
Kaplan–Meier estimates are diagnostics only. Final uncertainty must account for
nuisance-model estimation.

## Timing limitation

Formal target-trial emulation or landmark alignment will only be claimed if Stage 20
shows adequate true treatment-start coverage. Administrative created/updated timestamps
must not be used as treatment initiation dates.

## Multiplicity

One primary treatment arm, one primary estimand, and one primary adjustment strategy.
Other arms, estimands, and propensity strategies are sensitivity or exploratory analyses.

## Transparency

All final analyses use the verified cohort files and the frozen repeated split manifest:

`C:\Users\olegk\Desktop\multimodal-hte-breast-cancer\data\derived\verified_splits\23_verified_repeated_fold_assignments.csv`
