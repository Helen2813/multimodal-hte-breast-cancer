# Paper B analysis plan — DRAFT, not yet registered

## Working title

Prognostic and prescriptive utility of correlated multi-omics modalities for
heterogeneous treatment-effect estimation

## Primary contribution

A cross-fitted statistical framework separating prognostic modality utility from
incremental prescriptive utility under confounding, censoring, correlated modalities,
and limited event counts.

## Simulation-first requirement

Operating characteristics must be established before the final real-data modality
comparison. Simulations will assess bias, interval coverage, false modality discovery,
ranking accuracy, policy regret, censoring, overlap, modality correlation, and sample
size/event-count limitations.

## Primary real-data application

Verified outer HR-positive/HER2-negative hormone cohort:

1. clinical only;
2. clinical + RNA.

RNA is nearly universally available and can be compared on the same population.

## Exploratory complete-omics application

Verified complete-case hormone cohort:

1. clinical only;
2. clinical + RNA;
3. clinical + CNV;
4. clinical + mutation;
5. clinical + methylation;
6. clinical + miRNA;
7. clinical + protein;
8. clinical + all six omics.

These are prespecified contrasts. The 64-subset powerset is not a confirmatory analysis.
If calculated, it is descriptive Supplement material only.

## TNBC application

Outer TNBC chemotherapy may be used only as an exploratory treatment-arm replication
when Stage 21 balance and ESS are adequate. Complete-case TNBC is excluded from formal
modality attribution because of low event counts and unstable balance.

## Multiplicity

Primary modality contrasts use simultaneous bootstrap intervals or a family-wise
procedure such as Holm/max-T. No best-modality claim is selected from an unrestricted
powerset search.

## Evaluation

Every modality model uses the same verified patient splits:

`C:\Users\olegk\Desktop\multimodal-hte-breast-cancer\data\derived\verified_splits\23_verified_repeated_fold_assignments.csv`

Feature selection, dimension reduction, nuisance tuning, and HTE tuning occur strictly
inside training folds.

## Transparency

This plan is frozen only after simulation settings, utility estimands, and final model
registry are written and hashed.
