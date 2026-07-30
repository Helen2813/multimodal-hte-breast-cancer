# Stage 6 — verified sources and corrected cohorts

Before running, copy the original treatment file from the old project:

```text
Thesis_v3\data\drags\clinical.tsv
```

to:

```text
multimodal-hte-breast-cancer\
└── data\
    └── processed\
        └── 01_Clinical\
            └── drags\
                └── clinical.tsv
```

Do not substitute `clinical_filtered_raw.csv`: it contains only generic
`Pharmaceutical Therapy, NOS` and cannot reconstruct hormone, chemotherapy,
or targeted-therapy families.

Then copy this package into the project root and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_stage6_verified_sources.ps1
```

## Stage 16

Recovers only receptor labels that correspond exactly to the two repeated
standardized modes. All non-modal MICE/imputed values remain missing.

Outputs:

```text
data/derived/verified_sources/16_recovered_observed_receptor_labels.csv
results/tables/16_receptor_mode_recovery_summary.csv
results/tables/16_receptor_missingness_patterns.csv
```

## Stage 17

Reconstructs treatment families from the original columns:

```text
cases.submitter_id
treatments.treatment_type
treatments.therapeutic_agents
```

and compares them with the legacy `T_*` flags.

Outputs:

```text
data/derived/verified_sources/17_verified_treatment_flags.csv
results/tables/17_treatment_verification_summary.csv
results/tables/17_original_treatment_type_counts.csv
results/tables/17_therapeutic_agent_counts.csv
results/tables/17_true_treatment_timing_columns.csv
```

## Stage 18

Rebuilds eligibility cohorts using observed receptor labels only:

- HR-positive/HER2-negative: observed HER2-negative and at least one observed
  ER-positive or PR-positive label;
- TNBC: all three observed labels negative.

Outputs:

```text
data/derived/verified_cohorts/
results/tables/18_verified_cohort_summary.csv
results/tables/18_verified_vs_legacy_cohorts.csv
```

## Stage 19

Rebuilds compact adjustment matrices without standardized receptor scores,
then computes cross-fitted propensity, overlap weights, calibration weights,
SMD, and ESS.

Outputs:

```text
data/derived/verified_compact_adjustment/
results/tables/19_verified_balance_summary.csv
results/tables/19_verified_balance_*.csv
results/tables/19_verified_weights_*.csv
```

Do not begin HTE models until Stage 19 has been reviewed.
