# Paper A candidate analysis plan

## Status

**CANDIDATE_NOT_LOCKED.** This is the proposed final design after exploratory
source verification, temporal alignment, overlap diagnostics, and censoring
sensitivity analysis.

## Scientific question

Among verified HR-positive/HER2-negative patients alive and uncensored at
day 180 after diagnosis, what is the overlap-population
difference in subsequent restricted mean survival time between patients who
initiated verified hormone therapy by day 180 and patients
who had not initiated by that day?

## Important interpretation

This is an **early-initiation strategy** estimand. Patients initiating after
day 180 remain members of the no-initiation-by-day-180
strategy. The analysis is not interpreted as ever-treated versus never-treated
and not as a sustained-treatment per-protocol effect.

## Primary design

- Cohort: `outer_hormone_hrpos_her2neg`
- Eligible patients: 559
- Treated by landmark: 194
- Not treated by landmark: 365
- Events after landmark: 50
- Time zero: day 180 after diagnosis
- Primary horizon: 730 days after landmark
- Estimand: ATO RMST difference
- Primary propensity: cross-fitted compact clinical plus diagnosis-era model
- Primary censoring nuisance: regularized pooled logistic model
- Primary outcome nuisance: ridge regression
- Primary censoring survival truncation: 0.10

## Primary feasibility result

- AIPW-RMST difference: 28.8 days
- Influence-function diagnostic interval:
  -33.5 to
  91.0 days
- Fixed-nuisance bootstrap interval:
  -36.5 to
  99.0 days
- Maximum weighted SMD: 0.065
- ESS treated/control:
  182.3/299.8

## AI component

Boosted-AI censoring and outcome nuisance models are retained as a
prespecified sensitivity analysis. Their purpose is to assess whether flexible
machine learning improves nuisance prediction without destabilizing causal
weights or effect estimates. The primary estimator remains the more stable
regularized classical nuisance specification.

## Claims

The paper will evaluate reliability, model sensitivity, temporal alignment,
and precision. It will not claim that observational data establish hormone
therapy efficacy.

## Sensitivity analyses

1. censoring truncation at 0.05;
2. boosted-AI nuisance models;
3. 1095-day post-landmark RMST;
4. 365-day landmark;
5. descriptive analysis of later initiators.

## Required before final lock

- professor review of the early-initiation estimand;
- final full-pipeline bootstrap implementation;
- frozen model registry and software versions;
- final figure/table specifications;
- repository hash and git tag.
