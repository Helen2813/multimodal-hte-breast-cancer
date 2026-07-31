# Paper A analysis plan — Candidate V2

## Status

**CANDIDATE_V2_NOT_LOCKED.**

## Design terminology

This is a 180-day **landmark analysis with a treatment-initiation grace period**.
It compares two strategies among patients alive and uncensored at day 180:

1. initiate verified hormone therapy between diagnosis and day 180;
2. do not initiate by day 180.

The second strategy permits treatment initiation after day 180. Therefore, the
analysis does not compare permanently treated and permanently untreated patients.

## Primary estimand

The primary estimand is the overlap-population difference in 730-day post-landmark
restricted mean survival time among day-180 survivors.

Candidate estimate: 28.8 days, with the current
diagnostic interval -33.5 to
91.0 days.

## Control-strategy composition

The no-initiation-by-day-180 strategy contains:

- 177 patients with recorded initiation after day 180;
- 188 patients with no recorded later initiation.

The maximum absolute baseline SMD between these components is
0.450; the cross-fitted AUC for predicting later initiation is
0.617. These groups are described to clarify the strategy composition,
not treated as separate causal arms.

## Clone-censor-weight sensitivity

A full clone-censor-weight analysis will be implemented as a diagnosis-time sensitivity analysis.

CCW feasibility status: `CCW_SENSITIVITY_FEASIBLE`. Stage 39 is a weight/positivity diagnostic
only and does not itself estimate a treatment effect.

## Identifiability assumptions

1. recorded initiation timing is sufficiently accurate;
2. treatment strategies are conditionally exchangeable given baseline W;
3. positivity holds in the day-180 survivor population;
4. natural censoring is conditionally independent;
5. receptor, event, and clinical measurements are valid;
6. no interference occurs;
7. conditioning on survival and observation through day 180 defines the target
   population rather than estimating a diagnosis-time population;
8. later initiation is allowed after the grace period in the comparison strategy.

## Era analysis

Era-by-strategy interaction is `descriptive_only_due_to_sparse_event_cells`. When event cells are
sparse, era-specific results will be descriptive and will not support subgroup-effect
claims.

## AI interpretation

In this application, boosted nuisance models generated more extreme censoring-weight
behaviour and wider uncertainty than the regularized classical specification. This
statement is application-specific; the paper will not generalize that AI methods are
intrinsically less reliable.

## Changes relative to the earlier exploratory analysis

- imputed receptor scores were no longer thresholded as observed labels;
- treatment families were reconstructed from the authoritative source;
- five-year binary mortality was not treated as fully observed;
- ever-treated exposure was replaced by a time-aligned strategy;
- TNBC chemotherapy failed the overlap/balance/ESS gate;
- targeted therapy was too sparse for a primary analysis;
- individual treatment-effect claims were removed;
- the estimand became a landmark RMST contrast in the overlap population.

## Reporting assets

The staged pipeline now creates a cohort flow diagram, Table 1 before and after
overlap weighting, a primary love plot, a control-composition love plot, and complete
console transcripts.

## Before protocol lock

Professor review, full-pipeline bootstrap, the planned CCW sensitivity when feasible,
clinical contextualization, frozen software/model registry, and a git-tagged hash
manifest remain required.

## Stage 13 estimand-harmonization amendment

Status: **CANDIDATE_V3_NOT_LOCKED**.

The 180-day landmark and diagnosis-time clone-censor-weight analyses are retained as separate
estimands. They differ in time zero, eligibility conditioning, follow-up scale, and target weighting.
Their point estimates must not be pooled or used interchangeably. The final manuscript will frame
opposite-direction estimates as design sensitivity and will not select an analysis based on effect
direction.

Current gate: `CENTERING_PASSED_EXPORT_CCW_CURVES_BEFORE_FULL_BOOTSTRAP`.

Generated: 2026-07-30T01:39:31.153120+00:00.

## Stage 14 CCW curve and bootstrap-weight amendment

Status: **CANDIDATE_V4_NOT_LOCKED**.

The diagnosis-time clone-censor-weight survival curves were exported and decomposed into
day-0-to-day-180 and day-180-to-day-910 components. This decomposition is descriptive and does not
make the CCW and landmark ATO estimands interchangeable.

Stage 14 decision: `RUN_REESTIMATED_CCW_WEIGHT_TRUNCATION_SENSITIVITY_BEFORE_PUBLICATION_BOOTSTRAP`.

The 30-repetition landmark bootstrap centering gate passed but is described as
`BORDERLINE_ACCEPTABLE`. Repetition-level maximum clone weights are explicitly reported. When their
instability gate is triggered, a re-estimated weight-truncation sensitivity is required before the
publication bootstrap.

Generated: 2026-07-30T02:30:01.015621+00:00.

## Stage 15 common-target and re-estimated truncation amendment

Status: **CANDIDATE_V5_NOT_LOCKED**.

Decision: `ADHERENCE_OR_CENSORING_MODEL_DRIVES_SIGN_DISAGREEMENT_REQUIRE_BRIDGE_ESTIMATOR`.

Bridge classification: `CCW_ADHERENCE_OR_CENSORING_MODEL_REMAINS_PRIMARY_DIFFERENCE`.

The re-estimated truncation bootstrap is interpreted as
`direction-robust`. The publication bootstrap remains
locked until the estimator bridge is reflected in the final protocol.

Generated: 2026-07-30T20:17:19.226153+00:00.
