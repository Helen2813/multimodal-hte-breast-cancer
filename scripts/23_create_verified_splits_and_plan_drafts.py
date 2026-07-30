from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _common import (
    PROJECT_ROOT,
    DERIVED_DIR,
    RESULTS_DIR,
    ensure_dirs,
    read_table,
)


SEEDS = [42, 123, 456, 789, 1337]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_strategy(df: pd.DataFrame) -> tuple[pd.Series, int, str]:
    t = pd.to_numeric(
        df["analysis_treatment"], errors="raise"
    ).astype(int)
    e = pd.to_numeric(
        df["analysis_event"], errors="raise"
    ).astype(int)
    joint = (2 * t + e).astype(str)
    counts = joint.value_counts()

    for folds in (5, 4, 3):
        if counts.min() >= folds:
            return joint, folds, "treatment_x_event"
    for folds in (5, 4, 3):
        if t.value_counts().min() >= folds:
            return t.astype(str), folds, "treatment"
    raise ValueError("No valid repeated stratified split is possible.")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    split_dir = DERIVED_DIR / "verified_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"
    cohort_dir = DERIVED_DIR / "verified_cohorts"

    print("=" * 110)
    print("STAGE 23 — VERIFIED SPLITS AND TWO-PAPER ANALYSIS PLAN DRAFTS")
    print("=" * 110)

    assignments = []
    summary_rows = []
    input_manifest = []

    for path in sorted(cohort_dir.glob("*_verified.csv")):
        cohort = path.stem.replace("_verified", "")
        df = read_table(path)
        strata, folds, strategy = split_strategy(df)

        print("\n" + "-" * 110)
        print(
            f"COHORT: {cohort}; n={len(df)}; folds={folds}; "
            f"stratification={strategy}"
        )
        print("Stratum counts:")
        print(strata.value_counts().sort_index().to_string())

        base = pd.DataFrame(
            {
                "patient_id_normalized": df[
                    "patient_id_normalized"
                ],
                "analysis_treatment": pd.to_numeric(
                    df["analysis_treatment"], errors="raise"
                ).astype(int),
                "analysis_event": pd.to_numeric(
                    df["analysis_event"], errors="raise"
                ).astype(int),
            }
        )

        for repeat, seed in enumerate(SEEDS, start=1):
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )
            fold_assignment = np.full(len(df), -1)
            for fold, (_, test_idx) in enumerate(
                splitter.split(np.zeros(len(df)), strata), start=1
            ):
                fold_assignment[test_idx] = fold

            one = base.copy()
            one["cohort"] = cohort
            one["repeat"] = repeat
            one["seed"] = seed
            one["fold"] = fold_assignment.astype(int)
            one["n_folds"] = folds
            one["stratification"] = strategy
            assignments.append(one)

            print(f"\nRepeat {repeat}, seed {seed}")
            fold_rows = []
            for fold in range(1, folds + 1):
                test = one[one["fold"] == fold]
                train = one[one["fold"] != fold]
                row = {
                    "cohort": cohort,
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "n_folds": folds,
                    "stratification": strategy,
                    "train_n": len(train),
                    "train_treated": int(
                        train["analysis_treatment"].sum()
                    ),
                    "train_events": int(
                        train["analysis_event"].sum()
                    ),
                    "test_n": len(test),
                    "test_treated": int(
                        test["analysis_treatment"].sum()
                    ),
                    "test_controls": int(
                        len(test)
                        - test["analysis_treatment"].sum()
                    ),
                    "test_events": int(
                        test["analysis_event"].sum()
                    ),
                }
                summary_rows.append(row)
                fold_rows.append(row)
            print(
                pd.DataFrame(fold_rows)[
                    [
                        "fold",
                        "test_n",
                        "test_treated",
                        "test_controls",
                        "test_events",
                    ]
                ].to_string(index=False)
            )

        input_manifest.append(
            {
                "cohort": cohort,
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(df),
                "columns": len(df.columns),
            }
        )

    assignment_df = pd.concat(assignments, ignore_index=True)
    split_summary = pd.DataFrame(summary_rows)
    assignment_path = (
        split_dir / "23_verified_repeated_fold_assignments.csv"
    )
    assignment_df.to_csv(assignment_path, index=False)
    split_summary.to_csv(
        table_dir / "23_verified_split_summary.csv", index=False
    )

    propensity_summary_path = (
        table_dir / "21_propensity_strategy_summary.csv"
    )
    propensity = (
        read_table(propensity_summary_path)
        if propensity_summary_path.exists()
        else pd.DataFrame()
    )
    event_path = table_dir / "20_event_definition_summary.csv"
    event_summary = (
        read_table(event_path) if event_path.exists() else pd.DataFrame()
    )

    paper_a_dir = PROJECT_ROOT / "paper_A_treatment_effects"
    paper_b_dir = PROJECT_ROOT / "paper_B_modality_utility"

    paper_a = f"""# Paper A analysis plan — DRAFT, not yet registered

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

`{assignment_path}`

"""

    paper_b = f"""# Paper B analysis plan — DRAFT, not yet registered

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

`{assignment_path}`

Feature selection, dimension reduction, nuisance tuning, and HTE tuning occur strictly
inside training folds.

## Transparency

This plan is frozen only after simulation settings, utility estimands, and final model
registry are written and hashed.

"""

    write_text(paper_a_dir / "analysis_plan_DRAFT.md", paper_a)
    write_text(paper_b_dir / "analysis_plan_DRAFT.md", paper_b)

    configs = {
        paper_a_dir / "primary_estimand_DRAFT.json": {
            "paper": "A",
            "primary_cohort": "outer_hormone_hrpos_her2neg",
            "primary_treatment": "verified_hormone_therapy",
            "primary_estimand": "five_year_RMST_difference_ATO",
            "secondary_estimand": "five_year_survival_probability_difference_ATO",
            "primary_adjustment": "compact_baseline_plus_diagnosis_year_if_available",
            "sensitivity_adjustment": "full_baseline_elastic_net_overlap",
            "confirmatory_status": "DRAFT_NOT_LOCKED",
        },
        paper_b_dir / "primary_estimand_DRAFT.json": {
            "paper": "B",
            "primary_application": "outer_hormone_hrpos_her2neg_clinical_vs_RNA",
            "complete_omics_application": "complete_case_hormone_exploratory",
            "primary_comparisons": [
                "clinical_vs_clinical_plus_RNA",
                "clinical_vs_clinical_plus_CNV",
                "clinical_vs_clinical_plus_mutation",
                "clinical_vs_clinical_plus_methylation",
                "clinical_vs_clinical_plus_miRNA",
                "clinical_vs_clinical_plus_protein",
                "clinical_vs_clinical_plus_all_omics",
            ],
            "powerset_confirmatory": False,
            "simulation_first": True,
            "confirmatory_status": "DRAFT_NOT_LOCKED",
        },
    }
    for path, data in configs.items():
        write_text(path, json.dumps(data, indent=2))

    manifest = {
        "status": "DRAFT_NOT_LOCKED",
        "verified_cohort_inputs": input_manifest,
        "split_assignment": {
            "path": str(assignment_path),
            "sha256": sha256(assignment_path),
        },
        "supporting_tables": {
            "event_definition": str(event_path),
            "propensity_strategy": str(propensity_summary_path),
        },
    }
    manifest_path = (
        DERIVED_DIR / "manifests" / "23_design_lock_DRAFT.json"
    )
    write_text(manifest_path, json.dumps(manifest, indent=2))

    print("\n" + "=" * 110)
    print("STAGE 23 OUTPUTS")
    print("=" * 110)
    print(f"Verified fold assignments: {assignment_path}")
    print(f"Paper A draft plan: {paper_a_dir / 'analysis_plan_DRAFT.md'}")
    print(f"Paper B draft plan: {paper_b_dir / 'analysis_plan_DRAFT.md'}")
    print(f"Draft hash manifest: {manifest_path}")
    print("\nIMPORTANT: protocol status is DRAFT_NOT_LOCKED.")
    print(
        "Do not preregister or label the protocol final until Stage 20–23 "
        "outputs have been reviewed."
    )

    if not propensity.empty:
        print("\nCurrent propensity decisions")
        print(propensity.to_string(index=False))
    if not event_summary.empty:
        print("\nCurrent event-definition summary")
        print(event_summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
