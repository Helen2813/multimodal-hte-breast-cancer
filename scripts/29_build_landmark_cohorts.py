from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    PROCESSED_DIR,
    DERIVED_DIR,
    RESULTS_DIR,
    ensure_dirs,
    normalize_patient_id,
    read_table,
)
from _stage9_utils import LANDMARKS, stratified_folds


FAMILY = {
    "outer_hormone_hrpos_her2neg": ("hormone", ("hormone", "endocrine")),
    "outer_chemo_tnbc": ("chemo", ("chemo",)),
}


def clinical_path() -> Path:
    for path in (
        PROCESSED_DIR / "01_Clinical" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv",
    ):
        if path.exists():
            return path
    raise FileNotFoundError("Original clinical.tsv not found.")


def earliest_start(
    clinical: pd.DataFrame,
    terms: tuple[str, ...],
) -> pd.DataFrame:
    text = clinical["treatments.treatment_type"].astype(str).str.lower()
    mask = np.zeros(len(clinical), dtype=bool)
    for term in terms:
        mask |= text.str.contains(term, na=False).to_numpy()
    start = pd.to_numeric(
        clinical.loc[mask, "treatments.days_to_treatment_start"],
        errors="coerce",
    )
    temp = pd.DataFrame(
        {
            "patient_id_normalized": clinical.loc[
                mask, "cases.submitter_id"
            ].map(normalize_patient_id),
            "start": start,
        }
    )
    return (
        temp.groupby("patient_id_normalized", as_index=False)
        .agg(
            any_start_records=("start", "size"),
            nonmissing_start_records=("start", "count"),
            earliest_start_any=("start", "min"),
            earliest_start_nonnegative=(
                "start",
                lambda s: s[s >= 0].min() if (s >= 0).any() else np.nan,
            ),
            negative_start_records=("start", lambda s: int((s < 0).sum())),
        )
    )


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "verified_cohorts"
    compact_dir = DERIVED_DIR / "verified_compact_adjustment"
    landmark_dir = DERIVED_DIR / "landmark_cohorts"
    landmark_compact_dir = DERIVED_DIR / "landmark_compact"
    split_dir = DERIVED_DIR / "landmark_splits"
    for d in (landmark_dir, landmark_compact_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    clinical = pd.read_csv(clinical_path(), sep="\t", low_memory=False)
    years = read_table(
        DERIVED_DIR / "manifests" / "20_patient_diagnosis_year.csv"
    )[["patient_id_normalized", "diagnosis_year"]]

    print("=" * 115)
    print("STAGE 29 — LANDMARK COHORT CONSTRUCTION")
    print("=" * 115)

    summary_rows = []
    exclusion_rows = []

    for cohort, (_, terms) in FAMILY.items():
        source = read_table(cohort_dir / f"{cohort}_verified.csv")
        compact = read_table(
            compact_dir / f"{cohort}_compact_verified.csv"
        )
        starts = earliest_start(clinical, terms)

        base = (
            source.merge(
                starts,
                on="patient_id_normalized",
                how="left",
                validate="one_to_one",
            )
            .merge(
                years,
                on="patient_id_normalized",
                how="left",
                validate="one_to_one",
            )
        )
        legacy_treated = pd.to_numeric(
            base["analysis_treatment"], errors="raise"
        ).astype(int)

        print("\n" + "=" * 115)
        print(f"SOURCE COHORT: {cohort}")
        print(
            f"n={len(base)}, verified-ever-treated={int(legacy_treated.sum())}, "
            f"controls={int((legacy_treated == 0).sum())}, "
            f"events={int(pd.to_numeric(base['analysis_event']).sum())}"
        )

        for landmark in LANDMARKS:
            alive_uncensored = pd.to_numeric(
                base["analysis_time"], errors="coerce"
            ).gt(landmark)

            ambiguous_treated = (
                legacy_treated.eq(1)
                & (
                    base["earliest_start_nonnegative"].isna()
                    | base["earliest_start_nonnegative"].gt(3650)
                )
            )
            valid = alive_uncensored & ~ambiguous_treated
            lm = base.loc[valid].copy()

            lm["verified_ever_treated"] = legacy_treated.loc[valid].to_numpy()
            lm["analysis_treatment"] = (
                lm["earliest_start_nonnegative"]
                .between(0, landmark)
                .astype(int)
            )
            lm["later_initiator"] = (
                lm["earliest_start_nonnegative"].gt(landmark)
            ).astype(int)
            lm["analysis_time"] = (
                pd.to_numeric(lm["analysis_time"], errors="coerce")
                - landmark
            )
            lm["analysis_event"] = pd.to_numeric(
                lm["analysis_event"], errors="raise"
            ).astype(int)
            lm["landmark_day"] = landmark
            lm["exposure_definition"] = (
                f"initiation_within_{landmark}_days_after_diagnosis"
            )

            out_name = f"{cohort}_landmark{landmark}"
            lm.to_csv(
                landmark_dir / f"{out_name}.csv", index=False
            )

            c = compact.merge(
                lm[
                    [
                        "patient_id_normalized",
                        "analysis_treatment",
                        "analysis_event",
                        "analysis_time",
                        "diagnosis_year",
                        "later_initiator",
                    ]
                ],
                on="patient_id_normalized",
                how="inner",
                validate="one_to_one",
                suffixes=("_old", ""),
            )
            drop_old = [
                col
                for col in (
                    "analysis_treatment_old",
                    "analysis_event_old",
                    "analysis_time_old",
                )
                if col in c.columns
            ]
            c = c.drop(columns=drop_old)
            c["diagnosis_year_missing"] = (
                c["diagnosis_year"].isna().astype(float)
            )
            c.to_csv(
                landmark_compact_dir / f"{out_name}_compact.csv",
                index=False,
            )

            splits = stratified_folds(c)
            splits["cohort"] = out_name
            splits.to_csv(
                split_dir / f"{out_name}_splits.csv", index=False
            )

            a = c["analysis_treatment"].astype(int)
            e = c["analysis_event"].astype(int)
            later = c["later_initiator"].astype(int)
            summary = {
                "cohort": cohort,
                "landmark_day": landmark,
                "source_n": len(base),
                "eligible_landmark_n": len(c),
                "treated_by_landmark": int(a.sum()),
                "not_treated_by_landmark": int((a == 0).sum()),
                "later_initiators_in_control_strategy": int(
                    ((a == 0) & (later == 1)).sum()
                ),
                "events_after_landmark": int(e.sum()),
                "events_treated": int(e[a == 1].sum()),
                "events_control": int(e[a == 0].sum()),
                "excluded_dead_or_censored_before_landmark": int(
                    (~alive_uncensored).sum()
                ),
                "excluded_ambiguous_treatment_timing": int(
                    (alive_uncensored & ambiguous_treated).sum()
                ),
                "median_followup_after_landmark_days": float(
                    c["analysis_time"].median()
                ),
                "known_at_2y_post_landmark": int(
                    c["analysis_time"].ge(730).sum()
                ),
                "known_at_3y_post_landmark": int(
                    c["analysis_time"].ge(1095).sum()
                ),
            }
            summary_rows.append(summary)

            exclusion_rows.append(
                {
                    "cohort": cohort,
                    "landmark_day": landmark,
                    "dead_or_censored_before_landmark": int(
                        (~alive_uncensored).sum()
                    ),
                    "verified_treated_missing_or_extreme_start": int(
                        ambiguous_treated.sum()
                    ),
                    "negative_start_records_patients": int(
                        base["negative_start_records"].fillna(0).gt(0).sum()
                    ),
                }
            )

            print("\nLandmark:", landmark)
            print(pd.DataFrame([summary]).to_string(index=False))
            print("\nTreatment × event after landmark")
            print(
                pd.crosstab(
                    c["analysis_treatment"],
                    c["analysis_event"],
                    margins=True,
                ).to_string()
            )
            print("\nEra × treatment")
            era = pd.cut(
                c["diagnosis_year"],
                [-np.inf, 1999, 2004, 2009, np.inf],
                labels=["<2000", "2000-2004", "2005-2009", ">=2010"],
            ).astype("object").fillna("unknown")
            print(
                pd.crosstab(
                    era,
                    c["analysis_treatment"].map(
                        {0: "not_by_landmark", 1: "treated_by_landmark"}
                    ),
                    margins=True,
                ).to_string()
            )

    summary_df = pd.DataFrame(summary_rows)
    exclusion_df = pd.DataFrame(exclusion_rows)
    summary_df.to_csv(
        table_dir / "29_landmark_cohort_summary.csv", index=False
    )
    exclusion_df.to_csv(
        table_dir / "29_landmark_exclusion_summary.csv", index=False
    )

    print("\n" + "=" * 115)
    print("FINAL LANDMARK COHORT SUMMARY")
    print("=" * 115)
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
