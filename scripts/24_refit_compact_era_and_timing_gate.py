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
from _stage8_utils import (
    cohort_key,
    get_repeat_assignments,
    load_compact_with_year,
    crossfit_propensity,
    weighted_smd,
    effective_sample_size,
)


FAMILY_FOR_COHORT = {
    "hormone_hrpos_her2neg": "hormone",
    "chemo_tnbc": "chemo",
}
FAMILY_TERMS = {
    "hormone": ("hormone", "endocrine"),
    "chemo": ("chemo",),
}


def original_clinical_path() -> Path:
    for path in (
        PROCESSED_DIR / "01_Clinical" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv",
    ):
        if path.exists():
            return path
    raise FileNotFoundError("Original clinical.tsv not found.")


def family_start_table(clinical: pd.DataFrame, family: str) -> pd.DataFrame:
    id_col = "cases.submitter_id"
    type_col = "treatments.treatment_type"
    start_col = "treatments.days_to_treatment_start"
    required = {id_col, type_col, start_col}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Missing treatment timing columns: {sorted(missing)}")

    terms = FAMILY_TERMS[family]
    text = clinical[type_col].astype(str).str.lower()
    mask = np.zeros(len(clinical), dtype=bool)
    for term in terms:
        mask |= text.str.contains(term, na=False).to_numpy()

    temp = pd.DataFrame(
        {
            "patient_id_normalized": clinical.loc[mask, id_col].map(
                normalize_patient_id
            ),
            "treatment_start_day": pd.to_numeric(
                clinical.loc[mask, start_col], errors="coerce"
            ),
        }
    )
    grouped = (
        temp.groupby("patient_id_normalized", as_index=False)
        .agg(
            treatment_records=("treatment_start_day", "size"),
            start_records_nonmissing=("treatment_start_day", "count"),
            earliest_start_day=("treatment_start_day", "min"),
            median_start_day=("treatment_start_day", "median"),
            latest_start_day=("treatment_start_day", "max"),
        )
    )
    return grouped


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    weight_dir = DERIVED_DIR / "verified_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("STAGE 24 — CORRECT COMPACT+ERA PROPENSITY AND TREATMENT-TIMING GATE")
    print("=" * 110)

    clinical_path = original_clinical_path()
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    print(f"Original clinical source: {clinical_path}")
    print(f"Clinical rows × columns: {clinical.shape}")

    timing_rows = []
    balance_rows_all = []
    summary_rows = []

    for cohort_path in sorted(
        (DERIVED_DIR / "verified_cohorts").glob("*_verified.csv")
    ):
        cohort = cohort_key(cohort_path)
        df = read_table(cohort_path)
        compact, features = load_compact_with_year(cohort)
        assignments = get_repeat_assignments(cohort, repeat=1)
        ps, tuning = crossfit_propensity(
            compact, features, assignments, seed=2400
        )

        t = pd.to_numeric(
            compact["analysis_treatment"], errors="raise"
        ).astype(int).to_numpy()
        ow = np.where(t == 1, 1.0 - ps, ps)

        balance_rows = []
        for col in features:
            x = pd.to_numeric(compact[col], errors="coerce").to_numpy(float)
            balance_rows.append(
                {
                    "cohort": cohort,
                    "feature": col,
                    "smd_unweighted": weighted_smd(
                        x, t, np.ones(len(t))
                    ),
                    "smd_overlap": weighted_smd(x, t, ow),
                    "missing_fraction": float(np.mean(~np.isfinite(x))),
                }
            )
        balance = pd.DataFrame(balance_rows)
        balance["abs_smd_overlap"] = balance["smd_overlap"].abs()
        balance = balance.sort_values("abs_smd_overlap", ascending=False)
        balance.to_csv(
            table_dir / f"24_compact_era_balance_{cohort}.csv",
            index=False,
        )
        balance_rows_all.append(balance)

        weights = pd.DataFrame(
            {
                "patient_id_normalized": compact[
                    "patient_id_normalized"
                ],
                "analysis_treatment": t,
                "propensity_score_oof_compact_era": ps,
                "overlap_weight_compact_era": ow,
            }
        )
        weights.to_csv(
            weight_dir / f"24_compact_era_weights_{cohort}.csv",
            index=False,
        )
        tuning["cohort"] = cohort
        tuning.to_csv(
            table_dir / f"24_compact_era_tuning_{cohort}.csv",
            index=False,
        )

        max_smd = float(balance["abs_smd_overlap"].max())
        mean_smd = float(balance["abs_smd_overlap"].mean())
        ess_t = effective_sample_size(ow[t == 1])
        ess_c = effective_sample_size(ow[t == 0])

        if max_smd <= 0.10 and ess_t >= 50 and ess_c >= 50:
            status = "PRIMARY_BALANCE_READY"
        elif max_smd <= 0.15 and ess_t >= 40 and ess_c >= 25:
            status = "EXPLORATORY_BALANCE_READY"
        else:
            status = "NOT_BALANCE_READY"

        summary_rows.append(
            {
                "cohort": cohort,
                "n": len(compact),
                "treated": int(t.sum()),
                "control": int((t == 0).sum()),
                "events": int(
                    pd.to_numeric(
                        compact["analysis_event"], errors="raise"
                    ).sum()
                ),
                "n_compact_era_features": len(features),
                "max_abs_smd_overlap": max_smd,
                "mean_abs_smd_overlap": mean_smd,
                "ess_treated": ess_t,
                "ess_control": ess_c,
                "ps_min": float(ps.min()),
                "ps_p05": float(np.quantile(ps, 0.05)),
                "ps_median": float(np.median(ps)),
                "ps_p95": float(np.quantile(ps, 0.95)),
                "ps_max": float(ps.max()),
                "balance_status": status,
            }
        )

        print("\n" + "-" * 110)
        print(f"COHORT: {cohort}")
        print(
            f"n={len(compact)}, treated={int(t.sum())}, "
            f"controls={int((t == 0).sum())}, features={len(features)}"
        )
        print("\nPropensity tuning")
        print(tuning.to_string(index=False))
        print("\nTop residual balance after refitting compact model with diagnosis year")
        print(
            balance[
                ["feature", "smd_unweighted", "smd_overlap"]
            ].head(15).to_string(index=False)
        )
        print(
            f"\nmax |SMD|={max_smd:.4f}; mean |SMD|={mean_smd:.4f}; "
            f"ESS treated={ess_t:.1f}; ESS control={ess_c:.1f}; "
            f"status={status}"
        )

        short = None
        for suffix, family in FAMILY_FOR_COHORT.items():
            if cohort.endswith(suffix):
                short = family
                break
        if short is not None:
            starts = family_start_table(clinical, short)
            timing = df[
                ["patient_id_normalized", "analysis_treatment"]
            ].merge(
                starts,
                on="patient_id_normalized",
                how="left",
                validate="one_to_one",
            )
            treated = timing["analysis_treatment"].eq(1)
            controls = timing["analysis_treatment"].eq(0)
            valid_start = timing["earliest_start_day"].notna()
            row = {
                "cohort": cohort,
                "treatment_family": short,
                "n": len(timing),
                "treated": int(treated.sum()),
                "controls": int(controls.sum()),
                "treated_with_start": int((treated & valid_start).sum()),
                "treated_start_coverage": float(
                    (treated & valid_start).sum()
                    / max(1, treated.sum())
                ),
                "control_with_family_record": int(
                    (controls & timing["treatment_records"].notna()).sum()
                ),
                "treated_negative_start": int(
                    (treated & timing["earliest_start_day"].lt(0)).sum()
                ),
                "treated_start_0_90": int(
                    (
                        treated
                        & timing["earliest_start_day"].between(0, 90)
                    ).sum()
                ),
                "treated_start_0_180": int(
                    (
                        treated
                        & timing["earliest_start_day"].between(0, 180)
                    ).sum()
                ),
                "treated_start_0_365": int(
                    (
                        treated
                        & timing["earliest_start_day"].between(0, 365)
                    ).sum()
                ),
                "treated_start_gt365": int(
                    (treated & timing["earliest_start_day"].gt(365)).sum()
                ),
                "treated_start_extreme_abs_gt3650": int(
                    (
                        treated
                        & timing["earliest_start_day"].abs().gt(3650)
                    ).sum()
                ),
                "median_earliest_start_treated": float(
                    timing.loc[treated, "earliest_start_day"].median()
                ),
                "p05_earliest_start_treated": float(
                    timing.loc[treated, "earliest_start_day"].quantile(0.05)
                ),
                "p95_earliest_start_treated": float(
                    timing.loc[treated, "earliest_start_day"].quantile(0.95)
                ),
            }
            timing_rows.append(row)
            timing.to_csv(
                table_dir / f"24_treatment_start_patient_{cohort}.csv",
                index=False,
            )
            print("\nTreatment timing gate")
            print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(summary_rows).sort_values("cohort")
    timing_summary = pd.DataFrame(timing_rows).sort_values("cohort")
    summary.to_csv(
        table_dir / "24_compact_era_propensity_summary.csv", index=False
    )
    timing_summary.to_csv(
        table_dir / "24_treatment_timing_gate_summary.csv", index=False
    )
    pd.concat(balance_rows_all, ignore_index=True).to_csv(
        table_dir / "24_all_compact_era_balance.csv", index=False
    )

    print("\n" + "=" * 110)
    print("FINAL STAGE 24 BALANCE SUMMARY")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("\nFINAL STAGE 24 TIMING GATE")
    print(timing_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
