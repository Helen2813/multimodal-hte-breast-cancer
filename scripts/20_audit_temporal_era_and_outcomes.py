from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    PROCESSED_DIR,
    DERIVED_DIR,
    RESULTS_DIR,
    ensure_dirs,
    detect_id_column,
    normalize_patient_id,
    read_table,
)


TAU_DAYS = 1825.0
YEAR_TOKENS = (
    "year_of_diagnosis",
    "diagnosis_year",
    "year_of_initial_pathologic_diagnosis",
)
YEAR_EXCLUDE = (
    "birth",
    "death",
    "created",
    "updated",
    "follow",
    "collection",
)
TREATMENT_FAMILY_TERMS = {
    "hormone": ("hormone", "endocrine"),
    "chemo": ("chemo",),
    "targeted": ("targeted", "trastuz", "herceptin", "pertuz", "lapatinib"),
    "radiation": ("radiation", "radiotherapy"),
}


def read_delimited(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t" if path.suffix.lower() == ".tsv" else ",",
        nrows=nrows,
        low_memory=False,
    )


def find_original_clinical() -> Path:
    preferred = [
        PROCESSED_DIR / "01_Clinical" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(PROCESSED_DIR.rglob("clinical.tsv"))
    if not candidates:
        raise FileNotFoundError("Original clinical.tsv not found.")
    return candidates[0]


def patient_id_column(df: pd.DataFrame) -> str:
    detected = detect_id_column(df)
    if detected:
        return detected
    for candidate in ("cases.submitter_id", "case_submitter_id", "submitter_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Patient ID column could not be identified.")


def plausible_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    current_year = pd.Timestamp.today().year
    return numeric.where(numeric.between(1900, current_year))


def choose_year_field(df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    rows = []
    for col in map(str, df.columns):
        low = col.lower()
        if (
            any(token in low for token in YEAR_TOKENS)
            and not any(token in low for token in YEAR_EXCLUDE)
        ):
            values = plausible_year(df[col])
            rows.append(
                {
                    "column": col,
                    "nonmissing_plausible": int(values.notna().sum()),
                    "nonmissing_fraction": float(values.notna().mean()),
                    "minimum": float(values.min()) if values.notna().any() else np.nan,
                    "median": float(values.median()) if values.notna().any() else np.nan,
                    "maximum": float(values.max()) if values.notna().any() else np.nan,
                    "preferred_name_score": int(
                        low.endswith("year_of_diagnosis")
                        or low == "year_of_diagnosis"
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    if audit.empty:
        return None, audit
    audit = audit.sort_values(
        ["preferred_name_score", "nonmissing_plausible"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return str(audit.iloc[0]["column"]), audit


def aggregate_year_by_patient(
    df: pd.DataFrame, id_col: str, year_col: str
) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "patient_id_normalized": df[id_col].map(normalize_patient_id),
            "diagnosis_year": plausible_year(df[year_col]),
        }
    )
    conflicts = (
        temp.dropna()
        .groupby("patient_id_normalized")["diagnosis_year"]
        .nunique()
    )
    conflict_ids = conflicts[conflicts > 1].index
    if len(conflict_ids):
        print(
            f"\nWARNING: {len(conflict_ids)} patients have conflicting "
            "diagnosis years; median year will be used."
        )
    return (
        temp.groupby("patient_id_normalized", as_index=False)["diagnosis_year"]
        .median()
    )


def treatment_family(text: object) -> str:
    low = str(text).lower()
    matched = [
        family
        for family, terms in TREATMENT_FAMILY_TERMS.items()
        if any(term in low for term in terms)
    ]
    return "|".join(matched) if matched else "other_or_unknown"


def era_label(year: float) -> str:
    if not np.isfinite(year):
        return "unknown"
    if year < 2000:
        return "<2000"
    if year < 2005:
        return "2000-2004"
    if year < 2010:
        return "2005-2009"
    return ">=2010"


def event_audit(df: pd.DataFrame, cohort: str) -> tuple[dict[str, object], pd.DataFrame]:
    event = pd.to_numeric(df["analysis_event"], errors="coerce")
    time = pd.to_numeric(df["analysis_time"], errors="coerce")

    death_before_tau = event.eq(1) & time.le(TAU_DAYS)
    known_alive_tau = time.ge(TAU_DAYS)
    censored_before_tau = event.eq(0) & time.lt(TAU_DAYS)
    invalid = event.isna() | time.isna() | time.lt(0)

    status = pd.Series("invalid", index=df.index, dtype="object")
    status.loc[death_before_tau] = "death_by_5y"
    status.loc[known_alive_tau] = "alive_at_5y"
    status.loc[censored_before_tau] = "censored_before_5y"

    patient = pd.DataFrame(
        {
            "cohort": cohort,
            "patient_id_normalized": df["patient_id_normalized"],
            "analysis_event": event,
            "analysis_time": time,
            "five_year_status": status,
        }
    )

    known = status.isin(["death_by_5y", "alive_at_5y"])
    reconstructed = pd.Series(np.nan, index=df.index)
    reconstructed.loc[status.eq("death_by_5y")] = 1
    reconstructed.loc[status.eq("alive_at_5y")] = 0
    patient["five_year_event_reconstructed"] = reconstructed

    legacy_col = "Y" if "Y" in df.columns else (
        "Y_died_5yr" if "Y_died_5yr" in df.columns else None
    )
    mismatch = np.nan
    legacy_known = 0
    if legacy_col:
        legacy = pd.to_numeric(df[legacy_col], errors="coerce")
        comparable = known & legacy.notna()
        legacy_known = int(comparable.sum())
        mismatch = int(
            (legacy.loc[comparable].astype(int)
             != reconstructed.loc[comparable].astype(int)).sum()
        )
        patient["legacy_five_year_column"] = legacy_col
        patient["legacy_five_year_value"] = legacy

    row = {
        "cohort": cohort,
        "n": len(df),
        "events_total": int(event.eq(1).sum()),
        "death_by_5y": int(status.eq("death_by_5y").sum()),
        "alive_at_5y": int(status.eq("alive_at_5y").sum()),
        "censored_before_5y": int(status.eq("censored_before_5y").sum()),
        "invalid_or_missing": int(status.eq("invalid").sum()),
        "known_five_year_fraction": float(known.mean()),
        "median_followup_days": float(time.median()),
        "median_followup_years": float(time.median() / 365.25),
        "minimum_time": float(time.min()),
        "maximum_time": float(time.max()),
        "zero_time": int(time.eq(0).sum()),
        "negative_time": int(time.lt(0).sum()),
        "legacy_five_year_column": legacy_col or "",
        "legacy_comparable_patients": legacy_known,
        "legacy_mismatches": mismatch,
    }
    return row, patient


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    manifest_dir = DERIVED_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    clinical_path = find_original_clinical()
    clinical = read_delimited(clinical_path)
    id_col = patient_id_column(clinical)

    print("=" * 100)
    print("STAGE 20 — TEMPORAL, ERA, AND EVENT-DEFINITION AUDIT")
    print("=" * 100)
    print(f"Original clinical source: {clinical_path}")
    print(f"Clinical shape: {clinical.shape}")
    print(f"Patient ID column: {id_col}")
    print(
        f"Unique normalized patients: "
        f"{clinical[id_col].map(normalize_patient_id).nunique()}"
    )

    # Diagnosis-year audit.
    year_col, year_audit = choose_year_field(clinical)
    year_audit.to_csv(table_dir / "20_diagnosis_year_field_audit.csv", index=False)

    if year_col:
        years = aggregate_year_by_patient(clinical, id_col, year_col)
        years["diagnosis_era"] = years["diagnosis_year"].map(era_label)
        years.to_csv(
            manifest_dir / "20_patient_diagnosis_year.csv", index=False
        )
        print("\nDIAGNOSIS YEAR CANDIDATES")
        print(year_audit.to_string(index=False))
        print(f"\nSelected diagnosis-year field: {year_col}")
        print("\nPatient-level diagnosis-year distribution")
        print(years["diagnosis_year"].describe().to_string())
        print("\nDiagnosis-era counts")
        print(years["diagnosis_era"].value_counts(dropna=False).to_string())
    else:
        years = pd.DataFrame(
            columns=["patient_id_normalized", "diagnosis_year", "diagnosis_era"]
        )
        print("\nWARNING: no plausible diagnosis-year field was found.")

    # Exact treatment timing coverage.
    treatment_type_col = (
        "treatments.treatment_type"
        if "treatments.treatment_type" in clinical.columns
        else None
    )
    timing_cols = [
        str(col)
        for col in clinical.columns
        if str(col).lower().startswith("treatments.")
        and any(
            token in str(col).lower()
            for token in ("days_to", "start", "end", "date")
        )
        and "created_datetime" not in str(col).lower()
        and "updated_datetime" not in str(col).lower()
    ]

    timing_rows = []
    family_rows = []
    print("\nTRUE TREATMENT TIMING FIELDS")
    if not timing_cols:
        print("None found.")
    else:
        print("\n".join(f"  - {col}" for col in timing_cols))

    for col in timing_cols:
        values = pd.to_numeric(clinical[col], errors="coerce")
        row = {
            "column": col,
            "nonmissing_records": int(values.notna().sum()),
            "nonmissing_fraction_records": float(values.notna().mean()),
            "unique_patients_nonmissing": int(
                clinical.loc[values.notna(), id_col]
                .map(normalize_patient_id)
                .nunique()
            ),
            "negative_values": int(values.lt(0).sum()),
            "zero_values": int(values.eq(0).sum()),
            "minimum": float(values.min()) if values.notna().any() else np.nan,
            "median": float(values.median()) if values.notna().any() else np.nan,
            "maximum": float(values.max()) if values.notna().any() else np.nan,
        }
        timing_rows.append(row)

        if treatment_type_col:
            temp = pd.DataFrame(
                {
                    "patient_id_normalized": clinical[id_col].map(normalize_patient_id),
                    "family": clinical[treatment_type_col].map(treatment_family),
                    "value": values,
                }
            )
            for family, group in temp.groupby("family", dropna=False):
                family_rows.append(
                    {
                        "timing_column": col,
                        "treatment_family": family,
                        "records": len(group),
                        "patients": group["patient_id_normalized"].nunique(),
                        "patients_nonmissing": group.loc[
                            group["value"].notna(), "patient_id_normalized"
                        ].nunique(),
                        "nonmissing_fraction_patients": float(
                            group.loc[
                                group["value"].notna(),
                                "patient_id_normalized",
                            ].nunique()
                            / max(1, group["patient_id_normalized"].nunique())
                        ),
                        "median": float(group["value"].median())
                        if group["value"].notna().any()
                        else np.nan,
                    }
                )

    timing_df = pd.DataFrame(timing_rows)
    family_df = pd.DataFrame(family_rows)
    timing_df.to_csv(
        table_dir / "20_treatment_timing_coverage.csv", index=False
    )
    family_df.to_csv(
        table_dir / "20_treatment_timing_by_family.csv", index=False
    )
    if not timing_df.empty:
        print("\nTreatment timing coverage")
        print(timing_df.to_string(index=False))
    if not family_df.empty:
        print("\nTreatment timing coverage by family")
        print(family_df.to_string(index=False))

    # Cohort-level era and event audits.
    cohort_paths = sorted(
        (DERIVED_DIR / "verified_cohorts").glob("*_verified.csv")
    )
    if not cohort_paths:
        raise FileNotFoundError("Verified cohorts were not found.")

    era_rows = []
    event_rows = []
    event_frames = []
    for path in cohort_paths:
        cohort = path.stem.replace("_verified", "")
        df = read_table(path)
        print("\n" + "-" * 100)
        print(f"COHORT: {cohort}")
        print(f"Rows: {len(df)}")
        print(
            f"Treated: {int(pd.to_numeric(df['analysis_treatment']).sum())}; "
            f"Controls: {int((1 - pd.to_numeric(df['analysis_treatment'])).sum())}; "
            f"Events: {int(pd.to_numeric(df['analysis_event']).sum())}"
        )

        if not years.empty:
            merged = df.merge(
                years,
                on="patient_id_normalized",
                how="left",
                validate="one_to_one",
            )
            t = pd.to_numeric(
                merged["analysis_treatment"], errors="raise"
            ).astype(int)
            print("\nDiagnosis year by treatment arm")
            arm_summary = (
                merged.groupby(t)["diagnosis_year"]
                .agg(["count", "median", "min", "max"])
                .rename(index={0: "control", 1: "treated"})
            )
            print(arm_summary.to_string())
            print("\nEra × treatment counts")
            era_table = pd.crosstab(
                merged["diagnosis_era"].fillna("unknown"),
                t.map({0: "control", 1: "treated"}),
                margins=True,
            )
            print(era_table.to_string())

            for arm in (0, 1):
                group = merged.loc[t.eq(arm)]
                era_rows.append(
                    {
                        "cohort": cohort,
                        "arm": "treated" if arm == 1 else "control",
                        "n": len(group),
                        "diagnosis_year_nonmissing": int(
                            group["diagnosis_year"].notna().sum()
                        ),
                        "diagnosis_year_missing_fraction": float(
                            group["diagnosis_year"].isna().mean()
                        ),
                        "median_year": float(group["diagnosis_year"].median())
                        if group["diagnosis_year"].notna().any()
                        else np.nan,
                        "minimum_year": float(group["diagnosis_year"].min())
                        if group["diagnosis_year"].notna().any()
                        else np.nan,
                        "maximum_year": float(group["diagnosis_year"].max())
                        if group["diagnosis_year"].notna().any()
                        else np.nan,
                    }
                )

        event_row, patient_event = event_audit(df, cohort)
        event_rows.append(event_row)
        event_frames.append(patient_event)
        print("\nFive-year event-definition audit")
        print(pd.DataFrame([event_row]).to_string(index=False))

    era_summary = pd.DataFrame(era_rows)
    event_summary = pd.DataFrame(event_rows)
    era_summary.to_csv(
        table_dir / "20_cohort_diagnosis_era_summary.csv", index=False
    )
    event_summary.to_csv(
        table_dir / "20_event_definition_summary.csv", index=False
    )
    pd.concat(event_frames, ignore_index=True).to_csv(
        manifest_dir / "20_patient_five_year_status.csv", index=False
    )

    print("\n" + "=" * 100)
    print("STAGE 20 DECISION FLAGS")
    print("=" * 100)
    print(event_summary.to_string(index=False))

    if timing_df.empty or timing_df["unique_patients_nonmissing"].max() < 100:
        print(
            "\nDECISION: treatment-start timing is insufficient for a "
            "formal landmark or target-trial time-zero analysis."
        )
    else:
        best = timing_df.sort_values(
            "unique_patients_nonmissing", ascending=False
        ).iloc[0]
        print(
            "\nDECISION: at least one treatment-timing field has "
            f"{int(best['unique_patients_nonmissing'])} patients. "
            "A separate timed-treatment subset analysis may be feasible, "
            "but coverage by treatment family must be reviewed."
        )

    print(f"\nAll Stage 20 tables saved under: {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
