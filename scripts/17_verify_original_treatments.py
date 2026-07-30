from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    PROCESSED_DIR,
    DERIVED_DIR,
    RESULTS_DIR,
    ensure_dirs,
    find_ite_file,
    detect_id_column,
    normalize_patient_id,
    read_table,
)


REQUIRED_COLUMNS = {
    "cases.submitter_id",
    "treatments.treatment_type",
}
OPTIONAL_AGENT = "treatments.therapeutic_agents"


def read_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)


def find_original_treatment_file() -> Path | None:
    preferred = [
        PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "original_clinical.tsv",
    ]
    for path in preferred:
        if path.exists():
            try:
                header = read_tsv(path, nrows=2)
                if REQUIRED_COLUMNS.issubset(set(header.columns)):
                    return path
            except Exception:
                pass

    for path in PROCESSED_DIR.rglob("clinical.tsv"):
        try:
            header = read_tsv(path, nrows=2)
        except Exception:
            continue
        if REQUIRED_COLUMNS.issubset(set(header.columns)):
            return path
    return None


def patient_set(table: pd.DataFrame, keyword: str) -> set[str]:
    mask = (
        table["treatments.treatment_type"]
        .astype(str)
        .str.lower()
        .str.contains(keyword, na=False)
    )
    return set(table.loc[mask, "patient_id_normalized"])


def compare(raw: pd.Series, current: pd.Series) -> dict[str, object]:
    raw = raw.astype(int)
    current = current.astype(int)
    tp = int(((raw == 1) & (current == 1)).sum())
    tn = int(((raw == 0) & (current == 0)).sum())
    fp = int(((raw == 0) & (current == 1)).sum())
    fn = int(((raw == 1) & (current == 0)).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "agreement": float((tp + tn) / len(raw)),
        "sensitivity_vs_original": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity_vs_original": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def main() -> int:
    ensure_dirs()
    verified_dir = DERIVED_DIR / "verified_sources"
    verified_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    clinical_path = find_original_treatment_file()
    if clinical_path is None:
        expected = (
            PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv"
        )
        message = (
            "\nORIGINAL TREATMENT FILE NOT FOUND.\n\n"
            "Copy this file from the old project:\n"
            "  Thesis_v3\\data\\drags\\clinical.tsv\n\n"
            "to:\n"
            f"  {expected}\n\n"
            "The required columns are:\n"
            "  cases.submitter_id\n"
            "  treatments.treatment_type\n"
            "  treatments.therapeutic_agents (optional for validation)\n"
        )
        print(message)
        (table_dir / "17_MISSING_ORIGINAL_TREATMENT_FILE.txt").write_text(
            message, encoding="utf-8"
        )
        return 2

    clinical = read_tsv(clinical_path)
    missing = REQUIRED_COLUMNS - set(clinical.columns)
    if missing:
        raise ValueError(f"Treatment file missing columns: {sorted(missing)}")

    clinical["patient_id_normalized"] = (
        clinical["cases.submitter_id"].map(normalize_patient_id)
    )
    treatment = clinical[
        ["patient_id_normalized", "treatments.treatment_type"]
        + ([OPTIONAL_AGENT] if OPTIONAL_AGENT in clinical.columns else [])
    ].copy()

    sets = {
        "hormone": patient_set(treatment, "hormone"),
        "chemo": patient_set(treatment, "chemo"),
        "targeted": patient_set(treatment, "targeted"),
        "radiation": patient_set(treatment, "radiation"),
    }
    sets["hormone_excl"] = (
        sets["hormone"] - sets["chemo"] - sets["targeted"]
    )

    ite_path = find_ite_file()
    if ite_path is None:
        raise FileNotFoundError("ITE ready dataset was not found.")
    ite = read_table(ite_path)
    ite_id = detect_id_column(ite)
    if ite_id is None:
        raise ValueError("Patient ID column not found in ITE table.")

    verified = pd.DataFrame(
        {"patient_id_normalized": ite[ite_id].map(normalize_patient_id)}
    )
    for family, patients in sets.items():
        verified[f"T_{family}_verified"] = (
            verified["patient_id_normalized"].isin(patients).astype(int)
        )

    mapping = {
        "hormone": "T_hormone",
        "chemo": "T_chemo",
        "targeted": "T_targeted",
        "radiation": "T_radiation",
        "hormone_excl": "T_hormone_excl",
    }
    comparison_rows = []
    for family, old_col in mapping.items():
        verified_col = f"T_{family}_verified"
        row = {
            "family": family,
            "original_source_path": str(clinical_path),
            "original_positive": int(verified[verified_col].sum()),
            "legacy_column": old_col,
        }
        if old_col in ite.columns:
            old = pd.to_numeric(ite[old_col], errors="coerce").fillna(0).astype(int)
            row["legacy_positive"] = int(old.sum())
            row.update(compare(verified[verified_col], old))
        else:
            row["legacy_positive"] = np.nan
        comparison_rows.append(row)

    verified.to_csv(
        verified_dir / "17_verified_treatment_flags.csv",
        index=False,
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        table_dir / "17_treatment_verification_summary.csv",
        index=False,
    )

    type_counts = (
        treatment["treatments.treatment_type"]
        .astype(str)
        .str.strip()
        .value_counts(dropna=False)
        .reset_index()
    )
    type_counts.columns = ["treatment_type", "records"]
    type_counts.to_csv(
        table_dir / "17_original_treatment_type_counts.csv",
        index=False,
    )

    if OPTIONAL_AGENT in treatment.columns:
        agents = (
            treatment[OPTIONAL_AGENT]
            .astype(str)
            .str.strip()
            .value_counts(dropna=False)
            .head(200)
            .reset_index()
        )
        agents.columns = ["therapeutic_agents", "records"]
        agents.to_csv(
            table_dir / "17_therapeutic_agent_counts.csv",
            index=False,
        )

    timing_cols = [
        str(col)
        for col in clinical.columns
        if (
            str(col).lower().startswith("treatments.")
            and any(
                token in str(col).lower()
                for token in ("days_to", "start", "end", "date")
            )
            and "created_datetime" not in str(col).lower()
            and "updated_datetime" not in str(col).lower()
        )
    ]
    pd.DataFrame({"treatment_timing_column": timing_cols}).to_csv(
        table_dir / "17_true_treatment_timing_columns.csv",
        index=False,
    )

    print(f"\nVerified original treatment file: {clinical_path}")
    print(comparison.to_string(index=False))
    print(f"\nTrue treatment timing columns: {len(timing_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
