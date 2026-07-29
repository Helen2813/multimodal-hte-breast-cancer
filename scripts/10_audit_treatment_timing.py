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
)


TIMING_TOKENS = ("days_to", "treatment_start", "treatment_end", "start_day", "end_day")
TREATMENT_TOKENS = ("treatment", "therapy", "regimen", "drug", "pharmaceutical")


def robust_read(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def find_patient_column(df: pd.DataFrame) -> str | None:
    detected = detect_id_column(df)
    if detected:
        return detected
    for col in df.columns:
        low = str(col).lower()
        if "submitter_id" in low or "patient" in low or "barcode" in low:
            sample = df[col].dropna().astype(str).head(100)
            if len(sample) and sample.str.contains(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}", regex=True).mean() >= 0.5:
                return str(col)
    return None


def classify_family(text: object) -> str:
    value = str(text).strip().lower()
    hormone = ("hormone", "endocrine", "tamox", "letroz", "anastro", "exemest", "aromatase")
    chemo = ("chemotherapy", "chemo", "cytotoxic")
    targeted = ("targeted", "trastuz", "herceptin", "pertuz", "lapatinib", "anti-her2")
    if any(x in value for x in hormone):
        return "hormone"
    if any(x in value for x in targeted):
        return "targeted"
    if any(x in value for x in chemo):
        return "chemo"
    return "other_or_unknown"


def main() -> int:
    ensure_dirs()
    clinical_root = PROCESSED_DIR / "01_Clinical"
    table_dir = RESULTS_DIR / "tables"
    manifest_dir = DERIVED_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        [p for p in clinical_root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}]
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV/TSV files found under {clinical_root}")

    file_rows = []
    column_rows = []
    family_rows = []
    patient_timing_frames = []

    for path in candidates:
        try:
            header = robust_read(path, nrows=20)
        except Exception as exc:
            file_rows.append({"path": str(path), "readable": 0, "error": str(exc)})
            continue

        relevant = [
            str(c) for c in header.columns
            if any(tok in str(c).lower() for tok in TIMING_TOKENS + TREATMENT_TOKENS)
        ]
        if not relevant:
            continue

        try:
            df = robust_read(path)
        except Exception as exc:
            file_rows.append({"path": str(path), "readable": 0, "error": str(exc)})
            continue

        patient_col = find_patient_column(df)
        timing_cols = [
            str(c) for c in df.columns
            if any(tok in str(c).lower() for tok in TIMING_TOKENS)
        ]
        treatment_cols = [
            str(c) for c in df.columns
            if any(tok in str(c).lower() for tok in TREATMENT_TOKENS)
        ]

        file_rows.append(
            {
                "path": str(path),
                "readable": 1,
                "rows": len(df),
                "columns": len(df.columns),
                "patient_id_column": patient_col or "",
                "n_timing_columns": len(timing_cols),
                "timing_columns": "|".join(timing_cols),
                "n_treatment_columns": len(treatment_cols),
                "treatment_columns": "|".join(treatment_cols),
            }
        )

        for col in timing_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            column_rows.append(
                {
                    "path": str(path),
                    "column": col,
                    "nonmissing": int(numeric.notna().sum()),
                    "nonmissing_fraction": float(numeric.notna().mean()),
                    "negative_values": int((numeric < 0).sum()),
                    "zero_values": int((numeric == 0).sum()),
                    "minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
                    "median": float(numeric.median()) if numeric.notna().any() else np.nan,
                    "maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
                }
            )

        if patient_col and treatment_cols:
            text_col = max(
                treatment_cols,
                key=lambda c: df[c].astype(str).str.len().mean()
                if c in df.columns else -1,
            )
            temp = pd.DataFrame(
                {
                    "patient_id_normalized": df[patient_col].map(normalize_patient_id),
                    "treatment_text": df[text_col],
                }
            )
            temp["treatment_family"] = temp["treatment_text"].map(classify_family)
            for col in timing_cols:
                temp[col] = pd.to_numeric(df[col], errors="coerce")
            patient_timing_frames.append(temp)

            grouped = (
                temp.groupby("treatment_family", dropna=False)
                .agg(
                    records=("patient_id_normalized", "size"),
                    patients=("patient_id_normalized", "nunique"),
                )
                .reset_index()
            )
            grouped["path"] = str(path)
            grouped["treatment_text_column"] = text_col
            family_rows.extend(grouped.to_dict("records"))

    pd.DataFrame(file_rows).to_csv(table_dir / "10_treatment_file_audit.csv", index=False)
    pd.DataFrame(column_rows).to_csv(table_dir / "10_treatment_timing_columns.csv", index=False)
    pd.DataFrame(family_rows).to_csv(table_dir / "10_treatment_family_counts.csv", index=False)

    if patient_timing_frames:
        patient_timing = pd.concat(patient_timing_frames, ignore_index=True)
        patient_timing.to_csv(
            manifest_dir / "10_patient_treatment_timing_records.csv", index=False
        )

        summary_rows = []
        timing_cols = [
            c for c in patient_timing.columns
            if c not in {"patient_id_normalized", "treatment_text", "treatment_family"}
        ]
        for family, group in patient_timing.groupby("treatment_family"):
            row = {
                "treatment_family": family,
                "records": len(group),
                "patients": group["patient_id_normalized"].nunique(),
            }
            for col in timing_cols:
                row[f"{col}_patients_nonmissing"] = group.loc[
                    pd.to_numeric(group[col], errors="coerce").notna(),
                    "patient_id_normalized",
                ].nunique()
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            table_dir / "10_treatment_timing_by_family.csv", index=False
        )

    print("\nTreatment timing audit complete.")
    print(f"Files inspected with relevant fields: {len(file_rows)}")
    print(f"Timing columns summarized: {len(column_rows)}")
    print(f"Outputs saved to: {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
