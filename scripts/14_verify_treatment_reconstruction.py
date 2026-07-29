from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    PROCESSED_DIR,
    RESULTS_DIR,
    ensure_dirs,
    detect_id_column,
    normalize_patient_id,
    find_ite_file,
    read_table,
)


PREFERRED_TREATMENT_COLUMNS = [
    "treatment_type.treatments.diagnoses",
    "treatments.treatment_type",
    "treatment_type",
]

FAMILY_TERMS = {
    "hormone": [
        "hormone therapy", "endocrine", "tamoxifen", "letrozole",
        "anastrozole", "exemestane", "aromatase",
    ],
    "chemo": ["chemotherapy", "chemo", "cytotoxic"],
    "targeted": [
        "targeted molecular therapy", "targeted", "trastuzumab",
        "herceptin", "pertuzumab", "lapatinib",
    ],
    "radiation": ["radiation therapy", "radiotherapy", "radiation"],
}


def read_any(path: Path, usecols=None) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, usecols=usecols, low_memory=False)


def flatten_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            text = " | ".join(map(str, parsed))
    except Exception:
        pass
    return text.lower()


def choose_treatment_column(columns: list[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in columns}
    for preferred in PREFERRED_TREATMENT_COLUMNS:
        if preferred.lower() in lookup:
            return lookup[preferred.lower()]
    candidates = [
        str(c) for c in columns
        if "treatment_type" in str(c).lower()
    ]
    return candidates[0] if candidates else None


def family_flags(text: str) -> dict[str, int]:
    return {
        family: int(any(term in text for term in terms))
        for family, terms in FAMILY_TERMS.items()
    }


def confusion(raw: pd.Series, derived: pd.Series) -> dict[str, float]:
    raw = raw.astype(int)
    derived = derived.astype(int)
    tp = int(((raw == 1) & (derived == 1)).sum())
    tn = int(((raw == 0) & (derived == 0)).sum())
    fp = int(((raw == 0) & (derived == 1)).sum())
    fn = int(((raw == 1) & (derived == 0)).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "agreement": float((tp + tn) / len(raw)) if len(raw) else np.nan,
        "sensitivity_vs_raw": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity_vs_raw": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    ite_path = find_ite_file()
    if ite_path is None:
        raise FileNotFoundError("ITE ready dataset not found.")
    ite = read_table(ite_path)
    ite_id = detect_id_column(ite)
    if ite_id is None:
        raise ValueError("Cannot identify patient ID in ITE dataset.")
    ite["patient_id_normalized"] = ite[ite_id].map(normalize_patient_id)

    candidates = []
    for path in PROCESSED_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        if any(x in str(path).lower() for x in ("statistical_filtered", "mb_results", "merge")):
            continue
        try:
            header = read_any(path, usecols=None).head(2)
        except Exception:
            continue
        treatment_col = choose_treatment_column(list(map(str, header.columns)))
        id_col = detect_id_column(header)
        if treatment_col and id_col:
            score = (
                0 if treatment_col.lower() == "treatment_type.treatments.diagnoses"
                else 1
            )
            candidates.append((score, path, id_col, treatment_col))

    if not candidates:
        raise FileNotFoundError(
            "No patient-level table with a treatment_type column was found."
        )
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    _, source_path, source_id, treatment_col = candidates[0]

    raw = read_any(source_path, usecols=[source_id, treatment_col])
    raw["patient_id_normalized"] = raw[source_id].map(normalize_patient_id)
    raw["treatment_text_normalized"] = raw[treatment_col].map(flatten_text)

    flag_frames = []
    for _, row in raw.iterrows():
        flags = family_flags(row["treatment_text_normalized"])
        flags["patient_id_normalized"] = row["patient_id_normalized"]
        flag_frames.append(flags)
    flags = pd.DataFrame(flag_frames)
    flags = (
        flags.groupby("patient_id_normalized", as_index=False)
        .max()
    )

    merged = ite.merge(
        flags,
        on="patient_id_normalized",
        how="left",
        validate="one_to_one",
    )
    for family in FAMILY_TERMS:
        merged[family] = merged[family].fillna(0).astype(int)

    mapping = {
        "hormone": "T_hormone",
        "chemo": "T_chemo",
        "targeted": "T_targeted",
        "radiation": "T_radiation",
    }
    rows = []
    for family, derived_col in mapping.items():
        if derived_col not in merged.columns:
            continue
        result = confusion(
            merged[family],
            pd.to_numeric(merged[derived_col], errors="coerce").fillna(0).astype(int),
        )
        rows.append(
            {
                "family": family,
                "raw_positive": int(merged[family].sum()),
                "derived_positive": int(
                    pd.to_numeric(merged[derived_col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .sum()
                ),
                "derived_column": derived_col,
                "source_path": str(source_path),
                "source_patient_id_column": source_id,
                "source_treatment_column": treatment_col,
                **result,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        table_dir / "14_treatment_reconstruction_comparison.csv",
        index=False,
    )
    merged[
        ["patient_id_normalized"]
        + list(FAMILY_TERMS.keys())
        + [c for c in mapping.values() if c in merged.columns]
    ].to_csv(
        table_dir / "14_patient_treatment_flag_comparison.csv",
        index=False,
    )

    examples = (
        raw[["treatment_text_normalized"]]
        .value_counts()
        .head(50)
        .reset_index(name="records")
    )
    examples.to_csv(
        table_dir / "14_treatment_text_examples.csv", index=False
    )

    # Explicit treatment timing search: only columns containing both treatment
    # semantics and time/start/end semantics qualify.
    timing_rows = []
    for path in PROCESSED_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            header = pd.read_csv(path, sep=sep, nrows=2, low_memory=False)
        except Exception:
            continue
        for col in map(str, header.columns):
            low = col.lower()
            if (
                any(tok in low for tok in ("treatment", "therapy", "drug", "regimen"))
                and any(tok in low for tok in ("days_to", "start", "end", "date"))
            ):
                timing_rows.append({"path": str(path), "column": col})
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(
        table_dir / "14_true_treatment_timing_fields.csv", index=False
    )

    print("\nTreatment reconstruction comparison:")
    print(comparison.to_string(index=False))
    print(
        f"\nTrue treatment timing fields found: {len(timing)}. "
        "Dates such as birth, follow-up, and sample collection are intentionally excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
