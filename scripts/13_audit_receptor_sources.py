from __future__ import annotations

import json
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
)


RECEPTOR_PATTERNS = {
    "ER": [
        re.compile(r"(^|[^a-z])er[_\s.-]*status([^a-z]|$)", re.I),
        re.compile(r"estrogen[_\s.-]*receptor", re.I),
        re.compile(r"breast[_\s.-]*carcinoma[_\s.-]*estrogen", re.I),
    ],
    "PR": [
        re.compile(r"(^|[^a-z])pr[_\s.-]*status([^a-z]|$)", re.I),
        re.compile(r"progesterone[_\s.-]*receptor", re.I),
        re.compile(r"breast[_\s.-]*carcinoma[_\s.-]*progesterone", re.I),
    ],
    "HER2": [
        re.compile(r"her2", re.I),
        re.compile(r"her[_\s.-]*2", re.I),
        re.compile(r"erbb2", re.I),
    ],
}

EXCLUDE_DIR_TOKENS = {
    "statistical_filtered",
    "mb_results",
    "merge",
    "merge_continuous_outer",
    "paper1_panels",
}


def read_header(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, nrows=10, low_memory=False)


def read_selected(path: Path, usecols: list[str]) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, usecols=usecols, low_memory=False)


def receptor_role(column: str) -> str | None:
    for role, patterns in RECEPTOR_PATTERNS.items():
        if any(pattern.search(column) for pattern in patterns):
            return role
    return None


def classify_source(path: Path, series: pd.Series) -> str:
    name = str(path).lower()
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(numeric.notna().mean())
    unique_numeric = set(numeric.dropna().unique().tolist())
    text = series.astype(str).str.strip().str.lower()
    posneg = text.isin(
        {
            "positive", "negative", "pos", "neg", "yes", "no",
            "present", "absent", "1", "0", "1.0", "0.0",
        }
    ).mean()

    if "raw" in name and posneg >= 0.8:
        return "preferred_raw_textual"
    if unique_numeric and unique_numeric.issubset({0.0, 1.0}) and numeric_fraction >= 0.8:
        return "binary_numeric"
    if any(token in name for token in ("preprocessed", "ite_ready", "master_")):
        return "processed_or_model_ready"
    if posneg >= 0.8:
        return "textual_posneg"
    if numeric_fraction >= 0.8:
        return "continuous_numeric_review"
    return "categorical_review"


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    manifest_dir = DERIVED_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for path in PROCESSED_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts & EXCLUDE_DIR_TOKENS:
            continue
        paths.append(path)

    rows = []
    errors = []
    for path in sorted(paths):
        try:
            header = read_header(path)
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        candidate_cols = [
            str(col) for col in header.columns
            if receptor_role(str(col)) is not None
        ]
        if not candidate_cols:
            continue

        id_col = detect_id_column(header)
        usecols = candidate_cols + ([id_col] if id_col and id_col not in candidate_cols else [])
        try:
            df = read_selected(path, usecols)
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        for col in candidate_cols:
            series = df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            text_counts = (
                series.astype(str)
                .str.strip()
                .value_counts(dropna=False)
                .head(15)
                .to_dict()
            )
            unique_numeric = sorted(set(numeric.dropna().unique().tolist()))
            exact_binary = bool(
                unique_numeric and set(unique_numeric).issubset({0.0, 1.0})
            )
            rows.append(
                {
                    "receptor": receptor_role(col),
                    "path": str(path),
                    "filename": path.name,
                    "column": col,
                    "patient_id_column": id_col or "",
                    "rows": len(df),
                    "nonmissing": int(series.notna().sum()),
                    "n_unique": int(series.nunique(dropna=True)),
                    "numeric_fraction": float(numeric.notna().mean()),
                    "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                    "numeric_q01": float(numeric.quantile(0.01)) if numeric.notna().any() else np.nan,
                    "numeric_median": float(numeric.median()) if numeric.notna().any() else np.nan,
                    "numeric_q99": float(numeric.quantile(0.99)) if numeric.notna().any() else np.nan,
                    "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
                    "exact_binary_0_1": int(exact_binary),
                    "source_class": classify_source(path, series),
                    "top_value_counts_json": json.dumps(text_counts, ensure_ascii=False),
                }
            )

    audit = pd.DataFrame(rows)
    if audit.empty:
        audit = pd.DataFrame(
            columns=[
                "receptor", "path", "filename", "column", "source_class"
            ]
        )
    else:
        priority = {
            "preferred_raw_textual": 0,
            "binary_numeric": 1,
            "textual_posneg": 2,
            "categorical_review": 3,
            "continuous_numeric_review": 4,
            "processed_or_model_ready": 5,
        }
        audit["priority"] = audit["source_class"].map(priority).fillna(99)
        audit = audit.sort_values(["receptor", "priority", "path", "column"])

    audit.to_csv(table_dir / "13_receptor_source_audit.csv", index=False)
    pd.DataFrame(errors).to_csv(
        table_dir / "13_receptor_source_audit_errors.csv", index=False
    )

    # Explicitly summarize the current ITE receptor fields.
    current = audit[
        audit["filename"].str.contains("ite_ready", case=False, na=False)
    ].copy()
    current.to_csv(
        table_dir / "13_current_ite_receptor_fields.csv", index=False
    )

    summary = []
    for receptor in ("ER", "PR", "HER2"):
        subset = audit[audit["receptor"] == receptor]
        preferred = subset[
            subset["source_class"].isin(
                ["preferred_raw_textual", "binary_numeric", "textual_posneg"]
            )
        ]
        current_subset = current[current["receptor"] == receptor]
        summary.append(
            {
                "receptor": receptor,
                "candidate_fields": len(subset),
                "preferred_raw_or_binary_fields": len(preferred),
                "current_ite_fields": len(current_subset),
                "current_ite_all_exact_binary": int(
                    len(current_subset) > 0
                    and current_subset["exact_binary_0_1"].eq(1).all()
                ),
                "status": (
                    "RAW_OR_BINARY_SOURCE_FOUND"
                    if len(preferred) > 0
                    else "NO_VERIFIED_RAW_SOURCE"
                ),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        table_dir / "13_receptor_source_summary.csv", index=False
    )

    print("\nReceptor source summary:")
    print(summary_df.to_string(index=False))
    print(
        "\nDo not rebuild subgroups automatically yet. Review "
        "13_receptor_source_audit.csv and choose authoritative fields."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
