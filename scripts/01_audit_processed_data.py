from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from _common import (
    MODALITY_FOLDERS, PROCESSED_DIR, DERIVED_DIR, ensure_dirs,
    read_table, detect_id_column, add_normalized_patient_id,
    modality_counts, treatment_columns, outcome_columns, time_columns,
    find_ite_file, write_markdown,
)


def audit_matrix(path: Path, label: str):
    df = read_table(path)
    id_col = detect_id_column(df)
    unique_ids = np.nan
    duplicate_ids = np.nan
    if id_col:
        tmp = add_normalized_patient_id(df, id_col)
        unique_ids = int(tmp["patient_id_normalized"].nunique(dropna=True))
        duplicate_ids = int(tmp["patient_id_normalized"].duplicated().sum())

    counts = modality_counts(df.columns)
    row = {
        "label": label,
        "path": str(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "id_column": id_col or "",
        "unique_patient_ids": unique_ids,
        "duplicate_patient_ids": duplicate_ids,
        "overall_missing_fraction": float(df.isna().mean().mean()),
        "treatment_columns": "|".join(treatment_columns(df)),
        "outcome_columns": "|".join(outcome_columns(df)),
        "time_columns": "|".join(time_columns(df)),
        **{f"n_{k}": v for k, v in counts.items()},
    }
    columns = pd.DataFrame({
        "matrix": label,
        "column": [str(c) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_fraction": [float(df[c].isna().mean()) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return row, columns


def main() -> int:
    ensure_dirs()
    out_dir = DERIVED_DIR / "audits"
    targets: list[tuple[str, Path]] = []

    ite = find_ite_file()
    if ite:
        targets.append(("ite_ready", ite))

    for folder_name, representation in (("MERGE", "complete_case"), ("MERGE_continuous_outer", "outer")):
        folder = PROCESSED_DIR / folder_name
        if folder.exists():
            for path in sorted(folder.glob("*.csv")):
                targets.append((f"{representation}:{path.stem}", path))

    matrix_rows = []
    column_frames = []
    for label, path in targets:
        print(f"Auditing {label}: {path.name}")
        row, cols = audit_matrix(path, label)
        matrix_rows.append(row)
        column_frames.append(cols)

    matrix_df = pd.DataFrame(matrix_rows)
    columns_df = pd.concat(column_frames, ignore_index=True)
    matrix_df.to_csv(out_dir / "01_matrix_audit.csv", index=False)
    columns_df.to_csv(out_dir / "01_column_inventory.csv", index=False)

    modality_rows = []
    for modality, folder_name in MODALITY_FOLDERS.items():
        folder = PROCESSED_DIR / folder_name
        summaries = sorted(folder.rglob("summary_all_results.csv")) if folder.exists() else []
        candidate_files = sorted((folder / "statistical_filtered").glob("*.csv")) if (folder / "statistical_filtered").exists() else []
        feature_lists = []
        if (folder / "mb_results").exists():
            feature_lists = list((folder / "mb_results").rglob("*_genes.txt")) + list((folder / "mb_results").rglob("*_features.txt"))
        modality_rows.append({
            "modality": modality,
            "folder": str(folder),
            "n_summary_files": len(summaries),
            "summary_paths": "|".join(str(x) for x in summaries),
            "n_candidate_csv": len(candidate_files),
            "n_feature_lists": len(feature_lists),
        })
    modality_df = pd.DataFrame(modality_rows)
    modality_df.to_csv(out_dir / "01_modality_inventory.csv", index=False)

    lines = ["# Processed-data audit", "", f"- Matrices audited: **{len(matrix_df)}**", "", "## Matrices", ""]
    for row in matrix_rows:
        lines.append(
            f"- `{row['label']}`: {row['rows']} × {row['columns']}; "
            f"ID=`{row['id_column']}`; duplicates={row['duplicate_patient_ids']}; "
            f"missing={row['overall_missing_fraction']:.4f}"
        )
    lines += ["", "## Modality folders", ""]
    for row in modality_rows:
        lines.append(
            f"- `{row['modality']}`: summaries={row['n_summary_files']}, "
            f"candidate CSVs={row['n_candidate_csv']}, feature lists={row['n_feature_lists']}"
        )
    write_markdown(lines, out_dir / "01_audit_report.md")

    print("\nMatrix audit:")
    print(matrix_df.to_string(index=False))
    print(f"\nSaved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
