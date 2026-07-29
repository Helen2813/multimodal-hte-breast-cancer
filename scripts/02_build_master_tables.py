from __future__ import annotations

from pathlib import Path
import pandas as pd

from _common import (
    DERIVED_DIR, ensure_dirs, read_table, detect_id_column,
    add_normalized_patient_id, find_ite_file, find_merge_file,
    require_unique_ids, modality_counts, sha256_file, save_json,
)


def metadata_columns(df: pd.DataFrame) -> list[str]:
    exact = {
        "patient_id", "T", "T_hormone", "T_hormone_excl", "T_chemo",
        "T_targeted", "T_radiation", "Y", "Y_died_5yr", "OS",
        "OS.time", "OS_time", "ER_status", "PR_status", "HER2_status",
    }
    keep = []
    for column in df.columns:
        text = str(column)
        low = text.lower()
        if (
            text in exact
            or low.startswith("pathology_details.")
            or low.startswith("days_to_")
            or low in {"event", "status", "time", "survival_time"}
        ):
            keep.append(text)
    return keep


def prepare_source(path: Path, label: str, report_dir: Path):
    df = read_table(path)
    id_col = detect_id_column(df)
    if id_col is None:
        raise ValueError(f"{label}: patient ID column not detected in {path}")
    df = add_normalized_patient_id(df, id_col)
    if df["patient_id_normalized"].isna().any():
        bad = df[df["patient_id_normalized"].isna()]
        bad.to_csv(report_dir / f"missing_ids_{label}.csv", index=False)
        raise ValueError(f"{label}: missing patient IDs found")
    require_unique_ids(df, label, report_dir)
    return df, id_col


def join_one(merge_path: Path, ite_df: pd.DataFrame, representation: str, report_dir: Path, cohort_dir: Path):
    merge_df, merge_id_col = prepare_source(merge_path, f"merge_{representation}", report_dir)
    ite_cols = ["patient_id_normalized"] + [
        c for c in metadata_columns(ite_df) if c != "patient_id_normalized"
    ]
    ite_meta = ite_df[ite_cols].copy()

    merge_ids = set(merge_df["patient_id_normalized"])
    ite_ids = set(ite_meta["patient_id_normalized"])
    pd.DataFrame({"patient_id_normalized": sorted(merge_ids - ite_ids)}).to_csv(
        report_dir / f"unmatched_{representation}_merge_only.csv", index=False
    )
    pd.DataFrame({"patient_id_normalized": sorted(ite_ids - merge_ids)}).to_csv(
        report_dir / f"unmatched_{representation}_ite_only.csv", index=False
    )

    columns_to_add = [
        c for c in ite_meta.columns
        if c == "patient_id_normalized" or c not in merge_df.columns
    ]
    joined = merge_df.merge(
        ite_meta[columns_to_add],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )

    output = cohort_dir / f"master_{representation}.csv"
    joined.to_csv(output, index=False)

    manifest = {
        "representation": representation,
        "source_merge": str(merge_path),
        "source_merge_sha256": sha256_file(merge_path),
        "n_merge_rows": int(len(merge_df)),
        "n_ite_rows": int(len(ite_meta)),
        "n_joined_rows": int(len(joined)),
        "n_unmatched_merge": len(merge_ids - ite_ids),
        "n_unmatched_ite": len(ite_ids - merge_ids),
        "merge_id_column": merge_id_col,
        "modality_feature_counts": modality_counts(joined.columns),
        "output": str(output),
    }
    save_json(manifest, report_dir / f"join_manifest_{representation}.json")
    return manifest


def main() -> int:
    ensure_dirs()
    report_dir = DERIVED_DIR / "audits"
    cohort_dir = DERIVED_DIR / "cohorts"

    ite_path = find_ite_file()
    if ite_path is None:
        raise FileNotFoundError("ITE dataset not found in data/processed/output")
    ite_df, _ = prepare_source(ite_path, "ite", report_dir)

    sources = {
        "complete_case": find_merge_file("MERGE"),
        "outer": find_merge_file("MERGE_continuous_outer"),
    }
    missing = [name for name, path in sources.items() if path is None]
    if missing:
        raise FileNotFoundError(f"Missing merge representations: {missing}")

    manifests = []
    for representation, path in sources.items():
        print(f"Building master table: {representation}")
        manifests.append(join_one(path, ite_df, representation, report_dir, cohort_dir))

    summary = pd.DataFrame(manifests)
    summary.to_csv(report_dir / "02_join_summary.csv", index=False)
    print("\nJoin summary:")
    print(summary[[
        "representation", "n_merge_rows", "n_ite_rows", "n_joined_rows",
        "n_unmatched_merge", "n_unmatched_ite",
    ]].to_string(index=False))
    print(f"\nMaster tables saved to: {cohort_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
