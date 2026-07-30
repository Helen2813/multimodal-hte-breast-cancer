from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    PROCESSED_DIR,
    DERIVED_DIR,
    RESULTS_DIR,
    normalize_patient_id,
    read_table,
)


COHORT = "outer_hormone_hrpos_her2neg"
LANDMARK = 180
HORIZON = 730


def primary_paths() -> dict[str, Path]:
    key = f"{COHORT}_landmark{LANDMARK}"
    return {
        "source_cohort": DERIVED_DIR / "verified_cohorts" / f"{COHORT}_verified.csv",
        "source_compact": DERIVED_DIR / "verified_compact_adjustment" / f"{COHORT}_compact_verified.csv",
        "landmark_cohort": DERIVED_DIR / "landmark_cohorts" / f"{key}.csv",
        "landmark_compact": DERIVED_DIR / "landmark_compact" / f"{key}_compact.csv",
        "landmark_weights": DERIVED_DIR / "landmark_weights" / f"{key}_weights.csv",
        "landmark_splits": DERIVED_DIR / "landmark_splits" / f"{key}_splits.csv",
        "landmark_summary": RESULTS_DIR / "tables" / "29_landmark_cohort_summary.csv",
        "landmark_balance": RESULTS_DIR / "tables" / "30_landmark_balance_summary.csv",
        "paperA_candidate": RESULTS_DIR / "tables" / "34_paperA_candidate_summary.csv",
        "diagnosis_year": DERIVED_DIR / "manifests" / "20_patient_diagnosis_year.csv",
    }


def original_clinical_path() -> Path:
    for path in (
        PROCESSED_DIR / "01_Clinical" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv",
    ):
        if path.exists():
            return path
    raise FileNotFoundError("Original clinical.tsv not found.")


def assert_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 11 inputs:\n" + "\n".join(missing))


def feature_columns(compact: pd.DataFrame) -> list[str]:
    columns = [c for c in compact.columns if c.startswith("W_")]
    for col in ("diagnosis_year", "diagnosis_year_missing"):
        if col in compact.columns:
            columns.append(col)
    return list(dict.fromkeys(columns))


def assemble_landmark_table() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    paths = primary_paths()
    assert_paths({k: paths[k] for k in ("landmark_cohort", "landmark_compact", "landmark_weights", "landmark_splits")})
    cohort = read_table(paths["landmark_cohort"])
    compact = read_table(paths["landmark_compact"])
    weights = read_table(paths["landmark_weights"])
    splits = read_table(paths["landmark_splits"])

    id_col = "patient_id_normalized"
    W_cols = feature_columns(compact)
    if not W_cols:
        raise ValueError("No compact adjustment features found.")

    cohort_side = cohort.drop(columns=[c for c in W_cols if c in cohort.columns], errors="ignore")
    compact_side = compact[[id_col] + W_cols].copy()
    required_weights = [id_col, "propensity_score_oof", "overlap_weight"]
    missing_weights = set(required_weights) - set(weights.columns)
    if missing_weights:
        raise ValueError(f"Landmark weights missing: {sorted(missing_weights)}")

    table = (
        cohort_side.merge(compact_side, on=id_col, how="inner", validate="one_to_one")
        .merge(weights[required_weights], on=id_col, how="inner", validate="one_to_one")
    )
    suffix_cols = [c for c in table.columns if c.endswith("_x") or c.endswith("_y")]
    if suffix_cols:
        raise ValueError(f"Unexpected merge suffix columns: {suffix_cols}")
    if len(table) != len(cohort):
        raise ValueError(f"Landmark assembly lost rows: {len(cohort)} -> {len(table)}")

    split_ids = set(splits.loc[splits["repeat"] == 1, id_col].astype(str))
    table_ids = set(table[id_col].astype(str))
    if split_ids != table_ids:
        raise ValueError("Landmark split patient IDs do not match.")

    for col in ("analysis_treatment", "analysis_event", "analysis_time", "overlap_weight", "later_initiator"):
        if col not in table.columns:
            raise ValueError(f"Landmark table missing {col}")
    _ = table[W_cols].apply(pd.to_numeric, errors="coerce")

    metadata = {
        "cohort": COHORT,
        "landmark_day": LANDMARK,
        "n": len(table),
        "treated": int(pd.to_numeric(table["analysis_treatment"], errors="raise").sum()),
        "control": int((1 - pd.to_numeric(table["analysis_treatment"], errors="raise")).sum()),
        "events": int(pd.to_numeric(table["analysis_event"], errors="raise").sum()),
        "compact_features": len(W_cols),
        "merge_suffix_columns": 0,
    }
    return table, W_cols, metadata


def assemble_source_table() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    paths = primary_paths()
    assert_paths({k: paths[k] for k in ("source_cohort", "source_compact", "diagnosis_year")})
    cohort = read_table(paths["source_cohort"])
    compact = read_table(paths["source_compact"])
    years = read_table(paths["diagnosis_year"])[["patient_id_normalized", "diagnosis_year"]]

    if "diagnosis_year" not in compact.columns:
        compact = compact.merge(years, on="patient_id_normalized", how="left", validate="one_to_one")
    if "diagnosis_year_missing" not in compact.columns:
        compact["diagnosis_year_missing"] = compact["diagnosis_year"].isna().astype(float)

    W_cols = feature_columns(compact)
    cohort_side = cohort.drop(columns=[c for c in W_cols if c in cohort.columns], errors="ignore")
    source = cohort_side.merge(
        compact[["patient_id_normalized"] + W_cols],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    if len(source) != len(cohort):
        raise ValueError("Source table assembly lost patients.")

    metadata = {
        "source_n": len(source),
        "source_treated_ever": int(pd.to_numeric(source["analysis_treatment"], errors="raise").sum()),
        "source_controls_ever": int((1 - pd.to_numeric(source["analysis_treatment"], errors="raise")).sum()),
        "compact_features": len(W_cols),
    }
    return source, W_cols, metadata


def extract_hormone_start_days(patient_ids: pd.Series) -> pd.DataFrame:
    clinical = pd.read_csv(original_clinical_path(), sep="\t", low_memory=False)
    required = {
        "cases.submitter_id",
        "treatments.treatment_type",
        "treatments.days_to_treatment_start",
    }
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Clinical treatment timing missing: {sorted(missing)}")

    text = clinical["treatments.treatment_type"].astype(str).str.lower()
    mask = text.str.contains("hormone|endocrine", regex=True, na=False)
    temp = pd.DataFrame(
        {
            "patient_id_normalized": clinical.loc[mask, "cases.submitter_id"].map(normalize_patient_id),
            "start_day": pd.to_numeric(
                clinical.loc[mask, "treatments.days_to_treatment_start"],
                errors="coerce",
            ),
        }
    )
    starts = (
        temp.groupby("patient_id_normalized", as_index=False)
        .agg(
            hormone_records=("start_day", "size"),
            start_records_nonmissing=("start_day", "count"),
            earliest_start_any=("start_day", "min"),
            earliest_start_nonnegative=(
                "start_day",
                lambda s: s[s >= 0].min() if (s >= 0).any() else np.nan,
            ),
            negative_start_records=("start_day", lambda s: int((s < 0).sum())),
        )
    )
    frame = pd.DataFrame({"patient_id_normalized": patient_ids.astype(str)})
    return frame.merge(starts, on="patient_id_normalized", how="left", validate="one_to_one")


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return np.nan
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def weighted_variance(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return np.nan
    mean = weighted_mean(x[mask], w[mask])
    return float(np.sum(w[mask] * (x[mask] - mean) ** 2) / np.sum(w[mask]))


def standardized_mean_difference(
    x: np.ndarray,
    group: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    if weights is None:
        weights = np.ones(len(x), dtype=float)
    m1 = weighted_mean(x[group == 1], weights[group == 1])
    m0 = weighted_mean(x[group == 0], weights[group == 0])
    v1 = weighted_variance(x[group == 1], weights[group == 1])
    v0 = weighted_variance(x[group == 0], weights[group == 0])
    pooled = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if np.sum(weights**2) <= 0:
        return np.nan
    return float((weights.sum() ** 2) / np.sum(weights**2))


def readable_feature(name: str) -> str:
    labels = {
        "W_age": "Age",
        "W_stage": "AJCC stage",
        "W_T": "Pathologic T category",
        "W_N": "Pathologic N category",
        "W_M": "Pathologic M category",
        "W_nodes_positive": "Positive lymph nodes",
        "W_nodes_tested": "Lymph nodes examined",
        "diagnosis_year": "Year of diagnosis",
        "diagnosis_year_missing": "Year of diagnosis missing",
    }
    if name in labels:
        return labels[name]
    return name.replace("W_", "").replace("_", " ").strip().capitalize()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result
