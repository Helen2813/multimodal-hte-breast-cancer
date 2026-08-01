from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats

from _metabric_m6_utils import normalize_tcga_id


def fast_harrell_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    if valid.sum() < 3 or event[valid].sum() == 0:
        return float("nan")
    # lifelines expects larger predicted scores to indicate longer survival.
    return float(
        concordance_index(
            time[valid],
            -risk[valid],
            event_observed=event[valid],
        )
    )


def candidate_id_columns(frame: pd.DataFrame, maximum: int = 12) -> list[str]:
    rows = []
    for position, column in enumerate(frame.columns):
        normalized = re.sub(r"[^A-Z0-9]+", "_", str(column).upper()).strip("_")
        score = 0.0
        if normalized in {
            "SAMPLE", "SAMPLE_ID", "PATIENT_ID", "CASE_ID",
            "SUBMITTER_ID", "UNNAMED_0"
        }:
            score += 120
        if any(token in normalized for token in ("SAMPLE", "PATIENT", "CASE", "SUBMITTER")):
            score += 50
        if normalized.startswith("UNNAMED"):
            score += 30
        sample = frame[column].dropna().astype(str).head(300)
        if len(sample):
            tcga_fraction = sample.str.contains(
                r"TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}",
                case=False,
                regex=True,
            ).mean()
            score += 150 * float(tcga_fraction)
            uniqueness = sample.nunique() / len(sample)
            score += 25 * float(uniqueness)
        score -= 0.001 * position
        rows.append((score, column))
    rows.sort(key=lambda item: (-item[0], str(item[1])))
    return [column for _, column in rows[:maximum]]


def resolve_tcga_id_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    minimum_overlap: int,
) -> tuple[str, str, list[dict]]:
    left_candidates = candidate_id_columns(left)
    right_candidates = candidate_id_columns(right)
    pair_rows = []
    for left_column in left_candidates:
        left_ids = {
            normalize_tcga_id(value)
            for value in left[left_column].dropna().astype(str)
        }
        for right_column in right_candidates:
            right_ids = {
                normalize_tcga_id(value)
                for value in right[right_column].dropna().astype(str)
            }
            overlap = len(left_ids & right_ids)
            pair_rows.append({
                "left_column": left_column,
                "right_column": right_column,
                "overlap": overlap,
                "left_unique": len(left_ids),
                "right_unique": len(right_ids),
                "left_coverage": overlap / len(left_ids) if left_ids else 0.0,
                "right_coverage": overlap / len(right_ids) if right_ids else 0.0,
            })
    pair_rows.sort(
        key=lambda row: (
            -int(row["overlap"]),
            -float(row["left_coverage"]),
            -float(row["right_coverage"]),
            str(row["left_column"]),
            str(row["right_column"]),
        )
    )
    if not pair_rows or int(pair_rows[0]["overlap"]) < int(minimum_overlap):
        best = pair_rows[0] if pair_rows else {}
        raise RuntimeError(
            f"Could not resolve TCGA identifiers with overlap >= {minimum_overlap}. "
            f"Best pair: {best}"
        )
    best = pair_rows[0]
    return str(best["left_column"]), str(best["right_column"]), pair_rows


def exact_tcga_stage_indicator(frame: pd.DataFrame, roman_stage: str) -> pd.Series:
    base = r"^CLIN_ajcc_pathologic_stage\.diagnoses_Stage "
    if roman_stage == "II":
        pattern = re.compile(base + r"II(?:A|B|C)?$", re.IGNORECASE)
    elif roman_stage == "III":
        pattern = re.compile(base + r"III(?:A|B|C)?$", re.IGNORECASE)
    elif roman_stage == "IV":
        pattern = re.compile(base + r"IV$", re.IGNORECASE)
    else:
        raise ValueError(f"Unsupported stage: {roman_stage}")
    columns = [column for column in frame.columns if pattern.match(str(column))]
    if not columns:
        return pd.Series(0.0, index=frame.index)
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .astype(float)
    )


def tcga_node_positive_indicator(frame: pd.DataFrame) -> pd.Series:
    pattern = re.compile(
        r"^CLIN_ajcc_pathologic_n\.diagnoses_N(?:1|2|3)",
        re.IGNORECASE,
    )
    columns = [column for column in frame.columns if pattern.match(str(column))]
    if not columns:
        return pd.Series(0.0, index=frame.index)
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .astype(float)
    )


def top_signed_spearman_chunked(
    frame: pd.DataFrame,
    train_positions: Sequence[int],
    outcome: np.ndarray,
    positive: int,
    negative: int,
    chunk_size: int = 512,
) -> tuple[list[str], list[dict]]:
    y = np.asarray(outcome, dtype=float)
    y_rank = stats.rankdata(y, method="average").astype(np.float64)
    y_centered = y_rank - y_rank.mean()
    y_norm = np.sqrt(np.sum(y_centered * y_centered))
    rows = []

    train_positions = np.asarray(train_positions, dtype=int)
    columns = list(frame.columns)
    for start in range(0, len(columns), chunk_size):
        chunk_columns = columns[start:start + chunk_size]
        chunk = frame.iloc[train_positions][chunk_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        medians = chunk.median(axis=0).fillna(0.0)
        chunk = chunk.fillna(medians)
        ranks = chunk.rank(axis=0, method="average").to_numpy(dtype=np.float64)
        ranks -= ranks.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(ranks * ranks, axis=0)) * y_norm
        numerator = ranks.T @ y_centered
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-15,
        )
        for column, correlation in zip(chunk_columns, correlations):
            rows.append({
                "feature": column,
                "spearman_rho": float(correlation),
            })

    positive_rows = [
        row for row in sorted(
            rows,
            key=lambda row: (-row["spearman_rho"], row["feature"]),
        )
        if row["spearman_rho"] > 0
    ][:positive]
    negative_rows = [
        row for row in sorted(
            rows,
            key=lambda row: (row["spearman_rho"], row["feature"]),
        )
        if row["spearman_rho"] < 0
    ][:negative]

    selected = []
    for row in positive_rows + negative_rows:
        if row["feature"] not in selected:
            selected.append(row["feature"])
    return selected, rows


def read_float32_csv(path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    if not header:
        raise RuntimeError(f"Empty CSV header: {path}")
    dtype = {column: "float32" for column in header[1:]}
    frame = pd.read_csv(path, dtype=dtype, low_memory=False)
    return frame
