from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DerivedFeature:
    name: str
    kind: str
    source_columns: list[str]
    notes: str


def _find_columns(columns: Iterable[str], include: Iterable[str], exclude: Iterable[str] = ()) -> list[str]:
    found: list[str] = []
    for col in map(str, columns):
        low = col.lower()
        if all(token.lower() in low for token in include) and not any(
            token.lower() in low for token in exclude
        ):
            found.append(col)
    return sorted(found)


def _first_column(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    cols = list(map(str, columns))
    for pattern in patterns:
        pattern_low = pattern.lower()
        for col in cols:
            if pattern_low in col.lower():
                return col
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _one_hot_ordinal(
    df: pd.DataFrame,
    columns: list[str],
    parser,
    feature_name: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Collapse one-hot columns into an ordinal value and an unknown indicator.
    If multiple categories are active, the maximum parsed severity is used.
    """
    value = pd.Series(np.nan, index=df.index, dtype=float)
    recognized_any = pd.Series(False, index=df.index)

    for col in columns:
        parsed = parser(col)
        active = _numeric(df[col]).fillna(0) > 0.5
        if parsed is not None:
            recognized_any |= active
            current = value.loc[active]
            value.loc[active] = np.where(
                current.isna(),
                parsed,
                np.maximum(current.astype(float), float(parsed)),
            )

    any_active = (
        df[columns].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0.5).any(axis=1)
        if columns
        else pd.Series(False, index=df.index)
    )
    unknown = (~recognized_any) | (~any_active)
    return value, unknown.astype(float)


_ROMAN_STAGE = [
    ("stage iv", 4.0),
    ("stage iii", 3.0),
    ("stage ii", 2.0),
    ("stage i", 1.0),
    ("stage 0", 0.0),
]


def _parse_stage(col: str) -> float | None:
    low = col.lower().replace("_", " ")
    if any(x in low for x in ("not reported", "unknown", "stage x", "stage is")):
        return None
    for token, value in _ROMAN_STAGE:
        if token in low:
            return value
    return None


def _parse_t(col: str) -> float | None:
    low = col.lower()
    if any(x in low for x in ("tx", "not reported", "unknown")):
        return None
    if "tis" in low or re.search(r"(?:^|[_\s])t0(?:[^0-9]|$)", low):
        return 0.0
    match = re.search(r"(?:^|[_\s])t([1-4])(?:[a-d]|\b|_)", low)
    return float(match.group(1)) if match else None


def _parse_n(col: str) -> float | None:
    low = col.lower()
    if any(x in low for x in ("nx", "not reported", "unknown")):
        return None
    match = re.search(r"(?:^|[_\s])n([0-3])(?:[a-d]|\b|_)", low)
    return float(match.group(1)) if match else None


def _parse_m(col: str) -> float | None:
    low = col.lower()
    if any(x in low for x in ("mx", "not reported", "unknown")):
        return None
    match = re.search(r"(?:^|[_\s])m([01])(?:[a-d]|\b|_)", low)
    return float(match.group(1)) if match else None


def _parse_grade(col: str) -> float | None:
    low = col.lower().replace("_", " ")
    if any(x in low for x in ("not reported", "unknown", "gx")):
        return None
    match = re.search(r"(?:grade|g)\s*([1-4])", low)
    return float(match.group(1)) if match else None


def build_compact_adjustment(df: pd.DataFrame) -> tuple[pd.DataFrame, list[DerivedFeature]]:
    """
    Build a compact baseline clinical adjustment matrix W.

    The function intentionally excludes treatment, outcome, molecular features,
    tissue-of-origin, staging-system edition, and other post-hoc administrative fields.
    """
    out = pd.DataFrame(index=df.index)
    manifest: list[DerivedFeature] = []
    columns = list(map(str, df.columns))

    # Age.
    age_col = _first_column(
        columns,
        (
            "age_at_index",
            "age_at_diagnosis",
            "diagnosis_age",
            "age.demographic",
        ),
    )
    if age_col:
        out["W_age"] = _numeric(df[age_col])
        out["W_age_missing"] = out["W_age"].isna().astype(float)
        manifest.append(DerivedFeature("W_age", "continuous", [age_col], "Age at baseline"))
        manifest.append(DerivedFeature("W_age_missing", "missing_indicator", [age_col], "Age missingness"))

    # Pathologic stage, excluding staging-system edition.
    stage_cols = _find_columns(
        columns,
        include=("ajcc_pathologic_stage",),
        exclude=("staging_system", "edition"),
    )
    if stage_cols:
        stage, missing = _one_hot_ordinal(df, stage_cols, _parse_stage, "W_stage")
        out["W_stage"] = stage
        out["W_stage_missing"] = missing
        manifest.append(DerivedFeature("W_stage", "ordinal", stage_cols, "Collapsed AJCC pathologic stage"))
        manifest.append(DerivedFeature("W_stage_missing", "missing_indicator", stage_cols, "Stage unavailable/unknown"))

    # T, N, M groups.
    group_specs = [
        ("W_T", ("ajcc_pathologic_t",), _parse_t),
        ("W_N", ("ajcc_pathologic_n",), _parse_n),
        ("W_M", ("ajcc_pathologic_m",), _parse_m),
    ]
    for feature_name, include_tokens, parser in group_specs:
        group_cols = _find_columns(
            columns,
            include=include_tokens,
            exclude=("staging_system", "edition"),
        )
        if group_cols:
            value, missing = _one_hot_ordinal(df, group_cols, parser, feature_name)
            out[feature_name] = value
            out[f"{feature_name}_missing"] = missing
            manifest.append(DerivedFeature(feature_name, "ordinal", group_cols, f"Collapsed {feature_name[2:]} category"))
            manifest.append(DerivedFeature(f"{feature_name}_missing", "missing_indicator", group_cols, "Unknown category"))

    # Lymph node counts.
    for output_name, patterns, note in [
        (
            "W_nodes_positive",
            ("pathology_details.lymph_nodes_positive", "lymph_nodes_positive"),
            "Number of positive lymph nodes",
        ),
        (
            "W_nodes_tested",
            ("pathology_details.lymph_nodes_tested", "lymph_nodes_tested"),
            "Number of tested lymph nodes",
        ),
    ]:
        source = _first_column(columns, patterns)
        if source:
            out[output_name] = _numeric(df[source])
            out[f"{output_name}_missing"] = out[output_name].isna().astype(float)
            manifest.append(DerivedFeature(output_name, "continuous", [source], note))
            manifest.append(DerivedFeature(f"{output_name}_missing", "missing_indicator", [source], f"{note} missingness"))

    # Histologic grade, when available.
    raw_grade = _first_column(
        columns,
        ("tumor_grade", "histologic_grade", "neoplasm_histologic_grade"),
    )
    if raw_grade and pd.to_numeric(df[raw_grade], errors="coerce").notna().mean() >= 0.5:
        out["W_grade"] = _numeric(df[raw_grade])
        out["W_grade_missing"] = out["W_grade"].isna().astype(float)
        manifest.append(DerivedFeature("W_grade", "ordinal", [raw_grade], "Histologic grade"))
        manifest.append(DerivedFeature("W_grade_missing", "missing_indicator", [raw_grade], "Grade missingness"))
    else:
        grade_cols = [
            col for col in columns
            if "grade" in col.lower()
            and not any(x in col.lower() for x in ("degrade", "upgrade"))
        ]
        if grade_cols:
            grade, missing = _one_hot_ordinal(df, grade_cols, _parse_grade, "W_grade")
            if grade.notna().any():
                out["W_grade"] = grade
                out["W_grade_missing"] = missing
                manifest.append(DerivedFeature("W_grade", "ordinal", grade_cols, "Collapsed histologic grade"))
                manifest.append(DerivedFeature("W_grade_missing", "missing_indicator", grade_cols, "Grade unavailable/unknown"))

    # Receptors: included only when they vary within the selected cohort.
    for receptor in ("ER_status", "PR_status", "HER2_status"):
        if receptor in df.columns:
            numeric = pd.to_numeric(df[receptor], errors="coerce")
            if numeric.nunique(dropna=True) > 1:
                name = f"W_{receptor.replace('_status', '')}"
                out[name] = numeric
                out[f"{name}_missing"] = numeric.isna().astype(float)
                manifest.append(DerivedFeature(name, "binary", [receptor], f"{receptor} baseline status"))
                manifest.append(DerivedFeature(f"{name}_missing", "missing_indicator", [receptor], f"{receptor} missingness"))

    # Remove empty/constant columns.
    keep = []
    for col in out.columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.nunique(dropna=True) > 1:
            keep.append(col)
    out = out[keep].copy()

    kept_names = set(out.columns)
    manifest = [item for item in manifest if item.name in kept_names]
    return out, manifest


def manifest_to_frame(manifest: list[DerivedFeature]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "derived_feature": item.name,
                "kind": item.kind,
                "source_columns": "|".join(item.source_columns),
                "notes": item.notes,
            }
            for item in manifest
        ]
    )
