#!/usr/bin/env python3
"""Recover the exact CCW row_id -> patient-ID mapping from the Stage 41 runtime.

Why this is necessary
---------------------
The saved Stage 14 clone-long CSV contains ``row_id`` but not the patient ID. The original
DataFrame index was intentionally not written by the generic trace exporter. Rather than guessing
which saved table can bridge the IDs, this script re-runs the already verified Stage 41 point
estimator once and traces the *call* to ``_clone_long_rows``. At that call, the exact input
DataFrame and its index are still available.

The script then validates one of three deterministic mappings:
1. row_id equals a named key column in the input DataFrame;
2. row_id equals the input DataFrame index;
3. row_id is exactly 0..n-1 and therefore maps to input-row position.

No order-only mapping is accepted unless the row_id values themselves are the complete integer
range 0..n-1.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage14_utils import project_root, read_csv, write_csv
from _stage15_utils import (
    choose_landmark_candidate,
    ensure_dirs,
    load_config,
    normalize_id_series,
    sha256_file,
)


def normalize_preserving_missing(series: pd.Series) -> pd.Series:
    missing = series.isna()
    out = normalize_id_series(series)
    out = out.mask(missing)
    return out.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NA": np.nan})


def named_patient_id_columns(df: pd.DataFrame) -> list[str]:
    exact_order = (
        "patient_id_normalized",
        "patient_id_norm",
        "normalized_patient_id",
        "patient_id",
        "cases.submitter_id",
        "case_submitter_id",
        "submitter_id",
        "case_id",
    )
    lower = {str(col).lower(): str(col) for col in df.columns}
    found = [lower[name.lower()] for name in exact_order if name.lower() in lower]
    for col in df.columns:
        name = str(col).lower()
        if str(col) in found:
            continue
        if any(token in name for token in ("patient_id", "submitter_id", "case_id")):
            if not any(token in name for token in ("rna", "cnv", "protein", "methyl", "mirna", "mutation")):
                found.append(str(col))
    return found


def identifier_sources(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    sources: list[tuple[str, pd.Series]] = []
    for col in named_patient_id_columns(df):
        sources.append((f"column:{col}", df[col]))
    # The runtime index is important because the Stage 14 CSV exporter used index=False.
    if not isinstance(df.index, pd.RangeIndex):
        sources.append(("index", pd.Series(df.index, index=df.index)))
    return sources


def overlap_with_landmark(series: pd.Series, landmark_ids: set[str]) -> tuple[float, int, int]:
    values = set(normalize_preserving_missing(series).dropna().astype(str).unique())
    intersection = len(values & landmark_ids)
    coverage = intersection / len(landmark_ids) if landmark_ids else 0.0
    return coverage, intersection, len(values)


def collect_dataframes(value: Any, label: str, output: list[tuple[str, pd.DataFrame]]) -> None:
    if isinstance(value, pd.DataFrame):
        output.append((label, value.copy()))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            collect_dataframes(item, f"{label}[{index}]", output)
    elif isinstance(value, dict):
        for key, item in value.items():
            collect_dataframes(item, f"{label}[{key!r}]", output)


def choose_landmark_id(cohort: pd.DataFrame) -> tuple[str, pd.Series, set[str]]:
    candidates = []
    for source_name, series in identifier_sources(cohort):
        normalized = normalize_preserving_missing(series)
        unique = normalized.nunique(dropna=True)
        if unique < 500:
            continue
        name_bonus = 10 if source_name.startswith("column:patient_id") else 0
        candidates.append((name_bonus + unique / 1000.0, source_name, series))
    if not candidates:
        raise RuntimeError(
            "The 559-row landmark candidate has no named patient-ID column or non-range patient index."
        )
    _, source_name, series = max(candidates, key=lambda item: item[0])
    landmark_ids = set(normalize_preserving_missing(series).dropna().astype(str).unique())
    return source_name, series, landmark_ids


def choose_runtime_base(
    captured_inputs: list[tuple[str, str, pd.DataFrame]],
    landmark_ids: set[str],
) -> tuple[str, str, pd.DataFrame, str, pd.Series, float]:
    scored = []
    for function_name, local_name, frame in captured_inputs:
        if not (580 <= len(frame) <= 620):
            continue
        for source_name, series in identifier_sources(frame):
            coverage, intersection, unique = overlap_with_landmark(series, landmark_ids)
            scored.append(
                (
                    coverage,
                    intersection,
                    unique,
                    function_name,
                    local_name,
                    frame,
                    source_name,
                    series,
                )
            )
    if not scored:
        raise RuntimeError(
            "No 580-620-row DataFrame with a named/index patient identifier was visible at the "
            "_clone_long_rows call."
        )
    best = max(scored, key=lambda item: (item[0], item[1]))
    coverage, intersection, unique, function_name, local_name, frame, source_name, series = best
    if coverage < 0.95:
        audit = pd.DataFrame(
            [
                {
                    "function": item[3],
                    "local": item[4],
                    "patient_source": item[6],
                    "landmark_coverage": item[0],
                    "intersection": item[1],
                    "unique_values": item[2],
                }
                for item in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[:20]
            ]
        )
        raise RuntimeError(
            "The runtime source DataFrame was captured, but no patient identifier covered at "
            f"least 95% of landmark IDs. Best coverage={coverage:.3f}. "
            f"Top candidates:\n{audit.to_string(index=False)}"
        )
    return function_name, local_name, frame, source_name, series, coverage


def choose_row_id_column(long_df: pd.DataFrame) -> str:
    exact = [str(col) for col in long_df.columns if str(col).lower() == "row_id"]
    if exact:
        return exact[0]
    candidates = [str(col) for col in long_df.columns if "row_id" in str(col).lower()]
    if not candidates:
        raise RuntimeError("The Stage 14 clone-long table has no row_id column.")
    return candidates[0]


def build_mapping(
    base: pd.DataFrame,
    patient_series: pd.Series,
    long_df: pd.DataFrame,
    row_id_col: str,
) -> tuple[pd.DataFrame, str]:
    patient_ids = normalize_preserving_missing(patient_series).reset_index(drop=True)
    row_raw = long_df[row_id_col]
    row_norm = normalize_preserving_missing(row_raw)

    # 1. A named base key exactly matches the clone row_id.
    for col in base.columns:
        name = str(col).lower()
        if "row_id" not in name and name not in {"source_row", "source_index", "original_row"}:
            continue
        base_key = normalize_preserving_missing(base[col]).reset_index(drop=True)
        if base_key.nunique(dropna=True) != len(base):
            continue
        mapping = pd.DataFrame(
            {"__stage15_row_key": base_key, "__stage15_patient_id": patient_ids}
        ).dropna()
        intersection = len(set(mapping["__stage15_row_key"]) & set(row_norm.dropna()))
        if intersection == row_norm.nunique(dropna=True):
            return mapping, f"NAMED_RUNTIME_KEY:{col}"

    # 2. The runtime input index exactly matches clone row_id.
    if not isinstance(base.index, pd.RangeIndex):
        index_key = normalize_preserving_missing(pd.Series(base.index)).reset_index(drop=True)
        if index_key.nunique(dropna=True) == len(base):
            mapping = pd.DataFrame(
                {"__stage15_row_key": index_key, "__stage15_patient_id": patient_ids}
            ).dropna()
            intersection = len(set(mapping["__stage15_row_key"]) & set(row_norm.dropna()))
            if intersection == row_norm.nunique(dropna=True):
                return mapping, "RUNTIME_INDEX_KEY"

    # 3. Accept positional mapping only when row_id is exactly the complete integer range.
    row_numeric = pd.to_numeric(row_raw, errors="coerce")
    unique_numeric = sorted(row_numeric.dropna().astype(int).unique().tolist())
    expected = list(range(len(base)))
    if unique_numeric == expected and np.allclose(row_numeric.dropna(), row_numeric.dropna().astype(int)):
        mapping = pd.DataFrame(
            {
                "__stage15_row_key": normalize_preserving_missing(
                    pd.Series(np.arange(len(base), dtype=int))
                ),
                "__stage15_patient_id": patient_ids,
            }
        ).dropna()
        return mapping, "VERIFIED_POSITIONAL_ROW_ID_0_TO_N_MINUS_1"

    raise RuntimeError(
        "The exact runtime base was captured, but row_id could not be tied to a named key, "
        "runtime index, or verified complete integer range 0..n-1."
    )


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    design = cfg["design"]
    tables = root / "results/tables"
    output_dir = root / "data/derived/stage15"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = root / "data/derived/stage14_trace/53_candidate_01.csv"
    long_df = read_csv(long_path)
    row_id_col = choose_row_id_column(long_df)

    landmark, landmark_path = choose_landmark_candidate(
        root, int(design["expected_landmark_n"])
    )
    landmark_source, _, landmark_ids = choose_landmark_id(landmark)

    stage41_outputs = [
        root / "results/tables/41_landmark_replication_check.csv",
        root / "results/tables/41_ccw_point_estimate.csv",
    ]
    hashes_before = {
        str(path.relative_to(root)): sha256_file(path)
        for path in stage41_outputs
        if path.exists()
    }

    target_file = str((root / "scripts/_stage12_utils.py").resolve())
    captured_inputs: list[tuple[str, str, pd.DataFrame]] = []
    captured_returns: list[tuple[str, pd.DataFrame]] = []

    def tracer(frame, event, arg):
        filename = str(Path(frame.f_code.co_filename).resolve())
        function_name = frame.f_code.co_name
        if filename != target_file or function_name != "_clone_long_rows":
            return tracer
        if event == "call":
            for local_name, value in list(frame.f_locals.items()):
                if isinstance(value, pd.DataFrame):
                    captured_inputs.append((function_name, local_name, value.copy()))
        elif event == "return":
            collect_dataframes(arg, "return", captured_returns)
        return tracer

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(root / "scripts/41_replicate_estimators.py")]
        sys.settrace(tracer)
        try:
            runpy.run_path(
                str(root / "scripts/41_replicate_estimators.py"),
                run_name="__main__",
            )
        except SystemExit as exc:
            if int(exc.code or 0) != 0:
                raise RuntimeError(f"Stage 41 traced run exited with code {exc.code}")
    finally:
        sys.settrace(None)
        sys.argv = old_argv

    (
        function_name,
        local_name,
        base,
        patient_source,
        patient_series,
        landmark_coverage,
    ) = choose_runtime_base(captured_inputs, landmark_ids)

    mapping, mapping_method = build_mapping(
        base, patient_series, long_df, row_id_col
    )
    if mapping["__stage15_row_key"].duplicated().any():
        raise RuntimeError("Recovered runtime row key is not unique.")
    if mapping["__stage15_patient_id"].duplicated().any():
        raise RuntimeError("Recovered patient ID is not unique in the source cohort.")

    mapped = long_df.copy()
    mapped["__stage15_row_key"] = normalize_preserving_missing(mapped[row_id_col])
    mapped = mapped.merge(
        mapping,
        on="__stage15_row_key",
        how="left",
        validate="many_to_one",
    )
    mapped_fraction = mapped["__stage15_patient_id"].notna().mean()
    mapped_unique = mapped["__stage15_patient_id"].nunique(dropna=True)
    mapped_ids = set(mapped["__stage15_patient_id"].dropna().astype(str).unique())
    landmark_intersection = len(mapped_ids & landmark_ids)
    landmark_coverage_final = landmark_intersection / len(landmark_ids)

    if mapped_fraction < 0.999:
        raise RuntimeError(f"Only {mapped_fraction:.2%} of clone rows received patient IDs.")
    if mapped_unique != int(design.get("expected_ccw_eligible_n", 594)):
        # The Stage 15 config may omit this key; 594 is the verified Stage 12/13 count.
        if mapped_unique != 594:
            raise RuntimeError(
                f"Expected 594 unique CCW patients after mapping, found {mapped_unique}."
            )
    if landmark_coverage_final < 0.999:
        raise RuntimeError(
            f"Only {landmark_coverage_final:.2%} of landmark patients occur in mapped CCW rows."
        )

    mapped_path = output_dir / "60_ccw_long_with_patient_id.csv"
    mapping_path = output_dir / "60_ccw_row_id_patient_map.csv"
    write_csv(mapped, mapped_path)
    write_csv(mapping, mapping_path)

    hashes_after = {
        str(path.relative_to(root)): sha256_file(path)
        for path in stage41_outputs
        if path.exists()
    }
    hash_rows = []
    for relative, before in hashes_before.items():
        after = hashes_after.get(relative, "")
        hash_rows.append(
            {
                "path": relative,
                "before_sha256": before,
                "after_sha256": after,
                "unchanged": before == after,
            }
        )
    hash_check = pd.DataFrame(hash_rows)
    if not hash_check.empty and not hash_check["unchanged"].all():
        raise RuntimeError("The traced Stage 41 run changed a verified Stage 41 output.")

    diagnostics = pd.DataFrame(
        [
            {
                "clone_long_path": str(long_path.relative_to(root)),
                "clone_long_rows": len(long_df),
                "row_id_column": row_id_col,
                "captured_clone_function": function_name,
                "captured_input_local": local_name,
                "runtime_base_rows": len(base),
                "runtime_patient_source": patient_source,
                "landmark_id_source": landmark_source,
                "runtime_patient_landmark_coverage": landmark_coverage,
                "row_mapping_method": mapping_method,
                "mapped_clone_row_fraction": mapped_fraction,
                "mapped_unique_patients": mapped_unique,
                "landmark_unique_patients": len(landmark_ids),
                "landmark_intersection": landmark_intersection,
                "landmark_coverage_final": landmark_coverage_final,
                "mapped_long_output": str(mapped_path.relative_to(root)),
                "mapping_output": str(mapping_path.relative_to(root)),
            }
        ]
    )
    write_csv(diagnostics, tables / "60_exact_runtime_row_map_diagnostics.csv")
    write_csv(hash_check, tables / "60_stage41_hash_check.csv")

    print("=" * 116)
    print("STAGE 60 — EXACT RUNTIME CCW ROW-ID MAP")
    print("=" * 116)
    print(diagnostics.to_string(index=False))
    print("\nStage 41 output hashes")
    print(hash_check.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
