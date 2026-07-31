#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import json
import os
import runpy
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from _stage14_utils import (
    discover_curve_columns,
    integrate_step_curve,
    prepare_survival_rows,
    project_root,
    read_csv,
    select_estimate_column,
    strategy_orientation,
    weighted_counting_process_km,
    write_csv,
    write_json,
    write_text,
)


def load_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "stage15_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/figures",
        "results/logs",
        "data/derived/stage15",
        "data/derived/manifests",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(
            lambda value: "" if pd.isna(value)
            else (f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        )
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines)


def normalize_id_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^A-Z0-9-]", "", regex=True)
    )



def _id_name_score(column: str) -> float:
    """Prioritize identifiers and strongly penalize omics/features/index-like columns."""
    name = str(column).lower()
    score = 0.0
    for token, points in (
        ("patient_id_norm", 60),
        ("normalized_patient", 55),
        ("patient_id", 50),
        ("submitter_id", 48),
        ("case_id", 42),
        ("case_submitter", 42),
        ("participant_id", 40),
        ("subject_id", 40),
        ("patient", 24),
        ("submitter", 20),
        ("case", 12),
        ("row_id", 4),
    ):
        if token in name:
            score += points
    for token, penalty in (
        ("rna_", 80),
        ("ensg", 80),
        ("cnv_", 80),
        ("mutation", 70),
        ("methyl", 70),
        ("mirna", 70),
        ("protein", 70),
        ("clin_", 25),
        ("feature", 50),
        ("score", 20),
        ("propensity", 50),
        ("weight", 50),
        ("event", 50),
        ("time", 30),
        ("unnamed", 8),
    ):
        if token in name:
            score -= penalty
    return score


def _normalized_nonmissing(series: pd.Series) -> pd.Series:
    original_missing = series.isna()
    out = normalize_id_series(series)
    out = out.mask(original_missing)
    out = out.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NA": np.nan})
    return out


def candidate_id_columns(
    df: pd.DataFrame,
    low: int,
    high: int,
    *,
    permit_low_name_score: bool = False,
) -> list[str]:
    """Find plausible ID columns without confusing omics features with identifiers."""
    rows = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        if not (low <= n_unique <= high):
            continue
        name_score = _id_name_score(str(col))
        normalized = _normalized_nonmissing(series)
        if normalized.notna().sum() == 0:
            continue
        # Identifiers usually have mostly unique values and relatively short strings.
        median_length = normalized.dropna().astype(str).str.len().median()
        uniqueness = n_unique / max(series.notna().sum(), 1)
        value_score = 0.0
        if uniqueness >= 0.8:
            value_score += 8
        if median_length <= 40:
            value_score += 3
        if normalized.dropna().astype(str).str.contains(r"TCGA|BRCA|PATIENT|CASE", regex=True).mean() > 0.2:
            value_score += 20
        total = name_score + value_score
        if permit_low_name_score or total > 0:
            rows.append((total, str(col)))
    return [col for _, col in sorted(rows, reverse=True)]


def _pair_overlap(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float, float, int]:
    left_set = set(_normalized_nonmissing(left).dropna().unique())
    right_set = set(_normalized_nonmissing(right).dropna().unique())
    if not left_set or not right_set:
        return 0.0, 0.0, 0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    jaccard = intersection / union if union else 0.0
    smaller_coverage = intersection / min(len(left_set), len(right_set))
    return jaccard, smaller_coverage, intersection


def _best_direct_id_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_range: tuple[int, int],
    right_range: tuple[int, int],
) -> tuple[float, float, int, str, str] | None:
    left_cols = candidate_id_columns(left_df, *left_range)
    right_cols = candidate_id_columns(right_df, *right_range)
    # Last-resort candidates are allowed only after named identifier columns.
    if not left_cols:
        left_cols = candidate_id_columns(left_df, *left_range, permit_low_name_score=True)
    if not right_cols:
        right_cols = candidate_id_columns(right_df, *right_range, permit_low_name_score=True)

    scored = []
    for left_col in left_cols:
        for right_col in right_cols:
            jaccard, coverage, intersection = _pair_overlap(left_df[left_col], right_df[right_col])
            name_bonus = 0.002 * (_id_name_score(left_col) + _id_name_score(right_col))
            score = coverage + jaccard + name_bonus
            scored.append((score, jaccard, coverage, intersection, left_col, right_col))
    if not scored:
        return None
    _, jaccard, coverage, intersection, left_col, right_col = max(scored)
    return jaccard, coverage, intersection, left_col, right_col


def _trace_manifest_for_bridge(root: Path) -> pd.DataFrame:
    path = root / "results/tables/53_ccw_trace_candidate_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def resolve_shared_patient_id(
    root: Path,
    long_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    minimum_jaccard: float,
) -> tuple[pd.DataFrame, str, str, float, str]:
    """Resolve patient IDs directly or through a captured 594-row source table.

    The Stage 14 clone-long table uses ``row_id`` as its linkage key in some runs. The
    landmark table does not necessarily retain that key. When a direct patient-ID match
    is unavailable, this function uses another Stage 14 captured source-cohort DataFrame
    to bridge:

        clone-long row_id -> source row_id -> source patient ID -> landmark patient ID

    No positional row matching is used.
    """
    direct = _best_direct_id_pair(
        long_df,
        cohort_df,
        (500, 700),
        (500, 650),
    )
    if direct is not None:
        jaccard, coverage, intersection, long_col, cohort_col = direct
        if jaccard >= minimum_jaccard or coverage >= 0.95:
            out = long_df.copy()
            out["__stage15_patient_id"] = _normalized_nonmissing(out[long_col])
            return out, "__stage15_patient_id", cohort_col, jaccard, "DIRECT_ID_MATCH"

    manifest = _trace_manifest_for_bridge(root)
    bridge_attempts = []
    if not manifest.empty:
        for _, meta in manifest.iterrows():
            rows = int(meta.get("rows", -1))
            if not (580 <= rows <= 620):
                continue
            path = root / str(meta["path"])
            if not path.exists():
                continue
            source = read_csv(path)

            long_source = _best_direct_id_pair(
                long_df,
                source,
                (500, 700),
                (580, 620),
            )
            source_cohort = _best_direct_id_pair(
                source,
                cohort_df,
                (580, 620),
                (500, 650),
            )
            if long_source is None or source_cohort is None:
                continue

            ls_j, ls_cov, ls_n, long_key, source_key = long_source
            sc_j, sc_cov, sc_n, source_patient, cohort_patient = source_cohort
            score = ls_cov + sc_cov + ls_j + sc_j
            bridge_attempts.append(
                (
                    score,
                    ls_j,
                    ls_cov,
                    sc_j,
                    sc_cov,
                    long_key,
                    source_key,
                    source_patient,
                    cohort_patient,
                    path,
                    source,
                )
            )

    if bridge_attempts:
        (
            _,
            ls_j,
            ls_cov,
            sc_j,
            sc_cov,
            long_key,
            source_key,
            source_patient,
            cohort_patient,
            source_path,
            source_df,
        ) = max(bridge_attempts, key=lambda item: item[0])

        if (ls_j >= minimum_jaccard or ls_cov >= 0.95) and (
            sc_j >= minimum_jaccard or sc_cov >= 0.95
        ):
            mapping = pd.DataFrame(
                {
                    "__bridge_key": _normalized_nonmissing(source_df[source_key]),
                    "__stage15_patient_id": _normalized_nonmissing(source_df[source_patient]),
                }
            ).dropna().drop_duplicates()

            if mapping["__bridge_key"].duplicated().any():
                raise RuntimeError(
                    f"Bridge key is not unique in {source_path.relative_to(root)}: {source_key}"
                )

            out = long_df.copy()
            out["__bridge_key"] = _normalized_nonmissing(out[long_key])
            out = out.merge(mapping, on="__bridge_key", how="left", validate="many_to_one")
            resolved_fraction = out["__stage15_patient_id"].notna().mean()
            if resolved_fraction < 0.95:
                raise RuntimeError(
                    f"Only {resolved_fraction:.1%} of clone rows were resolved through "
                    f"{source_path.relative_to(root)}."
                )
            effective_jaccard = sc_j
            method = (
                f"TRACE_BRIDGE:{source_path.relative_to(root)}:"
                f"{long_key}->{source_key}->{source_patient}"
            )
            return out, "__stage15_patient_id", cohort_patient, effective_jaccard, method

    detail = ""
    if direct is not None:
        jaccard, coverage, intersection, long_col, cohort_col = direct
        detail = (
            f" Best direct pair: {long_col} vs {cohort_col}; "
            f"Jaccard={jaccard:.3f}; smaller-set coverage={coverage:.3f}; "
            f"intersection={intersection}."
        )
    raise RuntimeError(
        "Could not resolve a patient-ID bridge between the CCW clone table and the "
        f"landmark cohort.{detail}"
    )


def detect_shared_patient_id(
    long_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    minimum_jaccard: float,
) -> tuple[str, str, float]:
    """Backward-compatible direct matcher.

    New code should call ``resolve_shared_patient_id`` because clone tables may use
    source-row identifiers rather than patient identifiers.
    """
    direct = _best_direct_id_pair(long_df, cohort_df, (500, 700), (500, 650))
    if direct is None:
        raise RuntimeError("No candidate patient-ID columns were found.")
    jaccard, coverage, intersection, long_col, cohort_col = direct
    if jaccard < minimum_jaccard and coverage < 0.95:
        raise RuntimeError(
            f"Best patient-ID match was below threshold: {long_col} vs {cohort_col}, "
            f"Jaccard={jaccard:.3f}, coverage={coverage:.3f}"
        )
    return long_col, cohort_col, jaccard

def detect_binary_column(
    df: pd.DataFrame,
    preferred_tokens: Sequence[str],
    excluded_tokens: Sequence[str] = (),
) -> str:
    scored = []
    for col in df.columns:
        name = str(col).lower()
        if any(token in name for token in excluded_tokens):
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        unique = set(values.unique())
        if unique.issubset({0, 1, 0.0, 1.0}) and len(unique) == 2:
            scored.append((10 * sum(token in name for token in preferred_tokens), str(col)))
    if not scored:
        raise RuntimeError(f"No binary column matched tokens={preferred_tokens}")
    return max(scored)[1]


def detect_time_column(df: pd.DataFrame) -> str:
    scored = []
    for col in df.columns:
        name = str(col).lower()
        if any(token in name for token in ("treatment_start", "days_to_treatment", "birth")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 0.8 * len(df) or finite.min() < 0 or finite.max() < 100:
            continue
        score = 0
        for token, points in (
            ("analysis_time", 20),
            ("time_after_landmark", 20),
            ("followup", 10),
            ("survival", 8),
            ("time", 5),
            ("days", 3),
        ):
            if token in name:
                score += points
        if score:
            scored.append((score, str(col)))
    if not scored:
        raise RuntimeError("No post-landmark time column was detected.")
    return max(scored)[1]


def detect_propensity_column(df: pd.DataFrame, minimum_unique: int) -> str:
    scored = []
    for col in df.columns:
        name = str(col).lower()
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 0.8 * len(df) or finite.nunique() < minimum_unique:
            continue
        if finite.min() < 0 or finite.max() > 1:
            continue
        score = 0
        for token, points in (
            ("oof_propensity", 30),
            ("propensity_score", 25),
            ("propensity", 20),
            ("ps_oof", 18),
            ("ps_", 8),
            ("score", 2),
        ):
            if token in name:
                score += points
        if score:
            scored.append((score, str(col)))
    if not scored:
        raise RuntimeError("No continuous propensity-score column was detected.")
    return max(scored)[1]


def trace_manifest(root: Path) -> pd.DataFrame:
    path = root / "results/tables/53_ccw_trace_candidate_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def choose_landmark_candidate(root: Path, expected_n: int) -> tuple[pd.DataFrame, str]:
    manifest = trace_manifest(root)
    candidates = []
    for _, row in manifest.iterrows():
        if int(row["rows"]) != expected_n:
            continue
        path = root / str(row["path"])
        df = read_csv(path)
        columns = [str(c).lower() for c in df.columns]
        score = 0
        score += 10 * sum("treatment" in c for c in columns)
        score += 10 * sum("propensity" in c or "ps_oof" in c for c in columns)
        score += 8 * sum("analysis_event" in c for c in columns)
        score += 8 * sum("analysis_time" in c for c in columns)
        candidates.append((score, df, str(path.relative_to(root))))
    if not candidates:
        raise RuntimeError(f"No {expected_n}-row landmark candidate was found.")
    _, df, path = max(candidates, key=lambda item: item[0])
    return df, path


def add_clone_id(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["clone_id"] = out["patient_id_norm"].astype(str) + "::" + out["strategy_raw"].astype(str)
    return out


def orient_strategy(rows: pd.DataFrame) -> dict[object, str]:
    raw_values = list(rows["strategy_raw"].dropna().unique())
    if len(raw_values) != 2:
        raise RuntimeError(f"Expected two strategies, found {raw_values}")
    mapping = strategy_orientation(rows["strategy_raw"])
    if mapping is not None:
        return mapping
    numeric_values = pd.to_numeric(pd.Series(raw_values), errors="coerce")
    if numeric_values.notna().all() and set(numeric_values.astype(float)) == {0.0, 1.0}:
        return {
            raw: "initiate_by_180" if float(raw) == 1.0 else "no_initiation_by_180"
            for raw in raw_values
        }
    raise RuntimeError(f"Could not orient strategy values: {raw_values}")


def conditional_strategy_effect(
    rows: pd.DataFrame,
    horizon: float,
    landmark: float,
) -> tuple[pd.DataFrame, float]:
    mapping = orient_strategy(rows)
    summaries = []
    for raw in rows["strategy_raw"].dropna().unique():
        curve, stats = weighted_counting_process_km(rows, raw, horizon)
        conditional, s180 = integrate_step_curve(curve, landmark, horizon, conditional=True)
        summaries.append(
            {
                "strategy": mapping[raw],
                "raw_strategy": str(raw),
                "conditional_rmst_day180_to_day910": conditional,
                "survival_day180": s180,
                "weight_mean": stats["weight_mean"],
                "weight_p99": stats["weight_p99"],
                "weight_max": stats["weight_max"],
                "n_clones": stats["n_clones"],
            }
        )
    summary = pd.DataFrame(summaries)
    init = float(summary.loc[summary["strategy"] == "initiate_by_180", "conditional_rmst_day180_to_day910"].iloc[0])
    noinit = float(summary.loc[summary["strategy"] == "no_initiation_by_180", "conditional_rmst_day180_to_day910"].iloc[0])
    return summary, init - noinit


def weighted_landmark_km(
    cohort: pd.DataFrame,
    treatment_col: str,
    event_col: str,
    time_col: str,
    weight_col: str,
    horizon: float,
) -> tuple[pd.DataFrame, float]:
    original_time = pd.to_numeric(cohort[time_col], errors="coerce")
    rows = pd.DataFrame(
        {
            "strategy_raw": pd.to_numeric(cohort[treatment_col], errors="coerce"),
            "weight": pd.to_numeric(cohort[weight_col], errors="coerce"),
            "event": pd.to_numeric(cohort[event_col], errors="coerce").fillna(0).astype(int),
            "start": 0.0,
            "stop": np.minimum(original_time, horizon),
            "clone_id": np.arange(len(cohort)).astype(str),
        }
    )
    rows.loc[original_time > horizon, "event"] = 0
    rows = rows.dropna()
    summaries = []
    for raw in sorted(rows["strategy_raw"].unique()):
        _, stats = weighted_counting_process_km(rows, raw, horizon)
        summaries.append(
            {
                "treatment": int(raw),
                "rmst_days": stats["rmst"],
                "survival_horizon": stats["survival_horizon"],
                "weight_mean": stats["weight_mean"],
                "weight_p99": stats["weight_p99"],
            }
        )
    summary = pd.DataFrame(summaries)
    treated = float(summary.loc[summary["treatment"] == 1, "rmst_days"].iloc[0])
    control = float(summary.loc[summary["treatment"] == 0, "rmst_days"].iloc[0])
    return summary, treated - control


def capture_long_from_call(function, args, kwargs) -> tuple[Any, pd.DataFrame]:
    captured: list[pd.DataFrame] = []
    code = getattr(function, "__code__", None)
    if code is None:
        raise TypeError("Function has no Python code object.")

    def tracer(frame, event, arg):
        if frame.f_code is code and event == "return":
            for value in frame.f_locals.values():
                if not isinstance(value, pd.DataFrame) or len(value) < 1000:
                    continue
                columns = [str(c).lower() for c in value.columns]
                if (
                    any("strategy" in c for c in columns)
                    and any("weight" in c for c in columns)
                    and any("event" in c for c in columns)
                ):
                    captured.append(value.copy())
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = function(*args, **kwargs)
    finally:
        sys.settrace(previous)
    if not captured:
        raise RuntimeError("Could not capture clone-level DataFrame from ccw_estimate.")
    return result, max(captured, key=len)


def capped_ccw_summary(long: pd.DataFrame, cap_spec: dict, horizon: float) -> tuple[dict[str, float], pd.DataFrame]:
    detected = discover_curve_columns(long)
    rows = prepare_survival_rows(long, detected, cap=None)
    if cap_spec["type"] == "fixed":
        cap = float(cap_spec["value"])
    elif cap_spec["type"] == "quantile":
        cap = float(rows["weight"].quantile(float(cap_spec["value"])))
    else:
        raise ValueError(cap_spec)
    rows["weight"] = np.minimum(rows["weight"], cap)
    mapping = orient_strategy(rows)
    outputs = {}
    for raw in rows["strategy_raw"].dropna().unique():
        _, stats = weighted_counting_process_km(rows, raw, horizon)
        outputs[mapping[raw]] = stats
    init = outputs["initiate_by_180"]
    noinit = outputs["no_initiation_by_180"]
    return {
        "cap_value": cap,
        "estimate_days": init["rmst"] - noinit["rmst"],
        "rmst_initiate": init["rmst"],
        "rmst_noinit": noinit["rmst"],
        "survival_initiate": init["survival_horizon"],
        "survival_noinit": noinit["survival_horizon"],
    }, rows


def update_ccw_result(result: Any, summary: dict[str, float], rows: pd.DataFrame) -> Any:
    updates = {
        "estimate_days": summary["estimate_days"],
        "rmst_initiate_by_180": summary["rmst_initiate"],
        "rmst_no_initiation_by_180": summary["rmst_noinit"],
        "survival_initiate": summary["survival_initiate"],
        "survival_no_initiation": summary["survival_noinit"],
        "weight_mean": float(rows["weight"].mean()),
        "weight_p95": float(rows["weight"].quantile(0.95)),
        "weight_p99": float(rows["weight"].quantile(0.99)),
        "weight_max": float(rows["weight"].max()),
        "fraction_weight_gt5": float((rows["weight"] > 5).mean()),
        "clone_rows": len(rows),
    }

    def apply(obj):
        if isinstance(obj, dict):
            out = dict(obj)
            out.update(updates)
            return out
        if isinstance(obj, pd.Series):
            out = obj.copy()
            for key, value in updates.items():
                out.loc[key] = value
            return out
        if isinstance(obj, tuple):
            return tuple(apply(item) if isinstance(item, (dict, pd.Series)) else item for item in obj)
        return obj

    return apply(result)


def stage43_files(root: Path) -> list[Path]:
    return sorted((root / "results/tables").glob("43_ccw_bootstrap*"))


@contextmanager
def preserve_stage43_outputs(root: Path):
    backup_dir = Path(tempfile.mkdtemp(prefix="stage15_stage43_", dir=str(root / "results/logs")))
    for path in stage43_files(root):
        shutil.move(str(path), str(backup_dir / path.name))
    try:
        yield
    finally:
        for path in stage43_files(root):
            path.unlink(missing_ok=True)
        for path in backup_dir.glob("*"):
            shutil.move(str(path), str(root / "results/tables" / path.name))
        shutil.rmtree(backup_dir, ignore_errors=True)


def run_stage43_with_cap(root: Path, cap_spec: dict, target_reps: int) -> list[Path]:
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    utils = importlib.import_module("_stage12_utils")
    if not hasattr(utils, "ccw_estimate"):
        raise AttributeError("_stage12_utils.ccw_estimate not found.")
    original = utils.ccw_estimate

    def wrapper(*args, **kwargs):
        result, long = capture_long_from_call(original, args, kwargs)
        summary, capped_rows = capped_ccw_summary(long, cap_spec, horizon=910.0)
        return update_ccw_result(result, summary, capped_rows)

    utils.ccw_estimate = wrapper
    sys.modules["_stage12_utils"] = utils
    env_name = "STAGE12_CCW_REPS"
    old_env = os.environ.get(env_name)
    os.environ[env_name] = str(target_reps)
    generated: list[Path] = []
    try:
        with preserve_stage43_outputs(root):
            old_argv = sys.argv[:]
            try:
                sys.argv = [str(root / "scripts/43_ccw_sensitivity_bootstrap.py")]
                try:
                    runpy.run_path(str(root / "scripts/43_ccw_sensitivity_bootstrap.py"), run_name="__main__")
                except SystemExit as exc:
                    if int(exc.code or 0) != 0:
                        raise RuntimeError(f"Stage 43 cap run exited with code {exc.code}")
            finally:
                sys.argv = old_argv
            outputs = stage43_files(root)
            if not outputs:
                raise RuntimeError("Stage 43 cap run produced no files.")
            for source in outputs:
                suffix = source.name.replace("43_ccw_bootstrap", "")
                target = root / "results/tables" / f"58_ccw_{cap_spec['name']}{suffix}"
                shutil.copy2(source, target)
                generated.append(target)
    finally:
        utils.ccw_estimate = original
        if old_env is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_env
    return generated
