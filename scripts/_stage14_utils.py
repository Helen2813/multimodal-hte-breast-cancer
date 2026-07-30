#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "stage14_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/figures",
        "results/logs",
        "data/derived/stage14_trace",
        "data/derived/manifests",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def numeric(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(
            lambda value: ""
            if pd.isna(value)
            else (f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        )
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[h]).replace("|", "\\|") for h in headers)
            + " |"
        )
    return "\n".join(lines)


def find_first(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def find_csv_by_columns(
    root: Path,
    required: Sequence[str],
    preferred_tokens: Sequence[str] = (),
) -> Path | None:
    required = set(required)
    scored = []
    for path in sorted((root / "results" / "tables").glob("*.csv")):
        try:
            columns = set(read_csv(path).columns)
        except Exception:
            continue
        if not required.issubset(columns):
            continue
        name = path.name.lower()
        score = 10 * sum(token.lower() in name for token in preferred_tokens)
        score += path.stat().st_mtime * 1e-9
        scored.append((score, path))
    return max(scored)[1] if scored else None


def point_estimate_table(root: Path) -> tuple[Path, pd.Series]:
    path = find_first(
        root,
        (
            "results/tables/41_ccw_point_estimate.csv",
            "results/tables/41_ccw_sensitivity_point_estimate.csv",
        ),
    )
    if path is None:
        path = find_csv_by_columns(
            root,
            ["estimate_days", "rmst_initiate_by_180", "rmst_no_initiation_by_180"],
            ("41", "ccw", "point"),
        )
    if path is None:
        raise FileNotFoundError("Could not find Stage 41 CCW point-estimate table.")
    df = read_csv(path)
    if df.empty:
        raise ValueError(f"Empty point-estimate table: {path}")
    return path, df.iloc[0]


def checkpoint_table(root: Path, kind: str) -> tuple[Path, pd.DataFrame]:
    if kind == "ccw":
        path = root / "results/tables/43_ccw_bootstrap_CHECKPOINT.csv"
    elif kind == "landmark":
        path = root / "results/tables/42_landmark_bootstrap_CHECKPOINT.csv"
    else:
        raise ValueError(kind)
    if not path.exists():
        raise FileNotFoundError(path)
    return path, read_csv(path)


def select_estimate_column(df: pd.DataFrame) -> str:
    for col in (
        "estimate_days",
        "bootstrap_estimate_days",
        "rmst_difference_days",
        "aipw_ato_rmst_difference_days",
        "effect_days",
        "estimate",
    ):
        if col in df.columns:
            return col
    candidates = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and any(token in col.lower() for token in ("estimate", "effect", "difference"))
    ]
    if not candidates:
        raise KeyError("No estimate column found.")
    return candidates[0]


def choose_column(
    columns: Sequence[str],
    exact: Sequence[str],
    contains: Sequence[str],
    exclude: Sequence[str] = (),
) -> str | None:
    cols = list(map(str, columns))
    lower = {col.lower(): col for col in cols}
    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]
    ranked = []
    for col in cols:
        name = col.lower()
        if any(token in name for token in exclude):
            continue
        score = sum(token in name for token in contains)
        if score:
            ranked.append((score, -len(name), col))
    return max(ranked)[2] if ranked else None


def dataframe_candidate_score(df: pd.DataFrame) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return -1.0
    cols = [str(c).lower() for c in df.columns]
    score = 0.0
    for token, points in (
        ("strategy", 8),
        ("clone", 7),
        ("weight", 6),
        ("event", 5),
        ("stop", 5),
        ("start", 4),
        ("time", 3),
        ("interval", 3),
        ("censor", 2),
    ):
        score += points * sum(token in col for col in cols)
    if len(df) >= 1000:
        score += 8
    if len(df) >= 10000:
        score += 8
    if 2 <= len(df.columns) <= 100:
        score += 2
    return score


def discover_curve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(map(str, df.columns))
    strategy = choose_column(
        columns,
        ("strategy", "clone_strategy", "treatment_strategy", "assigned_strategy", "arm"),
        ("strategy", "clone_arm", "assigned", "arm"),
    )
    weight = choose_column(
        columns,
        (
            "final_weight",
            "combined_weight",
            "stabilized_weight",
            "ipcw_weight",
            "adherence_weight",
            "weight",
        ),
        ("final_weight", "combined", "stabilized", "ipcw", "adherence", "weight"),
        ("numerator", "denominator", "raw_probability"),
    )
    event = choose_column(
        columns,
        ("event", "analysis_event", "event_at_end", "natural_event", "failure"),
        ("event", "failure", "death"),
        ("censor", "count"),
    )
    start = choose_column(
        columns,
        ("start", "start_day", "tstart", "interval_start", "time_start"),
        ("start", "tstart", "interval_begin"),
        ("treatment_start", "days_to_treatment"),
    )
    stop = choose_column(
        columns,
        ("stop", "stop_day", "tstop", "interval_end", "time_stop"),
        ("stop", "tstop", "interval_end"),
        ("treatment_stop",),
    )
    time = choose_column(
        columns,
        ("time", "time_days", "duration", "followup_days", "analysis_time"),
        ("duration", "followup", "time_days", "analysis_time"),
        ("start", "stop", "treatment", "calendar"),
    )
    clone_id = choose_column(
        columns,
        ("clone_id", "patient_clone_id", "id_clone"),
        ("clone_id", "clone"),
        ("strategy",),
    )
    patient_id = choose_column(
        columns,
        ("patient_id", "normalized_patient_id", "submitter_id"),
        ("patient_id", "submitter"),
    )
    return {
        "strategy": strategy,
        "weight": weight,
        "event": event,
        "start": start,
        "stop": stop,
        "time": time,
        "clone_id": clone_id,
        "patient_id": patient_id,
    }


def normalize_binary_event(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().mean() >= 0.8:
        return (numeric_values > 0).astype(int)
    text = series.astype(str).str.lower().str.strip()
    return text.isin({"1", "true", "yes", "event", "death", "dead", "failure"}).astype(int)


def strategy_orientation(values: pd.Series) -> dict[object, str] | None:
    unique = [value for value in values.dropna().unique()]
    if len(unique) != 2:
        return None
    mapping = {}
    for value in unique:
        text = str(value).lower()
        if any(token in text for token in ("no_init", "no init", "not_init", "no treatment", "control")):
            mapping[value] = "no_initiation_by_180"
        elif any(token in text for token in ("initiate", "initiation", "treated", "early")):
            mapping[value] = "initiate_by_180"
    if len(mapping) == 2 and len(set(mapping.values())) == 2:
        return mapping
    numeric_unique = pd.to_numeric(pd.Series(unique), errors="coerce")
    if numeric_unique.notna().all() and set(numeric_unique.astype(float)) == {0.0, 1.0}:
        for value in unique:
            mapping[value] = "initiate_by_180" if float(value) == 1.0 else "no_initiation_by_180"
        return mapping
    return None


def prepare_survival_rows(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    cap: float | None = None,
) -> pd.DataFrame:
    required = ("strategy", "weight", "event")
    missing = [key for key in required if not columns.get(key)]
    if missing:
        raise KeyError(f"Missing curve columns: {missing}")
    if not columns.get("stop") and not columns.get("time"):
        raise KeyError("Neither stop nor time column was detected.")

    out = pd.DataFrame()
    out["strategy_raw"] = df[columns["strategy"]]
    out["weight"] = pd.to_numeric(df[columns["weight"]], errors="coerce")
    out["event"] = normalize_binary_event(df[columns["event"]])
    if columns.get("start"):
        out["start"] = pd.to_numeric(df[columns["start"]], errors="coerce")
    else:
        out["start"] = 0.0
    source_time = columns["stop"] or columns["time"]
    out["stop"] = pd.to_numeric(df[source_time], errors="coerce")
    if columns.get("clone_id"):
        out["clone_id"] = df[columns["clone_id"]].astype(str)
    elif columns.get("patient_id"):
        out["clone_id"] = (
            df[columns["patient_id"]].astype(str) + "::" + df[columns["strategy"]].astype(str)
        )
    else:
        out["clone_id"] = np.arange(len(out)).astype(str)

    valid = (
        out["weight"].notna()
        & np.isfinite(out["weight"])
        & (out["weight"] > 0)
        & out["stop"].notna()
        & np.isfinite(out["stop"])
        & out["start"].notna()
        & np.isfinite(out["start"])
        & (out["stop"] >= out["start"])
    )
    out = out.loc[valid].copy()
    if cap is not None:
        out["weight"] = np.minimum(out["weight"], float(cap))
    return out


def weighted_counting_process_km(
    rows: pd.DataFrame,
    strategy_value: object,
    horizon: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    part = rows[rows["strategy_raw"] == strategy_value].copy()
    if part.empty:
        raise ValueError(f"No rows for strategy={strategy_value!r}")

    event_times = sorted(
        float(value)
        for value in part.loc[(part["event"] == 1) & (part["stop"] <= horizon), "stop"].unique()
        if np.isfinite(value) and value >= 0
    )
    survival = 1.0
    curve = [{"time": 0.0, "survival": 1.0, "weighted_risk": float(part.loc[part["start"] <= 0, "weight"].sum()), "weighted_events": 0.0}]
    last_time = 0.0
    rmst = 0.0

    for time in event_times:
        if time < last_time:
            continue
        rmst += survival * (time - last_time)
        risk_mask = (part["start"] < time) & (part["stop"] >= time)
        event_mask = (part["stop"] == time) & (part["event"] == 1)
        risk_weight = float(part.loc[risk_mask, "weight"].sum())
        event_weight = float(part.loc[event_mask, "weight"].sum())
        hazard = 0.0 if risk_weight <= 0 else min(max(event_weight / risk_weight, 0.0), 1.0)
        survival *= 1.0 - hazard
        curve.append(
            {
                "time": float(time),
                "survival": float(survival),
                "weighted_risk": risk_weight,
                "weighted_events": event_weight,
            }
        )
        last_time = time

    if last_time < horizon:
        rmst += survival * (horizon - last_time)

    curve_df = pd.DataFrame(curve)
    survival_horizon = step_value(curve_df["time"].to_numpy(), curve_df["survival"].to_numpy(), horizon)
    stats = {
        "rmst": float(rmst),
        "survival_horizon": float(survival_horizon),
        "n_rows": len(part),
        "n_clones": part["clone_id"].nunique(),
        "event_rows": int(part["event"].sum()),
        "weight_mean": float(part["weight"].mean()),
        "weight_p95": float(part["weight"].quantile(0.95)),
        "weight_p99": float(part["weight"].quantile(0.99)),
        "weight_max": float(part["weight"].max()),
    }
    return curve_df, stats


def step_value(times: np.ndarray, values: np.ndarray, at: float) -> float:
    order = np.argsort(times)
    times = np.asarray(times, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    index = np.searchsorted(times, at, side="right") - 1
    return 1.0 if index < 0 else float(values[index])


def integrate_step_curve(
    curve: pd.DataFrame,
    start: float,
    end: float,
    conditional: bool = False,
) -> tuple[float, float]:
    times = pd.to_numeric(curve["time"], errors="coerce").to_numpy()
    survival = pd.to_numeric(curve["survival"], errors="coerce").to_numpy()
    valid = np.isfinite(times) & np.isfinite(survival)
    times, survival = times[valid], survival[valid]
    s_start = step_value(times, survival, start)
    if conditional and s_start <= 0:
        return float("nan"), s_start
    knots = sorted(set([float(start), float(end)] + [float(t) for t in times if start < t < end]))
    area = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        value = step_value(times, survival, left)
        if conditional:
            value /= s_start
        area += value * (right - left)
    return float(area), float(s_start)


def detect_checkpoint_weight_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = list(map(str, df.columns))
    return {
        "max": choose_column(
            cols,
            ("weight_max", "max_weight", "bootstrap_weight_max"),
            ("weight_max", "max_weight"),
        ),
        "p99": choose_column(
            cols,
            ("weight_p99", "p99_weight", "bootstrap_weight_p99"),
            ("weight_p99", "p99_weight"),
        ),
    }
