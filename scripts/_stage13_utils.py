#!/usr/bin/env python3
"""Shared helpers for Stage 13.

Stage 13 deliberately does not overwrite Stage 11 or Stage 12 source data. It audits
existing outputs, extends the checkpointed pilot only through the existing Stage 12
bootstrap scripts, and writes new results under results/tables.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "stage13_config.json"


def load_config(root: Path | None = None) -> dict:
    path = config_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Stage 13 config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dirs(root: Path) -> None:
    for rel in ("results/tables", "results/logs", "data/derived/manifests"):
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_inventory(root: Path) -> pd.DataFrame:
    rows = []
    tables = root / "results" / "tables"
    if not tables.exists():
        return pd.DataFrame(columns=["path", "rows", "columns", "column_names", "sha256"])
    for path in sorted(tables.glob("*.csv")):
        try:
            df = read_csv(path)
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": "|".join(map(str, df.columns)),
                    "sha256": sha256_file(path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": np.nan,
                    "columns": np.nan,
                    "column_names": "",
                    "sha256": sha256_file(path),
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def find_csv_by_columns(
    root: Path,
    required: Sequence[str],
    preferred_tokens: Sequence[str] = (),
    excluded_tokens: Sequence[str] = (),
) -> Path | None:
    required_set = set(required)
    candidates: list[tuple[float, Path]] = []
    for path in sorted((root / "results" / "tables").glob("*.csv")):
        name = path.name.lower()
        if any(token.lower() in name for token in excluded_tokens):
            continue
        try:
            cols = set(read_csv(path).columns)
        except Exception:
            continue
        if not required_set.issubset(cols):
            continue
        score = 10.0 * sum(token.lower() in name for token in preferred_tokens)
        score -= 2.0 * sum(token.lower() in name for token in excluded_tokens)
        score += 0.001 * path.stat().st_mtime
        candidates.append((score, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_exact_or_columns(
    root: Path,
    exact_names: Sequence[str],
    required: Sequence[str],
    preferred_tokens: Sequence[str] = (),
) -> Path | None:
    for name in exact_names:
        path = root / "results" / "tables" / name
        if path.exists():
            return path
    return find_csv_by_columns(root, required, preferred_tokens)


def first_row(path: Path) -> pd.Series:
    df = read_csv(path)
    if df.empty:
        raise ValueError(f"Expected at least one row: {path}")
    return df.iloc[0]


def numeric(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def intish(value: object, default: int = -1) -> int:
    value = numeric(value)
    if not np.isfinite(value):
        return default
    return int(round(value))


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(
            lambda x: "" if pd.isna(x) else (f"{x:.6g}" if isinstance(x, float) else str(x))
        )
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines)


def select_estimate_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "estimate_days",
        "aipw_ato_rmst_difference_days",
        "rmst_difference_days",
        "bootstrap_estimate_days",
        "effect_days",
        "estimate",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and "day" in col.lower() and "weight" not in col.lower()
    ]
    return numeric_cols[0] if numeric_cols else None


@dataclass
class BootstrapStats:
    target_reps: int
    successful_reps: int
    mean: float
    sd: float
    median: float
    ci_low: float
    ci_high: float
    fraction_positive: float
    source_path: str


def bootstrap_stats_from_summary_or_checkpoint(
    root: Path,
    kind: str,
) -> BootstrapStats:
    if kind not in {"landmark", "ccw"}:
        raise ValueError(kind)

    if kind == "landmark":
        required = ["target_reps", "successful_reps", "bootstrap_mean_days", "bootstrap_sd_days"]
        preferred = ("42", "landmark", "summary")
        exact = (
            "42_landmark_bootstrap_summary.csv",
            "42_landmark_full_pipeline_bootstrap_summary.csv",
        )
        checkpoint = root / "results" / "tables" / "42_landmark_bootstrap_CHECKPOINT.csv"
    else:
        required = ["target_reps", "successful_reps", "bootstrap_mean_days", "bootstrap_sd_days"]
        preferred = ("43", "ccw", "summary")
        exact = (
            "43_ccw_bootstrap_summary.csv",
            "43_ccw_sensitivity_bootstrap_summary.csv",
        )
        checkpoint = root / "results" / "tables" / "43_ccw_bootstrap_CHECKPOINT.csv"

    path = find_exact_or_columns(root, exact, required, preferred)
    if path is not None:
        row = first_row(path)
        return BootstrapStats(
            target_reps=intish(row.get("target_reps")),
            successful_reps=intish(row.get("successful_reps")),
            mean=numeric(row.get("bootstrap_mean_days")),
            sd=numeric(row.get("bootstrap_sd_days")),
            median=numeric(row.get("median_days")),
            ci_low=numeric(row.get("percentile_ci_low_days")),
            ci_high=numeric(row.get("percentile_ci_high_days")),
            fraction_positive=numeric(row.get("fraction_positive")),
            source_path=str(path.relative_to(root)),
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Could not find a {kind} bootstrap summary or checkpoint under results/tables."
        )
    df = read_csv(checkpoint)
    estimate_col = select_estimate_column(df)
    if estimate_col is None:
        raise KeyError(f"No estimate column found in checkpoint: {checkpoint}")
    estimates = pd.to_numeric(df[estimate_col], errors="coerce").dropna()
    if estimates.empty:
        raise ValueError(f"No successful numeric estimates in checkpoint: {checkpoint}")
    return BootstrapStats(
        target_reps=len(df),
        successful_reps=len(estimates),
        mean=float(estimates.mean()),
        sd=float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
        median=float(estimates.median()),
        ci_low=float(estimates.quantile(0.025)),
        ci_high=float(estimates.quantile(0.975)),
        fraction_positive=float((estimates > 0).mean()),
        source_path=str(checkpoint.relative_to(root)),
    )


def find_point_rows(root: Path) -> tuple[pd.Series, pd.Series, Path, Path]:
    landmark_path = find_exact_or_columns(
        root,
        (
            "41_landmark_replication.csv",
            "41_landmark_replication_summary.csv",
        ),
        ["candidate_effect_days", "estimate_days", "replication_status"],
        ("41", "landmark", "replication"),
    )
    ccw_path = find_exact_or_columns(
        root,
        (
            "41_ccw_point_estimate.csv",
            "41_ccw_sensitivity_point_estimate.csv",
        ),
        ["estimate_days", "rmst_initiate_by_180", "rmst_no_initiation_by_180"],
        ("41", "ccw", "point"),
    )
    if landmark_path is None or ccw_path is None:
        # Some implementations save both rows in one file.
        combined = find_csv_by_columns(root, ["analysis", "estimate_days"], ("41", "replicate"))
        if combined is not None:
            df = read_csv(combined)
            if landmark_path is None:
                cand = df[df["analysis"].astype(str).str.contains("landmark", case=False, na=False)]
                if not cand.empty and "candidate_effect_days" in cand.columns:
                    landmark_path = combined
                    landmark_row = cand.iloc[0]
                else:
                    landmark_row = None
            else:
                landmark_row = first_row(landmark_path)
            if ccw_path is None:
                cand = df[df["analysis"].astype(str).str.contains("ccw", case=False, na=False)]
                if not cand.empty and {"rmst_initiate_by_180", "rmst_no_initiation_by_180"}.issubset(cand.columns):
                    ccw_path = combined
                    ccw_row = cand.iloc[0]
                else:
                    ccw_row = None
            else:
                ccw_row = first_row(ccw_path)
            if landmark_path is not None and ccw_path is not None and landmark_row is not None and ccw_row is not None:
                return landmark_row, ccw_row, landmark_path, ccw_path

    if landmark_path is None:
        raise FileNotFoundError("Stage 41 landmark replication table was not found.")
    if ccw_path is None:
        raise FileNotFoundError("Stage 41 CCW point-estimate table was not found.")
    return first_row(landmark_path), first_row(ccw_path), landmark_path, ccw_path


def discover_bootstrap_env_names(root: Path) -> dict[str, str | None]:
    names: set[str] = set()
    for path in sorted(root.glob("run_stage12*.ps1")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        names.update(re.findall(r"\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=", text))
    for script_name in ("42_landmark_full_pipeline_bootstrap.py", "43_ccw_sensitivity_bootstrap.py"):
        path = root / "scripts" / script_name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            names.update(
                re.findall(
                    r"(?:os\.getenv|os\.environ\.get)\(\s*[\"']([^\"']+)[\"']",
                    text,
                )
            )

    def choose(kind: str) -> str | None:
        scored = []
        for name in names:
            upper = name.upper()
            score = 0
            if kind.upper() in upper:
                score += 10
            if "BOOT" in upper:
                score += 5
            if "REP" in upper:
                score += 5
            if "TARGET" in upper:
                score += 2
            if score:
                scored.append((score, name))
        return max(scored)[1] if scored else None

    return {"landmark": choose("LANDMARK"), "ccw": choose("CCW")}


def run_python_script(root: Path, relative: str, env: dict[str, str]) -> None:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(path)
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(root),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{relative} failed with exit code {completed.returncode}")


def value_from_quantity_table(df: pd.DataFrame, label: str) -> float:
    if not {"quantity", "n"}.issubset(df.columns):
        return float("nan")
    matches = df[df["quantity"].astype(str).str.lower() == label.lower()]
    if matches.empty:
        return float("nan")
    return numeric(matches.iloc[0]["n"])


def find_ccw_curve_table(root: Path) -> tuple[Path | None, dict[str, str]]:
    candidates = []
    for path in sorted((root / "results").rglob("*.csv")):
        name = path.name.lower()
        if "ccw" not in name or not any(token in name for token in ("curve", "survival", "km")):
            continue
        try:
            df = read_csv(path)
        except Exception:
            continue
        lower = {str(c).lower(): str(c) for c in df.columns}
        time_col = next((lower[k] for k in lower if k in {"time", "day", "time_days", "timeline"}), None)
        survival_col = next((lower[k] for k in lower if "survival" in k), None)
        strategy_col = next((lower[k] for k in lower if k in {"strategy", "arm", "treatment_strategy"}), None)
        if time_col and survival_col and strategy_col:
            candidates.append((path, {"time": time_col, "survival": survival_col, "strategy": strategy_col}))
    if not candidates:
        return None, {}
    return candidates[0]


def step_survival_at(times: np.ndarray, surv: np.ndarray, t: float) -> float:
    order = np.argsort(times)
    times, surv = times[order], surv[order]
    idx = np.searchsorted(times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(surv[idx])


def conditional_rmst_from_curve(
    times: np.ndarray,
    surv: np.ndarray,
    start: float,
    end: float,
) -> tuple[float, float]:
    if end <= start:
        raise ValueError("end must exceed start")
    order = np.argsort(times)
    times = np.asarray(times, dtype=float)[order]
    surv = np.asarray(surv, dtype=float)[order]
    finite = np.isfinite(times) & np.isfinite(surv)
    times, surv = times[finite], surv[finite]
    if len(times) == 0:
        return float("nan"), float("nan")
    s_start = step_survival_at(times, surv, start)
    if s_start <= 0:
        return float("nan"), s_start
    knots = sorted(set([start, end] + [float(x) for x in times if start < x < end]))
    area = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        s_left = step_survival_at(times, surv, left) / s_start
        area += s_left * (right - left)
    return float(area), float(s_start)
