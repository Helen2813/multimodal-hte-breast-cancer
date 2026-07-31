#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from _stage16_utils import project_root


def load_stage18_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "stage18_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_stage18_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/logs",
        "data/derived/stage18",
        "data/derived/manifests",
        "paper_A_treatment_effects",
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


def dataframe_console(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "<empty table>"
    view = df if max_rows is None else df.head(max_rows)
    with pd.option_context(
        "display.max_rows", None if max_rows is None else max_rows,
        "display.max_columns", None,
        "display.width", 280,
        "display.max_colwidth", 100,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        return view.to_string(index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
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
        lines.append("| " + " | ".join(str(row[h]).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines)


def make_grouped_bootstrap_folds(
    treatment: np.ndarray,
    event: np.ndarray,
    original_patient_group: np.ndarray,
    seed: int,
    n_folds: int,
) -> tuple[np.ndarray, str, int]:
    treatment = np.asarray(treatment, dtype=int)
    event = np.asarray(event, dtype=int)
    groups = np.asarray(original_patient_group, dtype=int)
    if not (len(treatment) == len(event) == len(groups)):
        raise ValueError("Treatment, event, and group lengths differ.")

    candidates = [
        (2 * treatment + event, "treatment_x_event_grouped"),
        (treatment, "treatment_only_grouped"),
    ]
    for strata, label in candidates:
        for attempt in range(100):
            splitter = StratifiedGroupKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=seed + attempt,
            )
            fold = np.full(len(strata), -1, dtype=int)
            valid = True
            for f, (train, test) in enumerate(
                splitter.split(np.zeros(len(strata)), strata, groups=groups), start=1
            ):
                fold[test] = f
                if len(np.unique(treatment[train])) < 2:
                    valid = False
                    break
                if int(np.sum(treatment[train] == 0)) < 20 or int(np.sum(treatment[train] == 1)) < 20:
                    valid = False
                    break
            if not valid or np.any(fold < 1):
                continue
            leakage = False
            for group in np.unique(groups):
                if len(np.unique(fold[groups == group])) != 1:
                    leakage = True
                    break
            if leakage:
                raise RuntimeError("Duplicate copies of an original patient crossed nuisance folds.")
            return fold, label, attempt
    raise RuntimeError("Could not create valid grouped bootstrap folds.")


def aggregate_partition_patient_scores(scores: pd.DataFrame) -> dict[str, object]:
    required = {"bootstrap_row_index", "score_numerator", "h"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    grouped = (
        scores.groupby("bootstrap_row_index", as_index=False)
        .agg(
            score_numerator=("score_numerator", "mean"),
            h=("h", "mean"),
            original_patient_group=("original_patient_group", "first"),
        )
        .sort_values("bootstrap_row_index")
        .reset_index(drop=True)
    )
    denominator = float(grouped["h"].sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid repeated-score denominator.")
    theta = float(grouped["score_numerator"].sum() / denominator)
    mean_h = float(grouped["h"].mean())
    influence = (
        grouped["score_numerator"].to_numpy(float)
        - theta * grouped["h"].to_numpy(float)
    ) / mean_h
    se = float(np.std(influence, ddof=1) / np.sqrt(len(grouped)))
    return {
        "estimate_days": theta,
        "if_se_days": se,
        "if_ci_low_days": theta - 1.96 * se,
        "if_ci_high_days": theta + 1.96 * se,
        "patient": grouped,
    }


def append_replace_repetition(existing: pd.DataFrame, new_rows: pd.DataFrame, repetition: int) -> pd.DataFrame:
    if not existing.empty and "bootstrap_repetition" in existing.columns:
        keep = pd.to_numeric(existing["bootstrap_repetition"], errors="coerce") != repetition
        existing = existing.loc[keep].copy()
    return pd.concat([existing, new_rows], ignore_index=True)


def finite_quantile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float("nan")


def checkpoint_identity(root: Path, config: dict) -> dict[str, object]:
    paths = {
        "stage18_config": root / "stage18_config.json",
        "stage18_utils": root / "scripts/_stage18_utils.py",
        "stage18_preflight": root / "scripts/71_stage18_protocol_amendment.py",
        "stage18_bootstrap": root / "scripts/72_patient_bootstrap_pilot.py",
        "stage18_summary": root / "scripts/73_summarize_bootstrap_pilot.py",
        "stage18_decision": root / "scripts/74_generate_stage18_decision.py",
        "stage12_utils": root / "scripts/_stage12_utils.py",
        "stage16_utils": root / "scripts/_stage16_utils.py",
        "stage17_decision": root / "results/tables/70_stage17_decision.csv",
        "stage17_checks": root / "results/tables/70_stage17_decision_checks.csv",
    }
    return {
        "stage": 18,
        "protocol_status": config["protocol_status"],
        "bootstrap_pilot": config["bootstrap_pilot"],
        "input_hashes": {
            name: sha256_file(path) if path.exists() else None
            for name, path in paths.items()
        },
    }


def validate_or_create_checkpoint_identity(path: Path, identity: dict[str, object]) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != identity:
            raise RuntimeError(
                "Stage 18 checkpoint identity differs from current code/config/inputs. "
                "Move the old Stage 18 checkpoint files before resuming."
            )
    else:
        write_json(identity, path)
