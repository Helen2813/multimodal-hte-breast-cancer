#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _stage12_utils import (
    assemble_landmark_data,
    crossfit_arm_outcomes,
    crossfit_censor_survival,
    crossfit_propensity,
    ipcw_rmst_pseudo,
)
from _stage17_utils import effect_and_patient_components
from _stage18_utils import make_grouped_bootstrap_folds, project_root


def load_stage19_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "stage19_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_stage19_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/logs",
        "data/derived/stage19",
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


def dataframe_console(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "<empty table>"
    view = df if max_rows is None else df.head(max_rows)
    with pd.option_context(
        "display.max_rows", None if max_rows is None else max_rows,
        "display.max_columns", None,
        "display.width", 300,
        "display.max_colwidth", 120,
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


def bootstrap_payload() -> dict[str, object]:
    frame, features, _, metadata = assemble_landmark_data()
    frame = frame.copy().reset_index(drop=True)
    return {
        "frame": frame,
        "features": list(features),
        "a": pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy(),
        "event": pd.to_numeric(frame["analysis_event"], errors="raise").astype(int).to_numpy(),
        "observed_time": pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float),
        "metadata": metadata,
    }


def recreate_bootstrap_sample(payload: dict[str, object], repetition: int, cfg: dict) -> dict[str, object]:
    ext = cfg["extension"]
    n = len(payload["frame"])
    bootstrap_seed = int(ext["bootstrap_base_seed"]) + int(repetition)
    rng = np.random.default_rng(bootstrap_seed)
    sampled_original_rows = rng.integers(0, n, size=n, endpoint=False)
    return {
        "frame": payload["frame"].iloc[sampled_original_rows].copy().reset_index(drop=True),
        "features": list(payload["features"]),
        "a": np.asarray(payload["a"], dtype=int)[sampled_original_rows],
        "event": np.asarray(payload["event"], dtype=int)[sampled_original_rows],
        "observed_time": np.asarray(payload["observed_time"], dtype=float)[sampled_original_rows],
        "original_groups": sampled_original_rows.astype(int),
        "bootstrap_seed": bootstrap_seed,
    }


def fit_partition_summary(
    sample: dict[str, object],
    repetition: int,
    partition_number: int,
    base_seed: int,
    cfg: dict,
) -> dict[str, float | int | str]:
    ext = cfg["extension"]
    n_folds = int(ext["n_folds"])
    g_min = float(ext["primary_g_min"])
    horizon = float(ext["horizon_days"])
    interval = float(ext["interval_days"])
    maximum_retries = int(ext["maximum_partition_retries"])
    nominal_seed = int(base_seed) + int(repetition) * 1_000_000

    frame = sample["frame"]
    features = list(sample["features"])
    a = np.asarray(sample["a"], dtype=int)
    event = np.asarray(sample["event"], dtype=int)
    observed_time = np.asarray(sample["observed_time"], dtype=float)
    groups = np.asarray(sample["original_groups"], dtype=int)

    last_error: Exception | None = None
    for retry in range(maximum_retries):
        seed = nominal_seed + retry * 100_000
        try:
            fold, stratification, fold_retry = make_grouped_bootstrap_folds(
                a, event, groups, seed, n_folds
            )
            e = crossfit_propensity(frame, features, fold, "analysis_treatment", seed + 10)
            G, starts, ends, censor_metrics = crossfit_censor_survival(
                frame, features, fold, horizon, interval, seed + 100
            )
            y = ipcw_rmst_pseudo(observed_time, G, starts, ends, horizon, g_min)
            mu0_raw, mu1_raw = crossfit_arm_outcomes(frame, features, y, fold, seed + 200)
            mu0 = np.clip(mu0_raw, 0.0, horizon)
            mu1 = np.clip(mu1_raw, 0.0, horizon)
            summary, _ = effect_and_patient_components(y, a, e, mu0, mu1)
            return {
                "bootstrap_repetition": int(repetition),
                "bootstrap_seed": int(sample["bootstrap_seed"]),
                "partition": int(partition_number),
                "nominal_seed": int(nominal_seed),
                "seed_used": int(seed),
                "nuisance_retry": int(retry),
                "fold_retry": int(fold_retry),
                "fold_stratification": str(stratification),
                **summary,
                "propensity_min": float(np.min(e)),
                "propensity_p01": float(np.quantile(e, 0.01)),
                "propensity_p99": float(np.quantile(e, 0.99)),
                "propensity_max": float(np.max(e)),
                "censor_log_loss": float(censor_metrics["censor_log_loss"]),
                "censor_brier": float(censor_metrics["censor_brier"]),
                "G_min_raw": float(censor_metrics["G_min"]),
                "G_p01_raw": float(censor_metrics["G_p01"]),
                "pseudo_mean": float(np.mean(y)),
                "pseudo_sd": float(np.std(y, ddof=1)),
                "pseudo_p99": float(np.quantile(y, 0.99)),
                "pseudo_max": float(np.max(y)),
                "fraction_mu0_outside_before_bounding": float(
                    np.mean((mu0_raw < 0.0) | (mu0_raw > horizon))
                ),
                "fraction_mu1_outside_before_bounding": float(
                    np.mean((mu1_raw < 0.0) | (mu1_raw > horizon))
                ),
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Bootstrap repetition {repetition}, partition {partition_number} failed "
        f"after {maximum_retries} retries: {last_error}"
    ) from last_error


def denominator_weighted_effect(partitions: pd.DataFrame) -> float:
    required = {"estimate_days", "ato_denominator"}
    missing = required - set(partitions.columns)
    if missing:
        raise ValueError(f"Missing aggregation columns: {sorted(missing)}")
    theta = pd.to_numeric(partitions["estimate_days"], errors="raise").to_numpy(float)
    denominator = pd.to_numeric(partitions["ato_denominator"], errors="raise").to_numpy(float)
    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(denominator)) or denominator.sum() <= 0:
        raise ValueError("Invalid partition effects or ATO denominators.")
    return float(np.sum(theta * denominator) / np.sum(denominator))


def checkpoint_identity(root: Path, cfg: dict) -> dict[str, object]:
    paths = {
        "stage19_config": root / "stage19_config.json",
        "stage19_utils": root / "scripts/_stage19_utils.py",
        "stage19_preflight": root / "scripts/75_stage19_protocol_correction.py",
        "stage19_extension": root / "scripts/76_extend_bootstrap_partitions.py",
        "stage19_summary": root / "scripts/77_assess_inner_crossfit_convergence.py",
        "stage19_decision": root / "scripts/78_generate_stage19_decision.py",
        "stage18_config": root / "stage18_config.json",
        "stage18_repetitions": root / "results/tables/72_bootstrap_pilot_repetitions_checkpoint.csv",
        "stage18_partitions": root / "results/tables/72_bootstrap_pilot_partitions_checkpoint.csv",
    }
    return {
        "stage": 19,
        "protocol_status": cfg["protocol_status"],
        "extension": cfg["extension"],
        "input_hashes": {
            name: sha256_file(path) if path.exists() else None
            for name, path in paths.items()
        },
    }


def validate_or_create_identity(path: Path, identity: dict[str, object]) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != identity:
            raise RuntimeError(
                "Stage 19 checkpoint identity differs from the current code/config/Stage 18 inputs. "
                "Move old Stage 19 checkpoint files before continuing."
            )
    else:
        write_json(identity, path)
