#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

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
from _stage18_utils import make_grouped_bootstrap_folds
from _stage16_utils import project_root


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage21_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return load_json(root / "stage21_config.json")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def empty_or_read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_console(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "<empty table>"
    view = df if max_rows is None else df.head(max_rows)
    with pd.option_context(
        "display.max_rows", None if max_rows is None else max_rows,
        "display.max_columns", None,
        "display.width", 340,
        "display.max_colwidth", 120,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        return view.to_string(index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
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


def ensure_stage21_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/logs",
        "data/derived/stage21",
        "data/derived/stage21/partition_scores",
        "data/derived/manifests",
        "paper_A_treatment_effects",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


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
    bcfg = cfg["full_bootstrap"]
    n = len(payload["frame"])
    bootstrap_seed = int(bcfg["bootstrap_base_seed"]) + int(repetition)
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


def fit_partition(
    sample: dict[str, object],
    repetition: int,
    partition_number: int,
    base_seed: int,
    cfg: dict,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    bcfg = cfg["full_bootstrap"]
    n_folds = int(bcfg["n_folds"])
    g_min = float(bcfg["primary_g_min"])
    horizon = float(bcfg["horizon_days"])
    interval = float(bcfg["interval_days"])
    maximum_retries = int(bcfg["maximum_partition_retries"])
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
            summary, patient = effect_and_patient_components(y, a, e, mu0, mu1)
            score_numerator = patient["score_numerator"].to_numpy(float)
            h = patient["h"].to_numpy(float)
            row = {
                "bootstrap_repetition": int(repetition),
                "bootstrap_seed": int(sample["bootstrap_seed"]),
                "partition": int(partition_number),
                "base_seed": int(base_seed),
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
            return row, score_numerator, h
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Bootstrap repetition {repetition}, partition {partition_number} failed "
        f"after {maximum_retries} retries: {last_error}"
    ) from last_error


def save_partition_score_file(
    path: Path,
    partition_numbers: np.ndarray,
    score_numerators: np.ndarray,
    h_values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            partition_numbers=np.asarray(partition_numbers, dtype=int),
            score_numerators=np.asarray(score_numerators, dtype=float),
            h_values=np.asarray(h_values, dtype=float),
        )
    os.replace(temp, path)


def load_partition_score_file(path: Path, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        return (
            np.empty(0, dtype=int),
            np.empty((0, n_rows), dtype=float),
            np.empty((0, n_rows), dtype=float),
        )
    with np.load(path) as data:
        partitions = np.asarray(data["partition_numbers"], dtype=int)
        numerators = np.asarray(data["score_numerators"], dtype=float)
        h_values = np.asarray(data["h_values"], dtype=float)
    if numerators.shape != h_values.shape or numerators.shape[1:] != (n_rows,):
        raise RuntimeError(f"Invalid partition score checkpoint shape in {path}")
    if len(partitions) != numerators.shape[0]:
        raise RuntimeError(f"Partition number count differs from score array in {path}")
    return partitions, numerators, h_values


def append_partition_scores(
    path: Path,
    partition_number: int,
    score_numerator: np.ndarray,
    h: np.ndarray,
) -> None:
    n = len(score_numerator)
    partitions, numerators, h_values = load_partition_score_file(path, n)
    keep = partitions != int(partition_number)
    partitions = partitions[keep]
    numerators = numerators[keep]
    h_values = h_values[keep]
    partitions = np.concatenate([partitions, np.array([int(partition_number)], dtype=int)])
    numerators = np.vstack([numerators, np.asarray(score_numerator, dtype=float).reshape(1, n)])
    h_values = np.vstack([h_values, np.asarray(h, dtype=float).reshape(1, n)])
    order = np.argsort(partitions)
    save_partition_score_file(path, partitions[order], numerators[order], h_values[order])


def aggregate_score_arrays(
    partition_numbers: np.ndarray,
    score_numerators: np.ndarray,
    h_values: np.ndarray,
    expected_partitions: Iterable[int],
) -> dict[str, float]:
    expected = np.asarray(list(expected_partitions), dtype=int)
    if not np.array_equal(np.sort(partition_numbers), np.sort(expected)):
        raise RuntimeError(
            f"Partition score checkpoint is incomplete: observed={sorted(partition_numbers.tolist())}; "
            f"expected={sorted(expected.tolist())}"
        )
    mean_num = np.mean(score_numerators, axis=0)
    mean_h = np.mean(h_values, axis=0)
    denominator = float(np.sum(mean_h))
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Invalid repeated-score denominator.")
    theta = float(np.sum(mean_num) / denominator)
    hbar = float(np.mean(mean_h))
    influence = (mean_num - theta * mean_h) / hbar
    se = float(np.std(influence, ddof=1) / np.sqrt(len(mean_num)))
    return {
        "estimate_days": theta,
        "if_se_days": se,
        "if_ci_low_days": theta - 1.96 * se,
        "if_ci_high_days": theta + 1.96 * se,
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


def lock_manifest_path(root: Path) -> Path:
    return root / "data/derived/manifests/80_candidate_v9_protocol_lock_manifest.json"


def verify_locked_files(root: Path) -> pd.DataFrame:
    manifest_path = lock_manifest_path(root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Candidate V9 lock manifest is missing. Run run_stage20_candidate_v9_protocol_lock.ps1 first."
        )
    manifest = load_json(manifest_path)
    rows: list[dict] = []
    for item in manifest["locked_files"]:
        path = root / item["path"]
        observed = sha256_file(path) if path.exists() else None
        rows.append({
            "path": item["path"],
            "expected_sha256": item["sha256"],
            "observed_sha256": observed,
            "exists": path.exists(),
            "match": observed == item["sha256"],
        })
    result = pd.DataFrame(rows)
    if result.empty or not bool(result["match"].all()):
        raise RuntimeError("Candidate V9 lock integrity failed.\n" + dataframe_console(result[result["match"] == False]))
    return result


def checkpoint_identity(root: Path, cfg: dict) -> dict[str, object]:
    manifest_path = lock_manifest_path(root)
    paths = {
        "lock_manifest": manifest_path,
        "stage21_config": root / "stage21_config.json",
        "stage21_utils": root / "scripts/_stage21_utils.py",
        "stage21_bootstrap": root / "scripts/82_full_publication_bootstrap.py",
        "stage21_summary": root / "scripts/83_summarize_publication_bootstrap.py",
        "stage21_decision": root / "scripts/84_generate_publication_decision.py",
        "final_point": root / "results/tables/79_candidate_v9_final_point_estimate.csv",
    }
    return {
        "stage": 21,
        "protocol_status": cfg["protocol_status"],
        "full_bootstrap": cfg["full_bootstrap"],
        "inference": cfg["inference"],
        "input_hashes": {
            name: sha256_file(path) if path.exists() else None
            for name, path in paths.items()
        },
    }


def validate_or_create_checkpoint_identity(path: Path, identity: dict[str, object]) -> None:
    if path.exists():
        old = load_json(path)
        if old != identity:
            raise RuntimeError(
                "Stage 21 checkpoint identity differs from the locked code/config/inputs. "
                "Do not resume with modified files."
            )
    else:
        write_json(identity, path)
