#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _stage12_utils import (
    BASE_SEED,
    G_MIN,
    LANDMARK_HORIZON,
    LANDMARK_INTERVAL,
    aipw_ato,
    crossfit_arm_outcomes,
    crossfit_censor_survival,
    crossfit_propensity,
    ipcw_rmst_pseudo,
)
from _stage16_utils import (
    aipw_components,
    exact_landmark_payload,
    project_root,
    subset_aipw_effect,
)


def load_stage17_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "stage17_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_stage17_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/figures",
        "results/logs",
        "data/derived/stage17",
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
        "display.width", 260,
        "display.max_colwidth", 90,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        return view.to_string(index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def stable_repeat_seeds(config: dict) -> list[int]:
    rcfg = config["repeated_crossfit"]
    base = int(rcfg["base_seed"])
    n = int(rcfg["n_repeats"])
    return [base + i for i in range(1, n + 1)]


def make_repeated_fold_assignment(
    treatment: np.ndarray,
    event: np.ndarray,
    seed: int,
    n_folds: int,
) -> tuple[np.ndarray, str]:
    treatment = np.asarray(treatment, dtype=int)
    event = np.asarray(event, dtype=int)
    if len(treatment) != len(event):
        raise ValueError("Treatment/event lengths differ.")

    candidates = [
        (2 * treatment + event, "treatment_x_event"),
        (treatment, "treatment_only"),
    ]
    for strata, label in candidates:
        counts = pd.Series(strata).value_counts()
        if counts.empty or int(counts.min()) < n_folds:
            continue
        for attempt in range(100):
            splitter = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=seed + attempt,
            )
            fold = np.full(len(strata), -1, dtype=int)
            valid = True
            for f, (_, test) in enumerate(splitter.split(np.zeros(len(strata)), strata), start=1):
                fold[test] = f
                if len(np.unique(treatment[test])) < 2:
                    valid = False
                    break
            if valid and np.all(fold > 0):
                return fold, label
    raise RuntimeError("Could not construct repeated folds with both treatment arms in every fold.")


def patient_censoring_diagnostics(
    observed_time: np.ndarray,
    G: np.ndarray,
    starts: np.ndarray,
    g_min: float,
) -> pd.DataFrame:
    observed_time = np.asarray(observed_time, dtype=float)
    G = np.asarray(G, dtype=float)
    starts = np.asarray(starts, dtype=float)
    rows: list[dict[str, float | int]] = []
    for i, t in enumerate(observed_time):
        mask = starts < min(float(t), float(LANDMARK_HORIZON))
        if not mask.any():
            mask = np.zeros(len(starts), dtype=bool)
            mask[0] = True
        values = np.clip(G[i, mask], 1e-8, 1.0)
        rows.append(
            {
                "n_at_risk_intervals": int(mask.sum()),
                "g_min_at_risk_raw": float(values.min()),
                "g_p10_at_risk_raw": float(np.quantile(values, 0.10)),
                "g_median_at_risk_raw": float(np.median(values)),
                "max_inverse_g_raw": float(np.max(1.0 / values)),
                "max_inverse_g_after_floor": float(
                    np.max(1.0 / np.clip(values, float(g_min), 1.0))
                ),
            }
        )
    return pd.DataFrame(rows)


def effect_and_patient_components(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    comp = aipw_components(y, a, e, mu0, mu1)
    patient = comp["patient"].copy()
    summary = dict(comp["summary"])
    influence = pd.to_numeric(patient["influence"], errors="raise").to_numpy(float)
    summary["if_se_days"] = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    summary["if_ci_low_days"] = summary["estimate_days"] - 1.96 * summary["if_se_days"]
    summary["if_ci_high_days"] = summary["estimate_days"] + 1.96 * summary["if_se_days"]
    return summary, patient


def original_fold_effects(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    fold: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, float | int]] = []
    loo_rows: list[dict[str, float | int]] = []
    for f in sorted(np.unique(fold)):
        test = fold == f
        keep = fold != f
        fold_rows.append(
            {
                "fold": int(f),
                "fold_n": int(test.sum()),
                "fold_treated": int(a[test].sum()),
                "fold_control": int((1 - a[test]).sum()),
                "fold_effect_days": subset_aipw_effect(y, a, e, mu0, mu1, test),
            }
        )
        loo_rows.append(
            {
                "omitted_fold": int(f),
                "retained_n": int(keep.sum()),
                "leave_one_fold_out_effect_days": subset_aipw_effect(
                    y, a, e, mu0, mu1, keep
                ),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(loo_rows)


def checkpoint_identity(root: Path, config: dict) -> dict[str, object]:
    candidates = {
        "stage17_config": root / "stage17_config.json",
        "stage16_config": root / "stage16_config.json",
        "stage12_utils": root / "scripts/_stage12_utils.py",
        "stage16_utils": root / "scripts/_stage16_utils.py",
        "stage17_utils": root / "scripts/_stage17_utils.py",
        "stage17_preflight": root / "scripts/66_stage17_preflight.py",
        "stage17_forensics": root / "scripts/67_influence_and_fold2_forensics.py",
        "stage17_repeated_crossfit": root / "scripts/68_repeated_crossfit_stability.py",
        "stage17_aggregation": root / "scripts/69_repeated_score_aggregation.py",
        "stage17_decision": root / "scripts/70_generate_stage17_decision.py",
        "stage16_decision": root / "results/tables/65_stage16_decision.csv",
        "stage15_decision": root / "results/tables/59_stage15_decision.csv",
    }
    return {
        "stage": 17,
        "protocol_status": config["protocol_status"],
        "repeated_crossfit": config["repeated_crossfit"],
        "input_hashes": {
            name: sha256_file(path) if path.exists() else None
            for name, path in candidates.items()
        },
    }


def validate_or_create_checkpoint_identity(path: Path, identity: dict[str, object]) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != identity:
            raise RuntimeError(
                "Stage 17 checkpoint identity differs from the current code/config/inputs. "
                "Move or delete the old Stage 17 checkpoint files before rerunning."
            )
    else:
        write_json(identity, path)


def completed_repeat_numbers(
    estimates: pd.DataFrame,
    config: dict,
) -> set[int]:
    if estimates.empty or "repeat" not in estimates.columns:
        return set()
    rcfg = config["repeated_crossfit"]
    expected = (
        len(rcfg["g_min_values"])
        * len(rcfg["propensity_tracks"])
        * 2
    )
    counts = estimates.groupby("repeat").size()
    return {int(rep) for rep, count in counts.items() if int(count) == expected}


def append_replace_repeat(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    repeat: int,
) -> pd.DataFrame:
    if not existing.empty and "repeat" in existing.columns:
        existing = existing[pd.to_numeric(existing["repeat"], errors="coerce") != repeat]
    return pd.concat([existing, new_rows], ignore_index=True)


def aggregate_patient_scores(scores: pd.DataFrame) -> dict[str, object]:
    required = {"local_row_index", "score_numerator", "h", "original_fold"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    grouped = (
        scores.groupby(["local_row_index", "original_fold"], as_index=False)
        .agg(score_numerator=("score_numerator", "mean"), h=("h", "mean"))
        .sort_values("local_row_index")
        .reset_index(drop=True)
    )
    denominator = float(grouped["h"].sum())
    theta = float(grouped["score_numerator"].sum() / denominator)
    mean_h = float(grouped["h"].mean())
    influence = (
        grouped["score_numerator"].to_numpy(float)
        - theta * grouped["h"].to_numpy(float)
    ) / mean_h
    se = float(np.std(influence, ddof=1) / np.sqrt(len(grouped)))
    grouped["aggregated_influence"] = influence
    grouped["absolute_aggregated_influence"] = np.abs(influence)
    grouped["aggregated_normalized_contribution_days"] = (
        grouped["score_numerator"] / denominator
    )
    return {
        "estimate_days": theta,
        "if_se_days": se,
        "if_ci_low_days": theta - 1.96 * se,
        "if_ci_high_days": theta + 1.96 * se,
        "patient": grouped,
    }


def aggregate_loo_by_original_fold(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in sorted(patient["original_fold"].unique()):
        keep = patient["original_fold"] != f
        theta = float(
            patient.loc[keep, "score_numerator"].sum()
            / patient.loc[keep, "h"].sum()
        )
        rows.append(
            {
                "omitted_original_fold": int(f),
                "retained_n": int(keep.sum()),
                "aggregated_loo_effect_days": theta,
            }
        )
    return pd.DataFrame(rows)


def sign_nonzero(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def finite_quantile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float("nan")
