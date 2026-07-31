#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "stage16_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/figures",
        "results/logs",
        "data/derived/stage16",
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


def markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
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
            "| " + " | ".join(str(row[h]).replace("|", "\\|") for h in headers) + " |"
        )
    return "\n".join(lines)


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def exact_landmark_payload() -> dict[str, object]:
    """Reconstruct the exact Stage 41 landmark nuisance quantities.

    The function imports and uses the original Stage 12 implementation rather than
    reimplementing its censoring, pseudo-outcome, or AIPW formulas.
    """
    from _stage12_utils import (
        BASE_SEED,
        G_MIN,
        LANDMARK_HORIZON,
        LANDMARK_INTERVAL,
        aipw_ato,
        assemble_landmark_data,
        crossfit_arm_outcomes,
        crossfit_censor_survival,
        ipcw_rmst_pseudo,
    )

    frame, features, assignment, metadata = assemble_landmark_data()
    frame = frame.copy().reset_index(drop=True)
    fold = assignment["fold"].astype(int).to_numpy()
    a = pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy()
    observed_time = pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float)
    event = pd.to_numeric(frame["analysis_event"], errors="raise").astype(int).to_numpy()
    e = pd.to_numeric(
        frame["propensity_score_oof_stage30"], errors="raise"
    ).to_numpy(float)
    e = np.clip(e, 0.01, 0.99)

    G, starts, ends, censor_metrics = crossfit_censor_survival(
        frame,
        features,
        fold,
        LANDMARK_HORIZON,
        LANDMARK_INTERVAL,
        BASE_SEED + 100,
    )
    y = ipcw_rmst_pseudo(
        observed_time,
        G,
        starts,
        ends,
        LANDMARK_HORIZON,
        G_MIN,
    )
    mu0, mu1 = crossfit_arm_outcomes(
        frame,
        features,
        y,
        fold,
        BASE_SEED + 200,
    )
    theta, influence = aipw_ato(y, a, e, mu0, mu1)

    return {
        "frame": frame,
        "features": features,
        "fold": fold,
        "a": a,
        "event": event,
        "observed_time": observed_time,
        "e": e,
        "G": G,
        "starts": starts,
        "ends": ends,
        "y": y,
        "mu0": mu0,
        "mu1": mu1,
        "theta": theta,
        "influence": influence,
        "metadata": metadata,
        "censor_metrics": censor_metrics,
        "horizon": float(LANDMARK_HORIZON),
        "seed": int(BASE_SEED),
    }


def aipw_components(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> dict[str, object]:
    e = np.clip(np.asarray(e, dtype=float), 0.02, 0.98)
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=int)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    h = e * (1.0 - e)
    denominator = float(h.sum())

    plugin_numerator = h * (mu1 - mu0)
    treated_residual_numerator = h * a / e * (y - mu1)
    control_residual_numerator = -h * (1 - a) / (1 - e) * (y - mu0)
    score_numerator = (
        plugin_numerator + treated_residual_numerator + control_residual_numerator
    )
    theta = float(score_numerator.sum() / denominator)
    influence = (score_numerator - theta * h) / float(np.mean(h))

    w1 = a * (1.0 - e)
    w0 = (1 - a) * e
    direct_treated = float(np.sum(w1 * y) / np.sum(w1))
    direct_control = float(np.sum(w0 * y) / np.sum(w0))
    direct_effect = direct_treated - direct_control

    patient = pd.DataFrame(
        {
            "h": h,
            "plugin_numerator": plugin_numerator,
            "treated_residual_numerator": treated_residual_numerator,
            "control_residual_numerator": control_residual_numerator,
            "score_numerator": score_numerator,
            "normalized_contribution_days": score_numerator / denominator,
            "influence": influence,
        }
    )
    summary = {
        "estimate_days": theta,
        "direct_ato_ipw_treated_mean_days": direct_treated,
        "direct_ato_ipw_control_mean_days": direct_control,
        "direct_ato_ipw_effect_days": direct_effect,
        "plugin_component_days": float(plugin_numerator.sum() / denominator),
        "treated_residual_component_days": float(
            treated_residual_numerator.sum() / denominator
        ),
        "control_residual_component_days": float(
            control_residual_numerator.sum() / denominator
        ),
        "total_residual_augmentation_days": float(
            (treated_residual_numerator + control_residual_numerator).sum()
            / denominator
        ),
        "aipw_minus_direct_ato_ipw_days": theta - direct_effect,
        "ato_denominator": denominator,
    }
    return {"summary": summary, "patient": patient}


def factual_prediction_metrics(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    horizon: float,
    model_name: str,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=int)
    e = np.clip(np.asarray(e, dtype=float), 0.02, 0.98)
    factual = np.where(a == 1, mu1, mu0)
    rows = []
    for arm in ("all", 0, 1):
        mask = np.ones(len(y), dtype=bool) if arm == "all" else a == arm
        residual = y[mask] - factual[mask]
        if arm == "all":
            weights = np.ones(mask.sum(), dtype=float)
        elif arm == 1:
            weights = (1.0 - e[mask]).astype(float)
        else:
            weights = e[mask].astype(float)
        weights = weights / weights.sum()
        weighted_bias = float(np.sum(weights * residual))
        weighted_mse = float(np.sum(weights * residual**2))
        rows.append(
            {
                "model": model_name,
                "arm": arm,
                "n": int(mask.sum()),
                "factual_mse": float(np.mean(residual**2)),
                "factual_mae": float(np.mean(np.abs(residual))),
                "factual_bias_observed_minus_predicted": float(np.mean(residual)),
                "ato_weighted_bias_observed_minus_predicted": weighted_bias,
                "ato_weighted_mse": weighted_mse,
                "prediction_min": float(np.min(factual[mask])),
                "prediction_p01": float(np.quantile(factual[mask], 0.01)),
                "prediction_median": float(np.median(factual[mask])),
                "prediction_p99": float(np.quantile(factual[mask], 0.99)),
                "prediction_max": float(np.max(factual[mask])),
                "fraction_prediction_below_zero": float(
                    np.mean(factual[mask] < 0.0)
                ),
                "fraction_prediction_above_horizon": float(
                    np.mean(factual[mask] > horizon)
                ),
            }
        )
    return pd.DataFrame(rows)


def _crossfit_arm_mean(
    y: np.ndarray,
    a: np.ndarray,
    fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu0 = np.full(len(y), np.nan)
    mu1 = np.full(len(y), np.nan)
    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f
        for arm in (0, 1):
            values = y[train & (a == arm)]
            if len(values) < 1:
                raise ValueError(f"No training outcomes for arm={arm}, fold={f}")
            if arm == 0:
                mu0[test] = float(np.mean(values))
            else:
                mu1[test] = float(np.mean(values))
    return mu0, mu1


def _crossfit_arm_hgb(
    frame: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    a: np.ndarray,
    fold: np.ndarray,
    seed: int,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    X = numeric_frame(frame, features)
    mu0 = np.full(len(frame), np.nan)
    mu1 = np.full(len(frame), np.nan)
    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f
        for arm in (0, 1):
            arm_train = train & (a == arm)
            if int(arm_train.sum()) < 20:
                raise ValueError(
                    f"HGB outcome model arm={arm}, fold={f}: "
                    f"{int(arm_train.sum())} training rows."
                )
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=float(params["learning_rate"]),
                            max_iter=int(params["max_iter"]),
                            max_leaf_nodes=int(params["max_leaf_nodes"]),
                            min_samples_leaf=int(params["min_samples_leaf"]),
                            l2_regularization=float(params["l2_regularization"]),
                            random_state=seed + 10 * int(f) + arm,
                        ),
                    ),
                ]
            )
            model.fit(X.loc[arm_train], y[arm_train])
            pred = model.predict(X.loc[test])
            if arm == 0:
                mu0[test] = pred
            else:
                mu1[test] = pred
    return mu0, mu1


def _crossfit_pooled_interaction_ridge(
    frame: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    a: np.ndarray,
    fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = numeric_frame(frame, features)
    mu0 = np.full(len(frame), np.nan)
    mu1 = np.full(len(frame), np.nan)
    alphas = [0.1, 1.0, 10.0, 100.0]

    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X.loc[train]))
        X_test = scaler.transform(imputer.transform(X.loc[test]))
        a_train = a[train].astype(float)

        design_train = np.column_stack(
            [X_train, a_train, X_train * a_train[:, None]]
        )
        model = RidgeCV(alphas=alphas)
        model.fit(design_train, y[train])

        zeros = np.zeros(int(test.sum()), dtype=float)
        ones = np.ones(int(test.sum()), dtype=float)
        design0 = np.column_stack([X_test, zeros, X_test * zeros[:, None]])
        design1 = np.column_stack([X_test, ones, X_test * ones[:, None]])
        mu0[test] = model.predict(design0)
        mu1[test] = model.predict(design1)
    return mu0, mu1


def generate_outcome_predictions(
    payload: dict[str, object],
    model_name: str,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    frame = payload["frame"]
    features = payload["features"]
    y = np.asarray(payload["y"], dtype=float)
    a = np.asarray(payload["a"], dtype=int)
    fold = np.asarray(payload["fold"], dtype=int)
    horizon = float(payload["horizon"])

    if model_name == "arm_mean":
        mu0, mu1 = _crossfit_arm_mean(y, a, fold)
    elif model_name == "arm_ridge_unbounded":
        mu0 = np.asarray(payload["mu0"], dtype=float).copy()
        mu1 = np.asarray(payload["mu1"], dtype=float).copy()
    elif model_name == "arm_ridge_bounded":
        mu0 = np.clip(np.asarray(payload["mu0"], dtype=float), 0.0, horizon)
        mu1 = np.clip(np.asarray(payload["mu1"], dtype=float), 0.0, horizon)
    elif model_name == "pooled_interaction_ridge_bounded":
        mu0, mu1 = _crossfit_pooled_interaction_ridge(
            frame, features, y, a, fold
        )
        mu0, mu1 = np.clip(mu0, 0.0, horizon), np.clip(mu1, 0.0, horizon)
    elif model_name == "arm_hist_gradient_boosting_bounded":
        mu0, mu1 = _crossfit_arm_hgb(
            frame,
            features,
            y,
            a,
            fold,
            int(payload["seed"]) + 3000,
            config["fixed_hgb_parameters"],
        )
        mu0, mu1 = np.clip(mu0, 0.0, horizon), np.clip(mu1, 0.0, horizon)
    else:
        raise ValueError(f"Unknown outcome model: {model_name}")

    if not np.isfinite(mu0).all() or not np.isfinite(mu1).all():
        raise RuntimeError(f"Non-finite predictions from {model_name}")
    return mu0, mu1


def subset_aipw_effect(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    mask: np.ndarray,
) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return float("nan")
    return float(
        aipw_components(
            y[mask], a[mask], e[mask], mu0[mask], mu1[mask]
        )["summary"]["estimate_days"]
    )
