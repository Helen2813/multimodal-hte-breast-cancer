from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, read_table


LANDMARKS = (180, 365)
POST_LANDMARK_HORIZONS = (730.0, 1095.0)
INTERVAL_DAYS = 90.0


def make_propensity(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    cv=4,
                    scoring="neg_log_loss",
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=6000,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def ridge_regression() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ]
    )


def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return mean, var


def smd(x: np.ndarray, a: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = weighted_stats(x[a == 1], w[a == 1])
    m0, v0 = weighted_stats(x[a == 0], w[a == 0])
    pooled = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float((w.sum() ** 2) / np.sum(w**2)) if np.sum(w**2) > 0 else np.nan


def effective_sample_size(w: np.ndarray) -> float:
    """Backward-compatible descriptive name for the effective sample size."""
    return ess(w)


def stratified_folds(
    df: pd.DataFrame,
    seeds: Iterable[int] = (42, 123, 456),
) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    a = pd.to_numeric(df["analysis_treatment"], errors="raise").astype(int)
    e = pd.to_numeric(df["analysis_event"], errors="raise").astype(int)
    joint = 2 * a + e
    counts = joint.value_counts()
    n_splits = 5
    while n_splits > 2 and counts.min() < n_splits:
        n_splits -= 1
    if n_splits < 3:
        joint = a
        counts = joint.value_counts()
        n_splits = min(5, int(counts.min()))
    if n_splits < 3:
        raise ValueError("Insufficient treatment/event counts for 3-fold splitting.")

    rows = []
    for repeat, seed in enumerate(seeds, start=1):
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        fold = np.full(len(df), -1, dtype=int)
        for f, (_, test) in enumerate(
            skf.split(np.zeros(len(df)), joint), start=1
        ):
            fold[test] = f
        temp = pd.DataFrame(
            {
                "patient_id_normalized": df["patient_id_normalized"],
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "n_folds": n_splits,
                "analysis_treatment": a,
                "analysis_event": e,
            }
        )
        rows.append(temp)
    return pd.concat(rows, ignore_index=True)


def crossfit_propensity(
    df: pd.DataFrame,
    features: list[str],
    split: pd.DataFrame,
    repeat: int = 1,
    seed: int = 4200,
) -> tuple[np.ndarray, pd.DataFrame]:
    assignment = split[split["repeat"] == repeat][
        ["patient_id_normalized", "fold"]
    ]
    merged = df.merge(
        assignment,
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    X = merged[features].apply(pd.to_numeric, errors="coerce")
    a = pd.to_numeric(
        merged["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    fold = merged["fold"].astype(int).to_numpy()
    ps = np.full(len(df), np.nan)
    tuning = []
    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f
        model = make_propensity(seed + int(f))
        model.fit(X.loc[train], a[train])
        ps[test] = model.predict_proba(X.loc[test])[:, 1]
        tuning.append(
            {
                "fold": int(f),
                "chosen_C": float(model.named_steps["model"].C_[0]),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
            }
        )
    return np.clip(ps, 0.01, 0.99), pd.DataFrame(tuning)


def reverse_km(
    times: np.ndarray,
    events: np.ndarray,
    horizons: Iterable[float],
) -> dict[float, float]:
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    valid = np.isfinite(times) & np.isfinite(events) & (times >= 0)
    times, events = times[valid], events[valid]
    censor = 1 - events
    surv = 1.0
    values = {}
    hs = sorted(float(h) for h in horizons)
    i = 0
    for time in np.sort(np.unique(times[censor == 1])):
        while i < len(hs) and hs[i] < time:
            values[hs[i]] = surv
            i += 1
        risk = np.sum(times >= time)
        d = np.sum((times == time) & (censor == 1))
        if risk > 0:
            surv *= max(0.0, 1.0 - d / risk)
    while i < len(hs):
        values[hs[i]] = surv
        i += 1
    return values


def interval_long(
    df: pd.DataFrame,
    X: pd.DataFrame,
    horizon: float,
    interval_days: float = INTERVAL_DAYS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    starts = np.arange(0.0, horizon, interval_days)
    ends = np.minimum(starts + interval_days, horizon)
    time = pd.to_numeric(
        df["analysis_time"], errors="coerce"
    ).to_numpy(float)
    event = pd.to_numeric(
        df["analysis_event"], errors="raise"
    ).astype(int).to_numpy()
    a = pd.to_numeric(
        df["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()

    rows = []
    X = X.reset_index(drop=True)
    for i in range(len(df)):
        for k, (start, end) in enumerate(zip(starts, ends)):
            if not np.isfinite(time[i]) or time[i] < start:
                continue
            row = X.iloc[i].to_dict()
            row.update(
                {
                    "patient_row": i,
                    "interval": k,
                    "interval_start": start,
                    "treatment": a[i],
                    "censor_event": int(
                        event[i] == 0 and start <= time[i] <= end
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), starts, ends


def ipcw_rmst(
    observed_time: np.ndarray,
    G_start: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    horizon: float,
    g_min: float,
) -> np.ndarray:
    out = np.zeros(len(observed_time), dtype=float)
    for k, (start, end) in enumerate(zip(starts, ends)):
        if start >= horizon:
            break
        effective_end = min(end, horizon)
        length = np.maximum(
            0.0,
            np.minimum(observed_time, effective_end) - start,
        )
        at_risk = observed_time > start
        out += np.where(
            at_risk,
            length / np.clip(G_start[:, k], g_min, 1.0),
            0.0,
        )
    return out


def aipw_ato(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> tuple[float, np.ndarray]:
    e = np.clip(e, 0.02, 0.98)
    h = e * (1.0 - e)
    score = (
        h * (mu1 - mu0)
        + h * a / e * (y - mu1)
        - h * (1 - a) / (1 - e) * (y - mu0)
    )
    theta = float(score.sum() / h.sum())
    influence = (score - theta * h) / np.mean(h)
    return theta, influence


def crossfit_arm_outcomes(
    X: pd.DataFrame,
    y: np.ndarray,
    a: np.ndarray,
    assignment: pd.DataFrame,
    model_factory,
) -> tuple[np.ndarray, np.ndarray]:
    fold = assignment["fold"].astype(int).to_numpy()
    mu0 = np.full(len(y), np.nan)
    mu1 = np.full(len(y), np.nan)
    for f in sorted(np.unique(fold)):
        test = fold == f
        train = ~test
        for arm in (0, 1):
            arm_train = train & (a == arm)
            if arm_train.sum() < 20:
                raise ValueError(
                    f"Too few training patients for arm={arm}, fold={f}"
                )
            model = model_factory(int(f), arm)
            model.fit(X.loc[arm_train], y[arm_train])
            pred = model.predict(X.loc[test])
            if arm == 0:
                mu0[test] = pred
            else:
                mu1[test] = pred
    return mu0, mu1
