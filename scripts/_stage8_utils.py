from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, read_table


HORIZONS = (730.0, 1095.0, 1460.0, 1825.0)
INTERVAL_DAYS = 180.0


def cohort_key(path: Path) -> str:
    return path.stem.replace("_verified", "")


def get_repeat_assignments(cohort: str, repeat: int = 1) -> pd.DataFrame:
    path = (
        DERIVED_DIR
        / "verified_splits"
        / "23_verified_repeated_fold_assignments.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    folds = read_table(path)
    out = folds[
        (folds["cohort"] == cohort) & (folds["repeat"] == repeat)
    ].copy()
    if out.empty:
        raise ValueError(f"No fold assignments for {cohort}, repeat={repeat}")
    return out


def load_years() -> pd.DataFrame:
    path = DERIVED_DIR / "manifests" / "20_patient_diagnosis_year.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_table(path)[
        ["patient_id_normalized", "diagnosis_year"]
    ]


def load_compact_with_year(cohort: str) -> tuple[pd.DataFrame, list[str]]:
    compact_path = (
        DERIVED_DIR
        / "verified_compact_adjustment"
        / f"{cohort}_compact_verified.csv"
    )
    if not compact_path.exists():
        raise FileNotFoundError(compact_path)
    compact = read_table(compact_path)
    compact = compact.merge(
        load_years(),
        on="patient_id_normalized",
        how="left",
        validate="one_to_one",
    )
    compact["diagnosis_year_missing"] = (
        compact["diagnosis_year"].isna().astype(float)
    )
    features = [c for c in compact.columns if c.startswith("W_")]
    if (
        compact["diagnosis_year"].notna().sum()
        >= max(20, int(0.5 * len(compact)))
        and compact["diagnosis_year"].nunique(dropna=True) > 1
    ):
        features += ["diagnosis_year", "diagnosis_year_missing"]
    return compact, features


def make_propensity_model(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
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


def crossfit_propensity(
    compact: pd.DataFrame,
    features: list[str],
    assignments: pd.DataFrame,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = compact.merge(
        assignments[["patient_id_normalized", "fold"]],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(compact):
        raise ValueError("Fold assignments do not match compact cohort.")

    X = merged[features].apply(pd.to_numeric, errors="coerce")
    t = pd.to_numeric(
        merged["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    folds = merged["fold"].astype(int).to_numpy()
    ps = np.full(len(merged), np.nan)
    tuning = []

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        model = make_propensity_model(seed + int(fold))
        model.fit(X.loc[train], t[train])
        ps[test] = model.predict_proba(X.loc[test])[:, 1]
        tuning.append(
            {
                "fold": int(fold),
                "chosen_C": float(model.named_steps["logit"].C_[0]),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
            }
        )

    return np.clip(ps, 0.01, 0.99), pd.DataFrame(tuning)


def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return mean, var


def weighted_smd(
    x: np.ndarray, treatment: np.ndarray, weights: np.ndarray
) -> float:
    m1, v1 = weighted_stats(x[treatment == 1], weights[treatment == 1])
    m0, v0 = weighted_stats(x[treatment == 0], weights[treatment == 0])
    pooled = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights >= 0)]
    if len(weights) == 0 or np.sum(weights**2) <= 0:
        return np.nan
    return float((weights.sum() ** 2) / np.sum(weights**2))


def reverse_km_censoring(
    times: np.ndarray,
    events: np.ndarray,
    horizons: Iterable[float] = HORIZONS,
) -> dict[float, float]:
    """
    Reverse Kaplan–Meier for probability of remaining uncensored.
    Deaths/events are treated as censoring observations for the censoring process.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    valid = np.isfinite(times) & np.isfinite(events) & (times >= 0)
    times, events = times[valid], events[valid]
    censor_event = 1 - events

    censor_times = np.sort(np.unique(times[censor_event == 1]))
    survival = 1.0
    values = {}
    horizons_sorted = sorted(float(h) for h in horizons)
    h_idx = 0

    for time in censor_times:
        while h_idx < len(horizons_sorted) and horizons_sorted[h_idx] < time:
            values[horizons_sorted[h_idx]] = survival
            h_idx += 1
        at_risk = int(np.sum(times >= time))
        d = int(np.sum((times == time) & (censor_event == 1)))
        if at_risk > 0:
            survival *= max(0.0, 1.0 - d / at_risk)

    while h_idx < len(horizons_sorted):
        values[horizons_sorted[h_idx]] = survival
        h_idx += 1
    return values


def build_interval_rows(
    df: pd.DataFrame,
    W: pd.DataFrame,
    interval_days: float = INTERVAL_DAYS,
    max_horizon: float = 1825.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    starts = np.arange(0.0, max_horizon, interval_days)
    ends = np.minimum(starts + interval_days, max_horizon)
    rows = []

    time = pd.to_numeric(df["analysis_time"], errors="coerce").to_numpy(float)
    event = pd.to_numeric(df["analysis_event"], errors="raise").astype(int).to_numpy()
    treatment = pd.to_numeric(
        df["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()

    W_values = W.reset_index(drop=True)
    for i in range(len(df)):
        for k, (start, end) in enumerate(zip(starts, ends)):
            if not np.isfinite(time[i]) or time[i] < start:
                continue
            row = W_values.iloc[i].to_dict()
            row.update(
                {
                    "patient_row": i,
                    "interval": k,
                    "interval_start": start,
                    "interval_end": end,
                    "treatment": treatment[i],
                    "censor_event": int(
                        event[i] == 0 and time[i] >= start and time[i] <= end
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), starts, ends


def rmst_ipcw_pseudooutcome(
    observed_time: np.ndarray,
    G_start: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    horizon: float,
    min_g: float = 0.05,
) -> np.ndarray:
    observed_time = np.asarray(observed_time, dtype=float)
    n = len(observed_time)
    out = np.zeros(n, dtype=float)
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
            length / np.clip(G_start[:, k], min_g, 1.0),
            0.0,
        )
    return out


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 42,
    n_boot: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        estimates.append(float(np.mean(values[idx])))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )
