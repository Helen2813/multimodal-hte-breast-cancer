from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


ELIGIBILITY_SCORE_COLUMNS = {"W_ER", "W_PR", "W_HER2"}


def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return mean, var


def smd(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = weighted_stats(x[t == 1], w[t == 1])
    m0, v0 = weighted_stats(x[t == 0], w[t == 0])
    pooled = np.sqrt((v1 + v0) / 2)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def ess(w: np.ndarray) -> float:
    return float((w.sum() ** 2) / np.sum(w**2)) if np.sum(w**2) > 0 else np.nan


def cross_fit_ps(X: pd.DataFrame, t: np.ndarray, seed: int = 91) -> np.ndarray:
    counts = np.bincount(t, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        raise ValueError(f"Too few patients per treatment class: {counts}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ps = np.full(len(t), np.nan)
    for fold, (train, test) in enumerate(splitter.split(X, t), start=1):
        inner = max(2, int(min(4, np.bincount(t[train], minlength=2).min())))
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegressionCV(
                        Cs=[0.01, 0.03, 0.1, 0.3, 1, 3],
                        cv=inner,
                        scoring="neg_log_loss",
                        max_iter=5000,
                        n_jobs=-1,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train], t[train])
        ps[test] = model.predict_proba(X.iloc[test])[:, 1]
    return np.clip(ps, 0.01, 0.99)


def exponential_tilt_weights(X: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=float)

    def objective(lam):
        scores = X @ lam
        return float(logsumexp(scores) - np.log(len(scores)) - target @ lam + 1e-7 * np.sum(lam**2))

    result = minimize(
        objective,
        np.zeros(X.shape[1]),
        method="BFGS",
        options={"maxiter": 3000, "gtol": 1e-8},
    )
    scores = X @ result.x
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.mean(weights)
    return weights, bool(result.success)


def main() -> int:
    ensure_dirs()
    compact_dir = DERIVED_DIR / "compact_adjustment"
    table_dir = RESULTS_DIR / "tables"

    paths = sorted(compact_dir.glob("*_compact.csv"))
    if not paths:
        raise FileNotFoundError("Compact adjustment matrices were not found.")

    summary_rows = []
    for path in paths:
        cohort = path.stem.replace("_compact", "")
        df = read_table(path)
        t = pd.to_numeric(df["analysis_treatment"], errors="raise").astype(int).to_numpy()

        W_cols = [
            c for c in df.columns
            if c.startswith("W_") and c not in ELIGIBILITY_SCORE_COLUMNS
        ]
        X_raw = df[W_cols].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X = scaler.fit_transform(imputer.fit_transform(X_raw))

        ps = cross_fit_ps(X_raw, t)
        ow = np.where(t == 1, 1 - ps, ps)

        target = np.mean(X, axis=0)
        calibration = np.zeros(len(df))
        success = {}
        for group in (0, 1):
            group_w, ok = exponential_tilt_weights(X[t == group], target)
            calibration[t == group] = group_w
            success[group] = ok

        balance_rows = []
        for j, col in enumerate(W_cols):
            balance_rows.append(
                {
                    "cohort": cohort,
                    "feature": col,
                    "smd_unweighted": smd(X[:, j], t, np.ones(len(t))),
                    "smd_overlap": smd(X[:, j], t, ow),
                    "smd_calibration": smd(X[:, j], t, calibration),
                }
            )
        balance = pd.DataFrame(balance_rows)
        for method in ("unweighted", "overlap", "calibration"):
            balance[f"abs_smd_{method}"] = balance[f"smd_{method}"].abs()
        balance.to_csv(
            table_dir / f"15_balance_without_receptor_scores_{cohort}.csv",
            index=False,
        )

        summary_rows.append(
            {
                "cohort": cohort,
                "n": len(df),
                "treated": int(t.sum()),
                "control": int((t == 0).sum()),
                "n_covariates_without_receptor_scores": len(W_cols),
                "max_abs_smd_unweighted": float(balance["abs_smd_unweighted"].max()),
                "max_abs_smd_overlap": float(balance["abs_smd_overlap"].max()),
                "max_abs_smd_calibration": float(balance["abs_smd_calibration"].max()),
                "mean_abs_smd_overlap": float(balance["abs_smd_overlap"].mean()),
                "mean_abs_smd_calibration": float(balance["abs_smd_calibration"].mean()),
                "ess_overlap_treated": ess(ow[t == 1]),
                "ess_overlap_control": ess(ow[t == 0]),
                "ess_calibration_treated": ess(calibration[t == 1]),
                "ess_calibration_control": ess(calibration[t == 0]),
                "max_calibration_weight": float(np.max(calibration)),
                "calibration_optimizer_success_treated": int(success[1]),
                "calibration_optimizer_success_control": int(success[0]),
            }
        )

        pd.DataFrame(
            {
                "patient_id_normalized": df["patient_id_normalized"],
                "analysis_treatment": t,
                "propensity_oof_without_receptor_scores": ps,
                "overlap_weight_without_receptor_scores": ow,
                "calibration_weight_without_receptor_scores": calibration,
            }
        ).to_csv(
            table_dir / f"15_weights_without_receptor_scores_{cohort}.csv",
            index=False,
        )

    summary = pd.DataFrame(summary_rows).sort_values("cohort")
    summary["diagnostic_status"] = np.where(
        (summary["max_abs_smd_calibration"] <= 0.10)
        & (summary["ess_calibration_control"] >= 30)
        & (summary["ess_calibration_treated"] >= 50),
        "BALANCE_FEASIBLE_AFTER_SOURCE_CORRECTION",
        "STILL_LIMITED",
    )
    summary.to_csv(
        table_dir / "15_balance_sensitivity_summary.csv", index=False
    )

    print("\nBalance sensitivity without current receptor score columns:")
    print(summary.to_string(index=False))
    print(
        "\nThis is a diagnostic only. Verified raw/binary receptor statuses must be "
        "used to rebuild the final eligibility cohorts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
