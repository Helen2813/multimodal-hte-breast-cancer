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
from _compact_adjustment import build_compact_adjustment, manifest_to_frame


def weighted_moments(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return mean, var


def smd(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = weighted_moments(x[t == 1], w[t == 1])
    m0, v0 = weighted_moments(x[t == 0], w[t == 0])
    pooled = np.sqrt((v1 + v0) / 2)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def ess(w: np.ndarray) -> float:
    return (
        float((w.sum() ** 2) / np.sum(w**2))
        if np.sum(w**2) > 0
        else np.nan
    )


def cross_fitted_ps(X: pd.DataFrame, t: np.ndarray, seed: int = 117) -> np.ndarray:
    counts = np.bincount(t, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        raise ValueError(f"Too few observations per treatment class: {counts}")

    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    ps = np.full(len(t), np.nan)

    for fold, (train, test) in enumerate(splitter.split(X, t), start=1):
        inner = max(
            2,
            int(min(4, np.bincount(t[train], minlength=2).min())),
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegressionCV(
                        Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
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


def calibration_weights(X: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
    def objective(lam: np.ndarray) -> float:
        scores = X @ lam
        return float(
            logsumexp(scores)
            - np.log(len(scores))
            - target @ lam
            + 1e-7 * np.sum(lam**2)
        )

    result = minimize(
        objective,
        np.zeros(X.shape[1]),
        method="BFGS",
        options={"maxiter": 4000, "gtol": 1e-8},
    )
    scores = X @ result.x
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= np.mean(weights)
    return weights, bool(result.success)


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "verified_cohorts"
    compact_dir = DERIVED_DIR / "verified_compact_adjustment"
    compact_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    paths = sorted(cohort_dir.glob("*_verified.csv"))
    if not paths:
        raise FileNotFoundError(
            "Verified cohort files were not found. Run Stage 18 first."
        )

    summary_rows = []
    for path in paths:
        label = path.stem.replace("_verified", "")
        df = read_table(path)

        # Prevent standardized receptor scores from entering W.
        receptor_score_cols = [
            c for c in ("ER_status", "PR_status", "HER2_status")
            if c in df.columns
        ]
        adjustment_input = df.drop(columns=receptor_score_cols, errors="ignore")
        W, manifest = build_compact_adjustment(adjustment_input)
        if W.shape[1] < 3:
            raise ValueError(f"{label}: too few compact adjustment covariates.")

        metadata = [
            c for c in (
                "patient_id_normalized",
                "analysis_treatment",
                "analysis_event",
                "analysis_time",
                "ER_observed_binary",
                "PR_observed_binary",
                "HER2_observed_binary",
            )
            if c in df.columns
        ]
        compact = pd.concat(
            [df[metadata].reset_index(drop=True), W.reset_index(drop=True)],
            axis=1,
        )
        compact.to_csv(
            compact_dir / f"{label}_compact_verified.csv",
            index=False,
        )
        manifest_to_frame(manifest).to_csv(
            compact_dir / f"{label}_compact_manifest.csv",
            index=False,
        )

        t = pd.to_numeric(
            compact["analysis_treatment"], errors="raise"
        ).astype(int).to_numpy()
        W_cols = [c for c in compact.columns if c.startswith("W_")]
        X_raw = compact[W_cols].apply(pd.to_numeric, errors="coerce")

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X = scaler.fit_transform(imputer.fit_transform(X_raw))

        ps = cross_fitted_ps(X_raw, t)
        ow = np.where(t == 1, 1 - ps, ps)

        target = X.mean(axis=0)
        cw = np.zeros(len(t))
        optimizer_success = {}
        for group in (0, 1):
            group_w, success = calibration_weights(X[t == group], target)
            cw[t == group] = group_w
            optimizer_success[group] = success

        balance_rows = []
        for j, col in enumerate(W_cols):
            balance_rows.append(
                {
                    "cohort": label,
                    "feature": col,
                    "smd_unweighted": smd(
                        X[:, j], t, np.ones(len(t))
                    ),
                    "smd_overlap": smd(X[:, j], t, ow),
                    "smd_calibration": smd(X[:, j], t, cw),
                }
            )
        balance_df = pd.DataFrame(balance_rows)
        for method in ("unweighted", "overlap", "calibration"):
            balance_df[f"abs_smd_{method}"] = (
                balance_df[f"smd_{method}"].abs()
            )
        balance_df.to_csv(
            table_dir / f"19_verified_balance_{label}.csv",
            index=False,
        )

        pd.DataFrame(
            {
                "patient_id_normalized": compact[
                    "patient_id_normalized"
                ],
                "analysis_treatment": t,
                "propensity_score_oof": ps,
                "overlap_weight": ow,
                "calibration_weight": cw,
            }
        ).to_csv(
            table_dir / f"19_verified_weights_{label}.csv",
            index=False,
        )

        summary_rows.append(
            {
                "cohort": label,
                "n": len(compact),
                "treated": int(t.sum()),
                "control": int((t == 0).sum()),
                "events": int(
                    pd.to_numeric(compact["analysis_event"]).sum()
                ),
                "n_adjustment_covariates": len(W_cols),
                "max_abs_smd_unweighted": float(
                    balance_df["abs_smd_unweighted"].max()
                ),
                "max_abs_smd_overlap": float(
                    balance_df["abs_smd_overlap"].max()
                ),
                "max_abs_smd_calibration": float(
                    balance_df["abs_smd_calibration"].max()
                ),
                "mean_abs_smd_overlap": float(
                    balance_df["abs_smd_overlap"].mean()
                ),
                "mean_abs_smd_calibration": float(
                    balance_df["abs_smd_calibration"].mean()
                ),
                "ess_overlap_treated": ess(ow[t == 1]),
                "ess_overlap_control": ess(ow[t == 0]),
                "ess_calibration_treated": ess(cw[t == 1]),
                "ess_calibration_control": ess(cw[t == 0]),
                "max_calibration_weight": float(cw.max()),
                "optimizer_success_treated": int(
                    optimizer_success[1]
                ),
                "optimizer_success_control": int(
                    optimizer_success[0]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("cohort")
    summary["analysis_status"] = np.select(
        [
            (
                (summary["max_abs_smd_calibration"] <= 0.10)
                & (summary["ess_calibration_treated"] >= 50)
                & (summary["ess_calibration_control"] >= 50)
            ),
            (
                (summary["max_abs_smd_calibration"] <= 0.15)
                & (summary["ess_calibration_treated"] >= 40)
                & (summary["ess_calibration_control"] >= 25)
            ),
        ],
        [
            "PRIMARY_OR_SECONDARY_USABLE",
            "RESTRICTED_EXPLORATORY_USABLE",
        ],
        default="INSUFFICIENT_BALANCE_OR_ESS",
    )
    summary.to_csv(
        table_dir / "19_verified_balance_summary.csv",
        index=False,
    )

    print("\nVerified cohort balance summary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
