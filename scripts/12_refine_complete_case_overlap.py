from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    return float(mean), float(var)


def weighted_smd(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = weighted_stats(x[t == 1], w[t == 1])
    m0, v0 = weighted_stats(x[t == 0], w[t == 0])
    pooled = np.sqrt((v1 + v0) / 2)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float((w.sum() ** 2) / np.sum(w**2)) if np.sum(w**2) > 0 else np.nan


def cross_fit_ps(X: pd.DataFrame, t: np.ndarray, seed: int = 73) -> np.ndarray:
    counts = np.bincount(t, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        raise ValueError(f"Too few observations per treatment class: {counts}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ps = np.full(len(t), np.nan)
    for fold, (train, test) in enumerate(splitter.split(X, t), start=1):
        inner = int(min(4, np.bincount(t[train], minlength=2).min()))
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegressionCV(
                        Cs=[0.01, 0.03, 0.1, 0.3, 1, 3],
                        cv=max(2, inner),
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


def main() -> int:
    ensure_dirs()
    compact_dir = DERIVED_DIR / "compact_adjustment"
    cohort_dir = DERIVED_DIR / "cohorts"
    restricted_dir = DERIVED_DIR / "restricted_complete_case"
    restricted_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    cohorts = [
        "complete_case_hormone_hrpos_her2neg",
        "complete_case_chemo_tnbc",
    ]
    summary_rows = []

    for cohort_name in cohorts:
        compact_path = compact_dir / f"{cohort_name}_compact.csv"
        full_path = cohort_dir / f"{cohort_name}.csv"
        initial_ps_path = table_dir / f"06_compact_propensity_{cohort_name}.csv"
        for path in (compact_path, full_path, initial_ps_path):
            if not path.exists():
                raise FileNotFoundError(path)

        compact = read_table(compact_path)
        full = read_table(full_path)
        initial = read_table(initial_ps_path)
        merged = compact.merge(
            initial[["patient_id_normalized", "propensity_score_oof"]],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        t = pd.to_numeric(merged["analysis_treatment"], errors="raise").astype(int).to_numpy()
        ps0 = pd.to_numeric(merged["propensity_score_oof"], errors="raise").to_numpy()

        q_low = {
            group: float(np.quantile(ps0[t == group], 0.025))
            for group in (0, 1)
        }
        q_high = {
            group: float(np.quantile(ps0[t == group], 0.975))
            for group in (0, 1)
        }
        lower = max(q_low.values())
        upper = min(q_high.values())
        support = (ps0 >= lower) & (ps0 <= upper)

        restricted = merged.loc[support].reset_index(drop=True)
        W_cols = [c for c in restricted.columns if c.startswith("W_")]
        X = restricted[W_cols].apply(pd.to_numeric, errors="coerce")
        tr = pd.to_numeric(restricted["analysis_treatment"], errors="raise").astype(int).to_numpy()
        ps = cross_fit_ps(X, tr)
        ow = np.where(tr == 1, 1 - ps, ps)

        balance_rows = []
        for col in W_cols:
            x = pd.to_numeric(restricted[col], errors="coerce").to_numpy(dtype=float)
            balance_rows.append(
                {
                    "cohort": cohort_name,
                    "feature": col,
                    "smd_unweighted_restricted": weighted_smd(
                        x, tr, np.ones(len(tr))
                    ),
                    "smd_overlap_restricted": weighted_smd(x, tr, ow),
                }
            )
        balance = pd.DataFrame(balance_rows)
        balance["abs_smd_overlap"] = balance["smd_overlap_restricted"].abs()
        balance.to_csv(
            table_dir / f"12_restricted_balance_{cohort_name}.csv", index=False
        )

        weights = pd.DataFrame(
            {
                "patient_id_normalized": restricted["patient_id_normalized"],
                "analysis_treatment": tr,
                "propensity_score_oof_restricted": ps,
                "overlap_weight_restricted": ow,
            }
        )
        weights.to_csv(
            table_dir / f"12_restricted_weights_{cohort_name}.csv", index=False
        )

        selected_ids = set(restricted["patient_id_normalized"])
        full_restricted = full[
            full["patient_id_normalized"].isin(selected_ids)
        ].copy()
        full_restricted.to_csv(
            restricted_dir / f"{cohort_name}_restricted.csv", index=False
        )

        max_smd = float(balance["abs_smd_overlap"].max())
        n_t = int(tr.sum())
        n_c = int((tr == 0).sum())
        summary_rows.append(
            {
                "cohort": cohort_name,
                "original_n": len(merged),
                "support_lower": lower,
                "support_upper": upper,
                "restricted_n": len(restricted),
                "restricted_treated": n_t,
                "restricted_control": n_c,
                "retained_fraction": float(len(restricted) / len(merged)),
                "ess_overlap_treated": ess(ow[tr == 1]),
                "ess_overlap_control": ess(ow[tr == 0]),
                "max_abs_smd_overlap_restricted": max_smd,
                "status": (
                    "USABLE_AS_SECONDARY_HTE"
                    if n_t >= 50 and n_c >= 40 and max_smd <= 0.15
                    else "EXPLORATORY_ONLY"
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        table_dir / "12_restricted_complete_case_summary.csv", index=False
    )
    print("\nRestricted complete-case summary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
