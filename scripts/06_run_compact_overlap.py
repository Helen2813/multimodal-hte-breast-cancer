from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


def weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[mask]
    w = w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    return float(mean), float(var)


def weighted_smd(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = weighted_mean_var(x[t == 1], w[t == 1])
    m0, v0 = weighted_mean_var(x[t == 0], w[t == 0])
    pooled = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights >= 0)]
    if len(weights) == 0 or np.sum(weights**2) <= 0:
        return np.nan
    return float((weights.sum() ** 2) / np.sum(weights**2))


def cross_fitted_propensity(X: pd.DataFrame, t: np.ndarray, seed: int = 42) -> tuple[np.ndarray, list[float]]:
    counts = np.bincount(t.astype(int), minlength=2)
    n_splits = int(min(5, counts.min()))
    if n_splits < 3:
        raise ValueError(f"Too few observations in a treatment class for cross-fitting: {counts}")

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ps = np.full(len(t), np.nan, dtype=float)
    chosen_c: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, t), start=1):
        train_counts = np.bincount(t[train_idx].astype(int), minlength=2)
        inner_splits = int(min(4, train_counts.min()))
        if inner_splits < 2:
            inner_splits = 2

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegressionCV(
                        Cs=np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0]),
                        cv=inner_splits,
                        scoring="neg_log_loss",
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=seed + fold,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train_idx], t[train_idx])
        ps[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        chosen_c.append(float(model.named_steps["logit"].C_[0]))

    if np.isnan(ps).any():
        raise RuntimeError("Cross-fitted propensity scores contain missing values.")
    return np.clip(ps, 0.01, 0.99), chosen_c


def run_one(path: Path) -> dict[str, object]:
    label = path.stem.replace("_compact", "")
    df = read_table(path)
    feature_cols = [c for c in df.columns if str(c).startswith("W_")]
    if not feature_cols:
        raise ValueError(f"{path}: no W_ compact covariates found.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    t = pd.to_numeric(df["analysis_treatment"], errors="raise").astype(int).to_numpy()
    p = float(t.mean())

    ps, chosen_c = cross_fitted_propensity(X, t)

    # Stabilized IPTW. Report both raw and 99th-percentile truncated weights.
    ipw_raw = np.where(t == 1, p / ps, (1.0 - p) / (1.0 - ps))
    cap = float(np.quantile(ipw_raw, 0.99))
    ipw = np.minimum(ipw_raw, cap)

    # Overlap weights target the population with clinical equipoise.
    ow = np.where(t == 1, 1.0 - ps, ps)

    trim = (ps >= 0.10) & (ps <= 0.90)
    trim_w = trim.astype(float)

    balance_rows = []
    for col in feature_cols:
        x = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        unweighted = np.ones(len(df), dtype=float)
        balance_rows.append(
            {
                "cohort": label,
                "feature": col,
                "smd_unweighted": weighted_smd(x, t, unweighted),
                "smd_ipw_truncated_p99": weighted_smd(x, t, ipw),
                "smd_overlap_weighted": weighted_smd(x, t, ow),
                "smd_trimmed_unweighted": weighted_smd(x, t, trim_w),
                "missing_fraction": float(np.mean(~np.isfinite(x))),
            }
        )

    balance = pd.DataFrame(balance_rows)
    balance["abs_smd_unweighted"] = balance["smd_unweighted"].abs()
    balance["abs_smd_ipw"] = balance["smd_ipw_truncated_p99"].abs()
    balance["abs_smd_overlap"] = balance["smd_overlap_weighted"].abs()
    balance["abs_smd_trimmed"] = balance["smd_trimmed_unweighted"].abs()
    balance = balance.sort_values("abs_smd_unweighted", ascending=False)
    balance.to_csv(RESULTS_DIR / "tables" / f"06_compact_balance_{label}.csv", index=False)

    ps_table = pd.DataFrame(
        {
            "patient_id_normalized": df["patient_id_normalized"],
            "treatment": t,
            "propensity_score_oof": ps,
            "ipw_raw": ipw_raw,
            "ipw_truncated_p99": ipw,
            "overlap_weight": ow,
            "retained_trim_0.10_0.90": trim.astype(int),
        }
    )
    ps_table.to_csv(RESULTS_DIR / "tables" / f"06_compact_propensity_{label}.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(ps[t == 0], bins=20, density=True, alpha=0.55, label="Control")
    plt.hist(ps[t == 1], bins=20, density=True, alpha=0.55, label="Treated")
    plt.axvline(0.10, linestyle="--", linewidth=1)
    plt.axvline(0.90, linestyle="--", linewidth=1)
    plt.xlabel("Cross-fitted propensity score")
    plt.ylabel("Density")
    plt.title(label.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / f"06_compact_overlap_{label}.png", dpi=220)
    plt.close()

    auc = roc_auc_score(t, ps) if len(np.unique(t)) == 2 else np.nan
    ll = log_loss(t, ps, labels=[0, 1])
    outside = int((~trim).sum())
    retained_treated = int(np.sum(trim & (t == 1)))
    retained_control = int(np.sum(trim & (t == 0)))

    summary = {
        "cohort": label,
        "n": int(len(df)),
        "treated": int(t.sum()),
        "control": int((t == 0).sum()),
        "n_compact_covariates": len(feature_cols),
        "chosen_C_median": float(np.median(chosen_c)),
        "propensity_auc_oof": float(auc),
        "propensity_log_loss_oof": float(ll),
        "ps_min": float(np.min(ps)),
        "ps_p05": float(np.quantile(ps, 0.05)),
        "ps_median": float(np.median(ps)),
        "ps_p95": float(np.quantile(ps, 0.95)),
        "ps_max": float(np.max(ps)),
        "outside_0.10_0.90": outside,
        "outside_fraction": float(outside / len(df)),
        "trim_retained_total": int(trim.sum()),
        "trim_retained_treated": retained_treated,
        "trim_retained_control": retained_control,
        "ipw_raw_max": float(np.max(ipw_raw)),
        "ipw_p99_cap": cap,
        "ess_ipw_treated": ess(ipw[t == 1]),
        "ess_ipw_control": ess(ipw[t == 0]),
        "ess_overlap_treated": ess(ow[t == 1]),
        "ess_overlap_control": ess(ow[t == 0]),
        "max_abs_smd_unweighted": float(balance["abs_smd_unweighted"].max()),
        "max_abs_smd_ipw": float(balance["abs_smd_ipw"].max()),
        "max_abs_smd_overlap": float(balance["abs_smd_overlap"].max()),
        "max_abs_smd_trimmed": float(balance["abs_smd_trimmed"].max()),
        "mean_abs_smd_unweighted": float(balance["abs_smd_unweighted"].mean()),
        "mean_abs_smd_ipw": float(balance["abs_smd_ipw"].mean()),
        "mean_abs_smd_overlap": float(balance["abs_smd_overlap"].mean()),
        "mean_abs_smd_trimmed": float(balance["abs_smd_trimmed"].mean()),
    }

    # Conservative diagnostic label, not a causal-validity declaration.
    if (
        summary["outside_fraction"] <= 0.35
        and retained_treated >= 30
        and retained_control >= 30
        and summary["max_abs_smd_overlap"] <= 0.10
    ):
        summary["diagnostic_status"] = "PROCEED_WITH_OVERLAP_WEIGHTING_SENSITIVITY"
    elif retained_treated >= 20 and retained_control >= 20:
        summary["diagnostic_status"] = "LIMITED_SUPPORT_REQUIRES_RESTRICTED_ESTIMAND"
    else:
        summary["diagnostic_status"] = "INSUFFICIENT_COMMON_SUPPORT"

    return summary


def main() -> int:
    ensure_dirs()
    compact_dir = DERIVED_DIR / "compact_adjustment"
    files = sorted(compact_dir.glob("*_compact.csv"))
    if not files:
        raise FileNotFoundError(
            "No compact matrices found. Run 05_build_compact_adjustment.py first."
        )

    rows = []
    for path in files:
        print(f"Running compact cross-fitted overlap diagnostics: {path.stem}")
        rows.append(run_one(path))

    summary = pd.DataFrame(rows).sort_values("cohort")
    summary.to_csv(RESULTS_DIR / "tables" / "06_compact_overlap_summary.csv", index=False)

    old_path = RESULTS_DIR / "tables" / "04_overlap_summary.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old.rename(
            columns={
                "max_abs_smd": "legacy_max_abs_smd",
                "mean_abs_smd": "legacy_mean_abs_smd",
                "outside_0.10_0.90": "legacy_outside_0.10_0.90",
                "n_clinical_covariates": "legacy_n_clinical_covariates",
            }
        )
        keep = [
            c for c in (
                "cohort",
                "legacy_n_clinical_covariates",
                "legacy_outside_0.10_0.90",
                "legacy_max_abs_smd",
                "legacy_mean_abs_smd",
            )
            if c in old.columns
        ]
        comparison = summary.merge(old[keep], on="cohort", how="left")
        comparison.to_csv(
            RESULTS_DIR / "tables" / "06_legacy_vs_compact_overlap.csv",
            index=False,
        )

    cols = [
        "cohort",
        "n",
        "n_compact_covariates",
        "outside_fraction",
        "trim_retained_treated",
        "trim_retained_control",
        "max_abs_smd_unweighted",
        "max_abs_smd_ipw",
        "max_abs_smd_overlap",
        "diagnostic_status",
    ]
    print("\nCompact overlap summary:")
    print(summary[cols].to_string(index=False))
    print(f"\nSaved to: {RESULTS_DIR / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
