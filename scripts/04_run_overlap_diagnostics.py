from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


def smd(treated: pd.Series, control: pd.Series) -> float:
    x1 = pd.to_numeric(treated, errors="coerce")
    x0 = pd.to_numeric(control, errors="coerce")
    pooled = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((x1.mean() - x0.mean()) / pooled)


def clinical_columns(df: pd.DataFrame) -> list[str]:
    exact = {
        "ER_status", "PR_status", "HER2_status",
        "pathology_details.consistent_pathology_review",
        "pathology_details.lymph_nodes_positive",
        "pathology_details.lymph_nodes_tested",
    }
    candidate = [c for c in df.columns if str(c).startswith("CLIN_") or str(c) in exact]
    usable = []
    for c in candidate:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().sum() >= max(10, int(0.5 * len(df))) and numeric.nunique(dropna=True) > 1:
            usable.append(c)
    return usable


def run_one(path: Path, label: str):
    df = read_table(path)
    if "analysis_treatment" not in df.columns:
        raise ValueError(f"{path} has no analysis_treatment column")
    t = pd.to_numeric(df["analysis_treatment"], errors="coerce").astype(int)
    columns = clinical_columns(df)
    if not columns:
        raise ValueError(f"No usable clinical covariates in {path}")
    X = df[columns].apply(pd.to_numeric, errors="coerce")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42
        )),
    ])
    model.fit(X, t)
    ps = model.predict_proba(X)[:, 1]

    balance = pd.DataFrame({
        "cohort": label,
        "column": columns,
        "smd_unadjusted": [smd(X.loc[t == 1, c], X.loc[t == 0, c]) for c in columns],
        "missing_fraction": [float(X[c].isna().mean()) for c in columns],
    }).sort_values("smd_unadjusted", key=lambda s: s.abs(), ascending=False)
    balance.to_csv(RESULTS_DIR / "tables" / f"04_balance_{label}.csv", index=False)

    pd.DataFrame({
        "patient_id_normalized": df.get("patient_id_normalized", pd.Series(range(len(df)))),
        "treatment": t,
        "propensity_score": ps,
    }).to_csv(RESULTS_DIR / "tables" / f"04_propensity_{label}.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(ps[t == 0], bins=20, alpha=0.6, density=True, label="Control")
    plt.hist(ps[t == 1], bins=20, alpha=0.6, density=True, label="Treated")
    plt.xlabel("Estimated propensity score")
    plt.ylabel("Density")
    plt.title(label.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / f"04_propensity_overlap_{label}.png", dpi=200)
    plt.close()

    return {
        "cohort": label,
        "n": int(len(df)),
        "treated": int(t.sum()),
        "control": int(len(df) - t.sum()),
        "n_clinical_covariates": len(columns),
        "ps_min": float(np.min(ps)),
        "ps_p05": float(np.quantile(ps, 0.05)),
        "ps_median": float(np.median(ps)),
        "ps_p95": float(np.quantile(ps, 0.95)),
        "ps_max": float(np.max(ps)),
        "outside_0.10_0.90": int(((ps < 0.10) | (ps > 0.90)).sum()),
        "max_abs_smd": float(balance["smd_unadjusted"].abs().max()),
        "mean_abs_smd": float(balance["smd_unadjusted"].abs().mean()),
    }


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    files = sorted(cohort_dir.glob("*_hormone_hrpos_her2neg.csv")) + sorted(cohort_dir.glob("*_chemo_tnbc.csv"))
    if not files:
        raise FileNotFoundError("Run 03_create_analysis_cohorts.py first")

    rows = []
    for path in files:
        print(f"Running overlap diagnostics: {path.stem}")
        rows.append(run_one(path, path.stem))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "tables" / "04_overlap_summary.csv", index=False)
    print("\nOverlap summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved under: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
