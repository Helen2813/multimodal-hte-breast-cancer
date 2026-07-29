from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


NON_RNA = ["CNV", "Mutation", "Methylation", "miRNA", "Protein"]


def smd(x: pd.Series, selected: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce")
    a = values[selected == 1]
    b = values[selected == 0]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def cross_fitted_auc(X: pd.DataFrame, y: pd.Series) -> float:
    counts = y.value_counts()
    folds = min(5, int(counts.min()))
    if folds < 3:
        return np.nan
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pred = np.full(len(y), np.nan)
    for fold, (train, test) in enumerate(splitter.split(X, y), start=1):
        inner = min(4, int(y.iloc[train].value_counts().min()))
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegressionCV(
                        Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                        cv=max(2, inner),
                        scoring="neg_log_loss",
                        max_iter=5000,
                        n_jobs=-1,
                        random_state=100 + fold,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train], y.iloc[train])
        pred[test] = model.predict_proba(X.iloc[test])[:, 1]
    return float(roc_auc_score(y, pred))


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    compact_dir = DERIVED_DIR / "compact_adjustment"
    manifest_path = DERIVED_DIR / "manifests" / "07_patient_modality_availability.csv"
    table_dir = RESULTS_DIR / "tables"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    availability = read_table(manifest_path)

    cohort_names = ["outer_hormone_hrpos_her2neg", "outer_chemo_tnbc"]
    summary_rows = []
    balance_rows = []
    jaccard_rows = []
    id_rows = []

    for cohort_name in cohort_names:
        cohort_path = cohort_dir / f"{cohort_name}.csv"
        compact_path = compact_dir / f"{cohort_name}_compact.csv"
        if not cohort_path.exists() or not compact_path.exists():
            raise FileNotFoundError(f"Missing cohort/compact file for {cohort_name}")

        cohort = read_table(cohort_path)
        compact = read_table(compact_path)
        avail = availability[availability["cohort"] == cohort_name].copy()

        merged = compact.merge(
            avail.drop(columns=["cohort"]),
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        for modality in NON_RNA:
            col = f"available_{modality}"
            if col not in merged.columns:
                raise ValueError(f"Missing {col}")

        merged["complete_non_rna_omics"] = merged[
            [f"available_{m}" for m in NON_RNA]
        ].min(axis=1).astype(int)
        if "available_RNA" in merged.columns:
            merged["complete_six_omics"] = (
                merged["complete_non_rna_omics"] * merged["available_RNA"]
            ).astype(int)
        else:
            merged["complete_six_omics"] = merged["complete_non_rna_omics"]

        selected = merged["complete_six_omics"].astype(int)
        t = pd.to_numeric(merged["analysis_treatment"], errors="raise").astype(int)
        e = pd.to_numeric(merged["analysis_event"], errors="raise").astype(int)
        W_cols = [c for c in merged.columns if c.startswith("W_")]

        X = merged[W_cols + ["analysis_treatment"]].apply(
            pd.to_numeric, errors="coerce"
        )
        selection_auc = cross_fitted_auc(X, selected)

        summary_rows.append(
            {
                "cohort": cohort_name,
                "n": len(merged),
                "complete_six_omics_n": int(selected.sum()),
                "complete_six_omics_fraction": float(selected.mean()),
                "treated_fraction_available": float(selected[t == 1].mean()),
                "control_fraction_available": float(selected[t == 0].mean()),
                "event_fraction_available": float(selected[e == 1].mean())
                if (e == 1).any() else np.nan,
                "nonevent_fraction_available": float(selected[e == 0].mean())
                if (e == 0).any() else np.nan,
                "selection_auc_oof_W_plus_treatment": selection_auc,
            }
        )

        for col in W_cols:
            balance_rows.append(
                {
                    "cohort": cohort_name,
                    "feature": col,
                    "smd_complete_vs_incomplete": smd(merged[col], selected),
                }
            )

        masks = {m: merged[f"available_{m}"].astype(bool) for m in ["RNA"] + NON_RNA}
        for m1, mask1 in masks.items():
            for m2, mask2 in masks.items():
                union = (mask1 | mask2).sum()
                inter = (mask1 & mask2).sum()
                jaccard_rows.append(
                    {
                        "cohort": cohort_name,
                        "modality_1": m1,
                        "modality_2": m2,
                        "intersection": int(inter),
                        "union": int(union),
                        "jaccard": float(inter / union) if union else np.nan,
                        "identical_masks": int(mask1.equals(mask2)),
                    }
                )

        id_rows.extend(
            pd.DataFrame(
                {
                    "cohort": cohort_name,
                    "patient_id_normalized": merged["patient_id_normalized"],
                    "complete_six_omics": selected,
                }
            ).to_dict("records")
        )

    pd.DataFrame(summary_rows).to_csv(
        table_dir / "11_complete_omics_selection_summary.csv", index=False
    )
    balance = pd.DataFrame(balance_rows)
    balance["abs_smd"] = balance["smd_complete_vs_incomplete"].abs()
    balance.sort_values(["cohort", "abs_smd"], ascending=[True, False]).to_csv(
        table_dir / "11_complete_omics_selection_balance.csv", index=False
    )
    pd.DataFrame(jaccard_rows).to_csv(
        table_dir / "11_modality_availability_jaccard.csv", index=False
    )
    pd.DataFrame(id_rows).to_csv(
        DERIVED_DIR / "manifests" / "11_complete_omics_patient_ids.csv",
        index=False,
    )

    print("\nComplete multi-omics selection summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
