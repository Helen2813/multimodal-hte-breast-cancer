from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import RESULTS_DIR, ensure_dirs
from _stage11_utils import (
    LANDMARK,
    assemble_landmark_table,
    readable_feature,
    standardized_mean_difference,
)


def later_prediction_auc(X: pd.DataFrame, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    counts = np.bincount(y, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        return np.nan, pd.DataFrame()

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=3700)
    pred = np.full(len(y), np.nan)
    tuning = []
    for fold, (train, test) in enumerate(splitter.split(X, y), start=1):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegressionCV(
                        Cs=[0.003, 0.01, 0.03, 0.1, 0.3],
                        cv=3,
                        scoring="neg_log_loss",
                        max_iter=6000,
                        n_jobs=-1,
                        random_state=3700 + fold,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train], y[train])
        pred[test] = model.predict_proba(X.iloc[test])[:, 1]
        tuning.append(
            {
                "fold": fold,
                "chosen_C": float(model.named_steps["model"].C_[0]),
                "train_n": len(train),
                "test_n": len(test),
            }
        )
    return float(roc_auc_score(y, pred)), pd.DataFrame(tuning)


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    landmark, W_cols, metadata = assemble_landmark_table()

    treatment = pd.to_numeric(landmark["analysis_treatment"], errors="raise").astype(int)
    control = landmark.loc[treatment.eq(0)].copy()
    control["control_component"] = np.where(
        pd.to_numeric(control["later_initiator"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1),
        "later_initiator",
        "no_recorded_later_initiation",
    )
    later = control["control_component"].eq("later_initiator").astype(int).to_numpy()

    composition_rows = []
    for component, group in control.groupby("control_component"):
        start_series = (
            pd.to_numeric(group["earliest_start_nonnegative"], errors="coerce")
            if "earliest_start_nonnegative" in group.columns
            else pd.Series(np.nan, index=group.index)
        )
        composition_rows.append(
            {
                "control_component": component,
                "n": len(group),
                "events": int(pd.to_numeric(group["analysis_event"], errors="raise").sum()),
                "event_rate": float(pd.to_numeric(group["analysis_event"], errors="raise").mean()),
                "median_followup_post_landmark_days": float(
                    pd.to_numeric(group["analysis_time"], errors="coerce").median()
                ),
                "median_diagnosis_year": float(
                    pd.to_numeric(group["diagnosis_year"], errors="coerce").median()
                ),
                "median_start_day": float(start_series.median()),
            }
        )

    balance_rows = []
    for col in W_cols:
        x = pd.to_numeric(control[col], errors="coerce").to_numpy(float)
        balance_rows.append(
            {
                "feature": col,
                "feature_label": readable_feature(col),
                "later_mean": float(np.nanmean(x[later == 1])) if np.isfinite(x[later == 1]).any() else np.nan,
                "never_mean": float(np.nanmean(x[later == 0])) if np.isfinite(x[later == 0]).any() else np.nan,
                "smd_later_vs_never": standardized_mean_difference(x, later),
                "missing_fraction": float(np.mean(~np.isfinite(x))),
            }
        )
    balance = pd.DataFrame(balance_rows)
    balance["abs_smd"] = balance["smd_later_vs_never"].abs()
    balance = balance.sort_values("abs_smd", ascending=False)

    auc, tuning = later_prediction_auc(
        control[W_cols].apply(pd.to_numeric, errors="coerce"),
        later,
    )

    year = pd.to_numeric(landmark["diagnosis_year"], errors="coerce")
    era = pd.cut(
        year,
        [-np.inf, 2004, 2009, np.inf],
        labels=["<=2004", "2005-2009", ">=2010"],
    ).astype("object").fillna("unknown")
    era_rows = []
    for era_name in sorted(era.unique()):
        for arm in (0, 1):
            mask = era.eq(era_name) & treatment.eq(arm)
            era_rows.append(
                {
                    "era": era_name,
                    "arm": "initiated_by_180" if arm == 1 else "not_initiated_by_180",
                    "n": int(mask.sum()),
                    "events": int(
                        pd.to_numeric(landmark.loc[mask, "analysis_event"], errors="raise").sum()
                    ),
                }
            )
    era_df = pd.DataFrame(era_rows)
    modern = era_df[era_df["era"].isin(["2005-2009", ">=2010"])]
    formal_interaction_feasible = int(
        not modern.empty and (modern["n"] >= 40).all() and (modern["events"] >= 8).all()
    )
    era_df["formal_interaction_feasible"] = formal_interaction_feasible
    era_df["decision"] = (
        "formal_interaction_allowed"
        if formal_interaction_feasible
        else "descriptive_only_due_to_sparse_event_cells"
    )

    composition = pd.DataFrame(composition_rows)
    composition["later_prediction_oof_auc"] = auc
    composition["landmark_day"] = LANDMARK

    composition.to_csv(table_dir / "37_control_strategy_composition.csv", index=False)
    balance.to_csv(table_dir / "37_later_vs_never_balance.csv", index=False)
    tuning.to_csv(table_dir / "37_later_prediction_tuning.csv", index=False)
    era_df.to_csv(table_dir / "37_era_interaction_feasibility.csv", index=False)

    print("=" * 115)
    print("STAGE 37 — CONTROL STRATEGY COMPOSITION")
    print("=" * 115)
    print(pd.DataFrame([metadata]).to_string(index=False))
    print("\nControl components")
    print(composition.to_string(index=False))
    print(f"\nOOF AUC predicting later initiation: {auc:.3f}")
    print("\nTop later-versus-never baseline differences")
    print(
        balance[["feature_label", "later_mean", "never_mean", "smd_later_vs_never"]]
        .head(15)
        .to_string(index=False)
    )
    print("\nEra interaction feasibility")
    print(era_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
