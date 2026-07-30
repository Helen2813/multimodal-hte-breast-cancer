from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import RESULTS_DIR, ensure_dirs
from _stage11_utils import (
    LANDMARK,
    assemble_source_table,
    effective_sample_size,
    extract_hormone_start_days,
    readable_feature,
    standardized_mean_difference,
)


def crossfit_early_initiation(X: pd.DataFrame, a: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    counts = np.bincount(a, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        raise ValueError(f"Insufficient strategy counts for CCW feasibility: {counts}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=3900)
    ps = np.full(len(a), np.nan)
    tuning = []
    for fold, (train, test) in enumerate(splitter.split(X, a), start=1):
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
                        random_state=3900 + fold,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train], a[train])
        ps[test] = model.predict_proba(X.iloc[test])[:, 1]
        tuning.append(
            {
                "fold": fold,
                "chosen_C": float(model.named_steps["model"].C_[0]),
                "train_n": len(train),
                "test_n": len(test),
            }
        )
    return np.clip(ps, 0.01, 0.99), pd.DataFrame(tuning)


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    source, W_cols, metadata = assemble_source_table()
    timing = extract_hormone_start_days(source["patient_id_normalized"])
    source = source.merge(timing, on="patient_id_normalized", how="left", validate="one_to_one")

    ever = pd.to_numeric(source["analysis_treatment"], errors="raise").astype(int)
    start = pd.to_numeric(source["earliest_start_nonnegative"], errors="coerce")
    ambiguous = ever.eq(1) & (start.isna() | start.gt(3650))
    eligible = source.loc[~ambiguous].copy()
    start_e = pd.to_numeric(eligible["earliest_start_nonnegative"], errors="coerce")
    early = start_e.between(0, LANDMARK).astype(int)
    eligible["strategy_observed"] = np.where(
        early.eq(1),
        "initiate_by_180",
        "no_initiation_by_180",
    )
    eligible["early_initiation"] = early

    X = eligible[W_cols].apply(pd.to_numeric, errors="coerce")
    a = early.to_numpy(int)
    ps, tuning = crossfit_early_initiation(X, a)
    pbar = float(a.mean())
    stabilized = np.where(a == 1, pbar / ps, (1.0 - pbar) / (1.0 - ps))
    eligible["early_propensity_oof"] = ps
    eligible["stabilized_strategy_weight"] = stabilized

    weight_rows = []
    for arm, name in ((1, "initiate_by_180"), (0, "no_initiation_by_180")):
        w = stabilized[a == arm]
        weight_rows.append(
            {
                "strategy": name,
                "n": int((a == arm).sum()),
                "mean_weight": float(np.mean(w)),
                "sd_weight": float(np.std(w, ddof=1)),
                "median_weight": float(np.median(w)),
                "p95_weight": float(np.quantile(w, 0.95)),
                "p99_weight": float(np.quantile(w, 0.99)),
                "max_weight": float(np.max(w)),
                "fraction_gt_5": float(np.mean(w > 5)),
                "fraction_gt_10": float(np.mean(w > 10)),
                "ess": effective_sample_size(w),
            }
        )

    balance_rows = []
    for col in W_cols:
        x = pd.to_numeric(eligible[col], errors="coerce").to_numpy(float)
        balance_rows.append(
            {
                "feature": col,
                "feature_label": readable_feature(col),
                "smd_unweighted": standardized_mean_difference(x, a),
                "smd_stabilized_strategy_weight": standardized_mean_difference(x, a, stabilized),
            }
        )
    balance = pd.DataFrame(balance_rows)
    balance["abs_smd_weighted"] = balance["smd_stabilized_strategy_weight"].abs()
    balance = balance.sort_values("abs_smd_weighted", ascending=False)

    time = pd.to_numeric(eligible["analysis_time"], errors="coerce")
    event = pd.to_numeric(eligible["analysis_event"], errors="raise").astype(int)
    early_start = pd.to_numeric(eligible["earliest_start_nonnegative"], errors="coerce")

    early_natural_before_start = early.eq(1) & time.le(early_start)
    noinit_artificial_censor = early.eq(1) & early_start.lt(time)
    early_strategy_censor_at_180 = early.eq(0) & time.gt(LANDMARK)
    natural_before_landmark_noinit = early.eq(0) & time.le(LANDMARK)

    clone_flow = pd.DataFrame(
        [
            {"quantity": "source_patients", "n": len(source)},
            {"quantity": "excluded_ambiguous_timing", "n": int(ambiguous.sum())},
            {"quantity": "ccw_eligible_patients", "n": len(eligible)},
            {"quantity": "diagnosis_time_clones", "n": 2 * len(eligible)},
            {"quantity": "observed_initiate_by_180", "n": int(early.sum())},
            {"quantity": "observed_no_initiation_by_180", "n": int((early == 0).sum())},
            {
                "quantity": "no_initiation_clone_artificially_censored_at_start",
                "n": int(noinit_artificial_censor.sum()),
            },
            {
                "quantity": "initiation_clone_artificially_censored_at_day180",
                "n": int(early_strategy_censor_at_180.sum()),
            },
            {
                "quantity": "natural_event_or_censor_before_early_start",
                "n": int(early_natural_before_start.sum()),
            },
            {
                "quantity": "natural_event_or_censor_before_day180_noinit",
                "n": int(natural_before_landmark_noinit.sum()),
            },
            {"quantity": "events_by_day180", "n": int((event.eq(1) & time.le(LANDMARK)).sum())},
            {"quantity": "natural_censoring_by_day180", "n": int((event.eq(0) & time.le(LANDMARK)).sum())},
        ]
    )

    weight_summary = pd.DataFrame(weight_rows)
    max_p99 = float(weight_summary["p99_weight"].max())
    max_weight = float(weight_summary["max_weight"].max())
    min_ess = float(weight_summary["ess"].min())
    max_smd = float(balance["abs_smd_weighted"].max())
    if max_p99 <= 5 and max_weight <= 15 and min_ess >= 80 and max_smd <= 0.10:
        status = "CCW_SENSITIVITY_FEASIBLE"
    elif max_p99 <= 10 and max_weight <= 25 and min_ess >= 50 and max_smd <= 0.15:
        status = "CCW_SENSITIVITY_FEASIBLE_WITH_TRUNCATION"
    else:
        status = "CCW_NOT_STABLE_ENOUGH"

    decision = pd.DataFrame(
        [
            {
                **metadata,
                "eligible_n": len(eligible),
                "early_initiators": int(early.sum()),
                "no_initiation_by_180": int((early == 0).sum()),
                "excluded_ambiguous_timing": int(ambiguous.sum()),
                "max_weight_p99": max_p99,
                "max_weight": max_weight,
                "minimum_strategy_ess": min_ess,
                "max_weighted_smd": max_smd,
                "feasibility_status": status,
                "important_note": (
                    "This is a baseline-adherence weight diagnostic, not the final clone-censor-weight effect estimator."
                ),
            }
        ]
    )

    eligible[
        [
            "patient_id_normalized",
            "strategy_observed",
            "early_initiation",
            "early_propensity_oof",
            "stabilized_strategy_weight",
            "analysis_event",
            "analysis_time",
            "earliest_start_nonnegative",
        ]
    ].to_csv(table_dir / "39_ccw_patient_feasibility.csv", index=False)
    tuning.to_csv(table_dir / "39_ccw_propensity_tuning.csv", index=False)
    clone_flow.to_csv(table_dir / "39_ccw_clone_flow.csv", index=False)
    weight_summary.to_csv(table_dir / "39_ccw_weight_summary.csv", index=False)
    balance.to_csv(table_dir / "39_ccw_balance.csv", index=False)
    decision.to_csv(table_dir / "39_ccw_feasibility_decision.csv", index=False)

    print("=" * 115)
    print("STAGE 39 — CLONE-CENSOR-WEIGHT FEASIBILITY")
    print("=" * 115)
    print(decision.to_string(index=False))
    print("\nClone flow")
    print(clone_flow.to_string(index=False))
    print("\nStrategy-weight diagnostics")
    print(weight_summary.to_string(index=False))
    print("\nTop weighted baseline imbalances")
    print(
        balance[
            [
                "feature_label",
                "smd_unweighted",
                "smd_stabilized_strategy_weight",
            ]
        ].head(15).to_string(index=False)
    )
    print(
        "\nThis stage assesses whether a full CCW sensitivity is numerically plausible; "
        "it does not estimate a CCW treatment effect."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
