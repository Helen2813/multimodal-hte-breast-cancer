from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _stage8_utils import (
    HORIZONS,
    INTERVAL_DAYS,
    build_interval_rows,
    cohort_key,
    get_repeat_assignments,
    load_compact_with_year,
    reverse_km_censoring,
    rmst_ipcw_pseudooutcome,
)


def classical_censoring_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    cv=3,
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


def ai_censoring_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def crossfit_censoring(
    df: pd.DataFrame,
    compact: pd.DataFrame,
    features: list[str],
    assignments: pd.DataFrame,
    strategy: str,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    merged = df.merge(
        assignments[["patient_id_normalized", "fold"]],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    compact_m = compact.merge(
        merged[["patient_id_normalized", "fold"]],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    W = compact_m[features].apply(pd.to_numeric, errors="coerce")
    long_df, starts, ends = build_interval_rows(
        compact_m, W, INTERVAL_DAYS, max(HORIZONS)
    )
    feature_cols = features + [
        "treatment",
        "interval",
        "interval_start",
    ]

    n = len(compact_m)
    K = len(starts)
    q = np.full((n, K), np.nan)
    long_true = []
    long_pred = []

    for fold in sorted(compact_m["fold"].unique()):
        train_patients = set(
            compact_m.loc[
                compact_m["fold"] != fold, "patient_id_normalized"
            ]
        )
        test_rows = set(
            compact_m.index[compact_m["fold"] == fold].tolist()
        )
        train_mask = long_df["patient_row"].map(
            lambda i: compact_m.loc[int(i), "patient_id_normalized"]
            in train_patients
        )
        test_mask = long_df["patient_row"].isin(test_rows)

        X_train = long_df.loc[train_mask, feature_cols]
        y_train = long_df.loc[train_mask, "censor_event"].astype(int)
        X_test = long_df.loc[test_mask, feature_cols]
        y_test = long_df.loc[test_mask, "censor_event"].astype(int)

        if strategy == "classical":
            model = classical_censoring_model(2500 + int(fold))
        elif strategy == "ai_boosted":
            model = ai_censoring_model(2500 + int(fold))
        else:
            raise ValueError(strategy)

        model.fit(X_train, y_train)
        pred = np.clip(model.predict_proba(X_test)[:, 1], 1e-5, 0.95)
        test_long = long_df.loc[test_mask, ["patient_row", "interval"]]
        for (_, r), p in zip(test_long.iterrows(), pred):
            q[int(r["patient_row"]), int(r["interval"])] = p

        long_true.extend(y_test.tolist())
        long_pred.extend(pred.tolist())

    if np.isnan(q).any():
        # Intervals after each patient's observed time are irrelevant; set to zero.
        q = np.nan_to_num(q, nan=0.0)

    G_start = np.ones((n, K), dtype=float)
    for k in range(1, K):
        G_start[:, k] = (
            G_start[:, k - 1] * (1.0 - q[:, k - 1])
        )
    G_start = np.clip(G_start, 0.01, 1.0)

    y_arr = np.asarray(long_true, dtype=int)
    p_arr = np.asarray(long_pred, dtype=float)
    metrics = {
        "long_rows_oof": len(y_arr),
        "censor_events_long": int(y_arr.sum()),
        "log_loss_oof": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        "brier_oof": float(brier_score_loss(y_arr, p_arr)),
        "auc_oof": float(roc_auc_score(y_arr, p_arr))
        if len(np.unique(y_arr)) == 2
        else np.nan,
    }
    return G_start, metrics, starts, ends


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    pseudo_dir = DERIVED_DIR / "ipcw_pseudooutcomes"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("STAGE 25 — CENSORING FEASIBILITY AND IPCW RMST PSEUDO-OUTCOMES")
    print("=" * 110)

    cohort_names = [
        "outer_hormone_hrpos_her2neg",
        "outer_chemo_tnbc",
        "complete_case_hormone_hrpos_her2neg",
    ]
    reverse_rows = []
    model_rows = []
    pseudo_summary_rows = []

    for cohort in cohort_names:
        path = (
            DERIVED_DIR
            / "verified_cohorts"
            / f"{cohort}_verified.csv"
        )
        if not path.exists():
            continue
        df = read_table(path)
        compact, features = load_compact_with_year(cohort)
        assignments = get_repeat_assignments(cohort, repeat=1)

        print("\n" + "=" * 110)
        print(f"COHORT: {cohort}")
        print(
            f"n={len(df)}, treated="
            f"{int(pd.to_numeric(df['analysis_treatment']).sum())}, "
            f"controls="
            f"{int((1-pd.to_numeric(df['analysis_treatment'])).sum())}, "
            f"events={int(pd.to_numeric(df['analysis_event']).sum())}"
        )

        t = pd.to_numeric(
            df["analysis_treatment"], errors="raise"
        ).astype(int).to_numpy()
        times = pd.to_numeric(
            df["analysis_time"], errors="coerce"
        ).to_numpy(float)
        events = pd.to_numeric(
            df["analysis_event"], errors="raise"
        ).astype(int).to_numpy()

        print("\nReverse-KM probability of remaining uncensored")
        arm_rows = []
        for arm, arm_name in ((0, "control"), (1, "treated")):
            mask = t == arm
            values = reverse_km_censoring(
                times[mask], events[mask], HORIZONS
            )
            for horizon, value in values.items():
                row = {
                    "cohort": cohort,
                    "arm": arm_name,
                    "horizon_days": horizon,
                    "horizon_years": horizon / 365.25,
                    "reverse_km_uncensored_probability": value,
                    "at_risk_observed_time_ge_horizon": int(
                        np.sum(times[mask] >= horizon)
                    ),
                    "arm_n": int(mask.sum()),
                    "arm_events": int(events[mask].sum()),
                }
                reverse_rows.append(row)
                arm_rows.append(row)
        print(pd.DataFrame(arm_rows).to_string(index=False))

        for strategy in ("classical", "ai_boosted"):
            G, metrics, starts, ends = crossfit_censoring(
                df, compact, features, assignments, strategy
            )
            model_row = {
                "cohort": cohort,
                "strategy": strategy,
                "n": len(df),
                "n_covariates": len(features),
                **metrics,
                "G_start_min": float(G.min()),
                "G_start_p01": float(np.quantile(G, 0.01)),
                "G_start_median": float(np.median(G)),
            }
            model_rows.append(model_row)
            print(f"\nCensoring nuisance model: {strategy}")
            print(pd.DataFrame([model_row]).to_string(index=False))

            output = pd.DataFrame(
                {
                    "patient_id_normalized": compact[
                        "patient_id_normalized"
                    ],
                    "analysis_treatment": compact[
                        "analysis_treatment"
                    ],
                    "analysis_event": compact["analysis_event"],
                    "analysis_time": compact["analysis_time"],
                }
            )
            observed_time = pd.to_numeric(
                compact["analysis_time"], errors="coerce"
            ).to_numpy(float)

            for horizon in HORIZONS:
                pseudo = rmst_ipcw_pseudooutcome(
                    observed_time,
                    G,
                    starts,
                    ends,
                    horizon,
                    min_g=0.05,
                )
                col = f"rmst_ipcw_{int(horizon)}d"
                output[col] = pseudo
                pseudo_summary_rows.append(
                    {
                        "cohort": cohort,
                        "strategy": strategy,
                        "horizon_days": horizon,
                        "mean": float(np.mean(pseudo)),
                        "sd": float(np.std(pseudo, ddof=1)),
                        "minimum": float(np.min(pseudo)),
                        "p01": float(np.quantile(pseudo, 0.01)),
                        "median": float(np.median(pseudo)),
                        "p99": float(np.quantile(pseudo, 0.99)),
                        "maximum": float(np.max(pseudo)),
                        "fraction_above_1.5x_horizon": float(
                            np.mean(pseudo > 1.5 * horizon)
                        ),
                    }
                )
            output.to_csv(
                pseudo_dir / f"25_ipcw_rmst_{cohort}_{strategy}.csv",
                index=False,
            )

    reverse_df = pd.DataFrame(reverse_rows)
    model_df = pd.DataFrame(model_rows)
    pseudo_df = pd.DataFrame(pseudo_summary_rows)
    reverse_df.to_csv(
        table_dir / "25_reverse_km_censoring_summary.csv",
        index=False,
    )
    model_df.to_csv(
        table_dir / "25_censoring_model_summary.csv", index=False
    )
    pseudo_df.to_csv(
        table_dir / "25_ipcw_pseudooutcome_summary.csv",
        index=False,
    )

    gate_rows = []
    for cohort, group in reverse_df.groupby("cohort"):
        for horizon, hgroup in group.groupby("horizon_days"):
            minimum_g = float(
                hgroup["reverse_km_uncensored_probability"].min()
            )
            minimum_at_risk = int(
                hgroup["at_risk_observed_time_ge_horizon"].min()
            )
            if minimum_g >= 0.30 and minimum_at_risk >= 20:
                status = "STRONG_HORIZON"
            elif minimum_g >= 0.15 and minimum_at_risk >= 10:
                status = "SENSITIVITY_HORIZON"
            else:
                status = "WEAK_HORIZON"
            gate_rows.append(
                {
                    "cohort": cohort,
                    "horizon_days": horizon,
                    "minimum_arm_uncensored_probability": minimum_g,
                    "minimum_arm_at_risk": minimum_at_risk,
                    "horizon_status": status,
                }
            )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(
        table_dir / "25_horizon_feasibility_gate.csv", index=False
    )

    print("\n" + "=" * 110)
    print("FINAL STAGE 25 HORIZON FEASIBILITY GATE")
    print("=" * 110)
    print(gate.to_string(index=False))
    print("\nCensoring nuisance-model comparison")
    print(model_df.to_string(index=False))
    print("\nIPCW pseudo-outcome distribution")
    print(pseudo_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
