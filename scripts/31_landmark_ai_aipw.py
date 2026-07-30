from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _stage9_utils import (
    LANDMARKS,
    POST_LANDMARK_HORIZONS,
    aipw_ato,
    crossfit_arm_outcomes,
    interval_long,
    ipcw_rmst,
    reverse_km,
    ridge_regression,
)


COHORTS = (
    "outer_hormone_hrpos_her2neg",
    "outer_chemo_tnbc",
)
G_MINS = (0.05, 0.10)


def censor_model(kind: str, seed: int):
    if kind == "classical":
        return Pipeline(
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
                        random_state=seed,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=140,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=2.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def outcome_factory(kind: str):
    def factory(fold: int, arm: int):
        if kind == "classical":
            return ridge_regression()
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=160,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=3.0,
                        random_state=31000 + fold + 10 * arm,
                    ),
                ),
            ]
        )
    return factory


def fit_censoring(
    compact: pd.DataFrame,
    features: list[str],
    splits: pd.DataFrame,
    horizon: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    assignment = splits[splits["repeat"] == 1][
        ["patient_id_normalized", "fold"]
    ]
    base = compact.merge(
        assignment,
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    X = base[features].apply(pd.to_numeric, errors="coerce")
    long_df, starts, ends = interval_long(
        base, X, horizon
    )
    feature_cols = features + [
        "treatment",
        "interval",
        "interval_start",
    ]

    n = len(base)
    K = len(starts)
    q = np.full((n, K), np.nan)
    all_y = []
    all_p = []

    for fold in sorted(base["fold"].unique()):
        train_patients = set(
            base.loc[
                base["fold"] != fold, "patient_id_normalized"
            ]
        )
        test_rows = set(
            base.index[base["fold"] == fold].tolist()
        )
        train_mask = long_df["patient_row"].map(
            lambda i: base.loc[int(i), "patient_id_normalized"]
            in train_patients
        )
        test_mask = long_df["patient_row"].isin(test_rows)

        model = censor_model(kind, 3100 + int(fold))
        model.fit(
            long_df.loc[train_mask, feature_cols],
            long_df.loc[train_mask, "censor_event"].astype(int),
        )
        pred = np.clip(
            model.predict_proba(
                long_df.loc[test_mask, feature_cols]
            )[:, 1],
            1e-5,
            0.95,
        )
        test_info = long_df.loc[
            test_mask, ["patient_row", "interval"]
        ]
        for (_, row), p in zip(test_info.iterrows(), pred):
            q[int(row["patient_row"]), int(row["interval"])] = p
        all_y.extend(
            long_df.loc[test_mask, "censor_event"].astype(int).tolist()
        )
        all_p.extend(pred.tolist())

    q = np.nan_to_num(q, nan=0.0)
    G = np.ones((n, K), dtype=float)
    for k in range(1, K):
        G[:, k] = G[:, k - 1] * (1.0 - q[:, k - 1])
    G = np.clip(G, 0.005, 1.0)

    y = np.asarray(all_y, int)
    p = np.asarray(all_p, float)
    metrics = {
        "censor_oof_log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "censor_oof_brier": float(brier_score_loss(y, p)),
        "censor_oof_auc": float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else np.nan,
        "G_min_raw": float(G.min()),
        "G_p01_raw": float(np.quantile(G, 0.01)),
        "G_median_raw": float(np.median(G)),
    }
    return G, starts, ends, metrics


def fixed_bootstrap(
    score: np.ndarray,
    h: np.ndarray,
    seed: int,
    n_boot: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(h)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        estimates.append(float(score[idx].sum() / h[idx].sum()))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def main() -> int:
    ensure_dirs()
    compact_dir = DERIVED_DIR / "landmark_compact"
    split_dir = DERIVED_DIR / "landmark_splits"
    weight_dir = DERIVED_DIR / "landmark_weights"
    pseudo_dir = DERIVED_DIR / "landmark_pseudooutcomes"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    print("=" * 115)
    print("STAGE 31 — LANDMARK AI/CLASSICAL AIPW-RMST STABILITY")
    print("=" * 115)

    rows = []
    censor_rows = []
    reverse_rows = []
    skipped_rows = []

    balance_summary_path = (
        table_dir / "30_landmark_balance_summary.csv"
    )
    if not balance_summary_path.exists():
        raise FileNotFoundError(balance_summary_path)
    balance_summary = read_table(balance_summary_path)

    for cohort in COHORTS:
        for landmark in LANDMARKS:
            key = f"{cohort}_landmark{landmark}"

            status_rows = balance_summary[
                (balance_summary["cohort"] == cohort)
                & (balance_summary["landmark_day"] == landmark)
            ]
            if status_rows.empty:
                raise ValueError(
                    f"Stage 30 balance status missing for {key}."
                )

            balance_status = str(
                status_rows.iloc[0]["balance_status"]
            )
            if balance_status == "LANDMARK_NOT_READY":
                skipped = {
                    "cohort": cohort,
                    "landmark_day": landmark,
                    "balance_status": balance_status,
                    "reason": (
                        "Skipped before AIPW modeling because Stage 30 "
                        "found inadequate overlap/balance, ESS, or event counts."
                    ),
                }
                skipped_rows.append(skipped)
                print("\n" + "=" * 115)
                print(f"SKIPPING {key}")
                print(pd.DataFrame([skipped]).to_string(index=False))
                continue
            compact = read_table(
                compact_dir / f"{key}_compact.csv"
            )
            splits = read_table(split_dir / f"{key}_splits.csv")
            weights = read_table(
                weight_dir / f"{key}_weights.csv"
            )
            base = compact.merge(
                weights[
                    [
                        "patient_id_normalized",
                        "propensity_score_oof",
                    ]
                ],
                on="patient_id_normalized",
                how="inner",
                validate="one_to_one",
            )
            features = [
                c for c in compact.columns if c.startswith("W_")
            ] + ["diagnosis_year", "diagnosis_year_missing"]
            a = base["analysis_treatment"].astype(int).to_numpy()
            e = base["propensity_score_oof"].to_numpy(float)
            time = base["analysis_time"].to_numpy(float)
            event = base["analysis_event"].astype(int).to_numpy()

            print("\n" + "=" * 115)
            print(f"{cohort}, landmark={landmark}")
            print(
                f"n={len(base)}, treated={int(a.sum())}, "
                f"controls={int((a == 0).sum())}, events={int(event.sum())}"
            )

            for arm, name in ((0, "control"), (1, "treated")):
                values = reverse_km(
                    time[a == arm],
                    event[a == arm],
                    POST_LANDMARK_HORIZONS,
                )
                for horizon, g in values.items():
                    reverse_rows.append(
                        {
                            "cohort": cohort,
                            "landmark_day": landmark,
                            "arm": name,
                            "horizon_days_post_landmark": horizon,
                            "reverse_km_uncensored": g,
                            "at_risk": int(
                                np.sum(time[a == arm] >= horizon)
                            ),
                        }
                    )

            for horizon in POST_LANDMARK_HORIZONS:
                for kind in ("classical", "ai_boosted"):
                    G, starts, ends, metrics = fit_censoring(
                        compact, features, splits, horizon, kind
                    )
                    censor_rows.append(
                        {
                            "cohort": cohort,
                            "landmark_day": landmark,
                            "horizon_days_post_landmark": horizon,
                            "strategy": kind,
                            **metrics,
                        }
                    )

                    assignment = splits[splits["repeat"] == 1][
                        ["patient_id_normalized", "fold"]
                    ]
                    assignment = base[
                        ["patient_id_normalized"]
                    ].merge(
                        assignment,
                        on="patient_id_normalized",
                        how="left",
                        validate="one_to_one",
                    )

                    for g_min in G_MINS:
                        y = ipcw_rmst(
                            time,
                            G,
                            starts,
                            ends,
                            horizon,
                            g_min,
                        )
                        X = base[features].apply(
                            pd.to_numeric, errors="coerce"
                        )
                        mu0, mu1 = crossfit_arm_outcomes(
                            X,
                            y,
                            a,
                            assignment,
                            outcome_factory(kind),
                        )
                        theta, influence = aipw_ato(
                            y, a, e, mu0, mu1
                        )
                        se = float(
                            np.std(influence, ddof=1)
                            / np.sqrt(len(influence))
                        )
                        h = e * (1 - e)
                        score = (
                            h * (mu1 - mu0)
                            + h * a / np.clip(e, 0.02, 0.98)
                            * (y - mu1)
                            - h * (1 - a)
                            / np.clip(1 - e, 0.02, 0.98)
                            * (y - mu0)
                        )
                        boot_low, boot_high = fixed_bootstrap(
                            score,
                            h,
                            seed=31000
                            + landmark
                            + int(horizon)
                            + int(100 * g_min)
                            + (1 if kind == "ai_boosted" else 0),
                        )
                        row = {
                            "cohort": cohort,
                            "landmark_day": landmark,
                            "horizon_days_post_landmark": horizon,
                            "strategy": kind,
                            "g_min": g_min,
                            "n": len(base),
                            "treated": int(a.sum()),
                            "control": int((a == 0).sum()),
                            "events": int(event.sum()),
                            "aipw_ato_rmst_difference_days": theta,
                            "influence_se_days": se,
                            "influence_ci_low_days": theta - 1.96 * se,
                            "influence_ci_high_days": theta + 1.96 * se,
                            "fixed_boot_ci_low_days": boot_low,
                            "fixed_boot_ci_high_days": boot_high,
                            "pseudo_mean": float(y.mean()),
                            "pseudo_sd": float(y.std(ddof=1)),
                            "pseudo_p99": float(np.quantile(y, 0.99)),
                            "pseudo_max": float(y.max()),
                            "fraction_above_1_5x_horizon": float(
                                np.mean(y > 1.5 * horizon)
                            ),
                        }
                        rows.append(row)
                        print("\n" + "-" * 100)
                        print(
                            f"horizon={horizon}, strategy={kind}, Gmin={g_min}"
                        )
                        print(pd.DataFrame([row]).to_string(index=False))

                        pd.DataFrame(
                            {
                                "patient_id_normalized": base[
                                    "patient_id_normalized"
                                ],
                                "analysis_treatment": a,
                                "analysis_event": event,
                                "analysis_time": time,
                                "rmst_ipcw": y,
                            }
                        ).to_csv(
                            pseudo_dir
                            / f"{key}_{int(horizon)}d_{kind}_g{str(g_min).replace('.', '')}.csv",
                            index=False,
                        )

    results = pd.DataFrame(rows)
    censor = pd.DataFrame(censor_rows)
    reverse = pd.DataFrame(reverse_rows)
    if results.empty:
        raise RuntimeError(
            "No landmark design passed the Stage 30 readiness gate."
        )

    results.to_csv(
        table_dir / "31_landmark_aipw_results.csv", index=False
    )
    censor.to_csv(
        table_dir / "31_landmark_censoring_models.csv", index=False
    )
    reverse.to_csv(
        table_dir / "31_landmark_reverse_km.csv", index=False
    )

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df.to_csv(
        table_dir / "31_skipped_not_ready_designs.csv",
        index=False,
    )
    if not skipped_df.empty:
        print("\n" + "=" * 115)
        print("DESIGNS SKIPPED BEFORE AIPW MODELING")
        print("=" * 115)
        print(skipped_df.to_string(index=False))

    gates = []
    for (
        cohort,
        landmark,
        horizon,
    ), group in results.groupby(
        ["cohort", "landmark_day", "horizon_days_post_landmark"]
    ):
        directions = np.sign(
            group["aipw_ato_rmst_difference_days"].to_numpy()
        )
        stable_direction = int(np.all(directions == directions[0]))
        spread = float(
            group["aipw_ato_rmst_difference_days"].max()
            - group["aipw_ato_rmst_difference_days"].min()
        )
        max_tail = float(
            group["fraction_above_1_5x_horizon"].max()
        )
        max_halfwidth = float(
            (
                group["influence_ci_high_days"]
                - group["influence_ci_low_days"]
            ).max()
            / 2.0
        )
        if (
            stable_direction
            and spread <= 90
            and max_tail <= 0.08
            and max_halfwidth <= 180
        ):
            status = "LANDMARK_PAPER_A_FEASIBLE"
        elif stable_direction:
            status = "DIRECTION_STABLE_BUT_SENSITIVE"
        else:
            status = "UNSTABLE"
        gates.append(
            {
                "cohort": cohort,
                "landmark_day": landmark,
                "horizon_days_post_landmark": horizon,
                "direction_stable": stable_direction,
                "strategy_and_truncation_spread_days": spread,
                "maximum_tail_fraction": max_tail,
                "maximum_ci_halfwidth_days": max_halfwidth,
                "feasibility_status": status,
            }
        )
    gate_df = pd.DataFrame(gates)
    gate_df.to_csv(
        table_dir / "31_landmark_paperA_gate.csv", index=False
    )

    print("\n" + "=" * 115)
    print("FINAL LANDMARK PAPER A GATE")
    print("=" * 115)
    print(gate_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
