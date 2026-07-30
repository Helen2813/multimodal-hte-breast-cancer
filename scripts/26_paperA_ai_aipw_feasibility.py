from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _stage8_utils import (
    get_repeat_assignments,
    load_compact_with_year,
)


HORIZONS = (1095.0, 1825.0)
BOOTSTRAPS = 500


def outcome_model(strategy: str, seed: int):
    if strategy == "classical":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
            ]
        )
    if strategy == "ai_boosted":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=220,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=2.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(strategy)


def crossfit_mu(
    compact: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    assignments: pd.DataFrame,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    merged = compact.merge(
        assignments[["patient_id_normalized", "fold"]],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    X = merged[features].apply(pd.to_numeric, errors="coerce")
    t = pd.to_numeric(
        merged["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    folds = merged["fold"].astype(int).to_numpy()
    mu0 = np.full(len(merged), np.nan)
    mu1 = np.full(len(merged), np.nan)

    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        for arm in (0, 1):
            arm_train = train & (t == arm)
            if arm_train.sum() < 20:
                raise ValueError(
                    f"Too few arm={arm} training patients for fold={fold}"
                )
            model = outcome_model(strategy, 2600 + int(fold) + 10 * arm)
            model.fit(X.loc[arm_train], y[arm_train])
            pred = model.predict(X.loc[test])
            if arm == 0:
                mu0[test] = pred
            else:
                mu1[test] = pred

    return mu0, mu1


def aipw_ato(
    y: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> tuple[float, np.ndarray]:
    e = np.clip(e, 0.02, 0.98)
    h = e * (1.0 - e)
    numerator_score = (
        h * (mu1 - mu0)
        + h * a / e * (y - mu1)
        - h * (1 - a) / (1 - e) * (y - mu0)
    )
    theta = float(np.sum(numerator_score) / np.sum(h))
    influence = (numerator_score - theta * h) / np.mean(h)
    return theta, influence


def bootstrap_fixed_scores(
    numerator_score: np.ndarray,
    h: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(h)
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        estimates.append(
            float(
                np.sum(numerator_score[idx]) / np.sum(h[idx])
            )
        )
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    pseudo_dir = DERIVED_DIR / "ipcw_pseudooutcomes"
    weight_dir = DERIVED_DIR / "verified_weights"

    print("=" * 110)
    print("STAGE 26 — PAPER A AI-ASSISTED AIPW-RMST FEASIBILITY")
    print("=" * 110)
    print(
        "Pilot estimator: ATO AIPW applied to cross-fitted IPCW RMST "
        "pseudo-outcomes. CIs are influence-function and fixed-nuisance "
        "bootstrap diagnostics, not yet the final confirmatory inference."
    )

    cohorts = [
        "outer_hormone_hrpos_her2neg",
        "outer_chemo_tnbc",
    ]
    rows = []

    for cohort in cohorts:
        compact, features = load_compact_with_year(cohort)
        assignments = get_repeat_assignments(cohort, repeat=1)
        weight_path = (
            weight_dir / f"24_compact_era_weights_{cohort}.csv"
        )
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        weights = read_table(weight_path)
        base = compact.merge(
            weights[
                [
                    "patient_id_normalized",
                    "propensity_score_oof_compact_era",
                ]
            ],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        a = pd.to_numeric(
            base["analysis_treatment"], errors="raise"
        ).astype(int).to_numpy()
        e = pd.to_numeric(
            base["propensity_score_oof_compact_era"],
            errors="raise",
        ).to_numpy(float)

        print("\n" + "=" * 110)
        print(f"COHORT: {cohort}")
        print(
            f"n={len(base)}, treated={int(a.sum())}, "
            f"controls={int((a == 0).sum())}, "
            f"events={int(pd.to_numeric(base['analysis_event']).sum())}"
        )

        for strategy in ("classical", "ai_boosted"):
            pseudo_path = (
                pseudo_dir / f"25_ipcw_rmst_{cohort}_{strategy}.csv"
            )
            if not pseudo_path.exists():
                raise FileNotFoundError(pseudo_path)
            pseudo = read_table(pseudo_path)
            merged = base.merge(
                pseudo.drop(
                    columns=[
                        c
                        for c in (
                            "analysis_treatment",
                            "analysis_event",
                            "analysis_time",
                        )
                        if c in pseudo.columns
                    ]
                ),
                on="patient_id_normalized",
                how="inner",
                validate="one_to_one",
            )

            for horizon in HORIZONS:
                y = pd.to_numeric(
                    merged[f"rmst_ipcw_{int(horizon)}d"],
                    errors="raise",
                ).to_numpy(float)
                mu0, mu1 = crossfit_mu(
                    compact, features, y, assignments, strategy
                )
                theta, influence = aipw_ato(y, a, e, mu0, mu1)
                se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
                ci_low = theta - 1.96 * se
                ci_high = theta + 1.96 * se

                h = e * (1.0 - e)
                numerator_score = (
                    h * (mu1 - mu0)
                    + h * a / np.clip(e, 0.02, 0.98) * (y - mu1)
                    - h
                    * (1 - a)
                    / np.clip(1 - e, 0.02, 0.98)
                    * (y - mu0)
                )
                boot_low, boot_high = bootstrap_fixed_scores(
                    numerator_score,
                    h,
                    seed=26000 + int(horizon) + (
                        1 if strategy == "ai_boosted" else 0
                    ),
                )

                observed_mu = np.where(a == 1, mu1, mu0)
                mse = float(np.mean((y - observed_mu) ** 2))
                row = {
                    "cohort": cohort,
                    "strategy": strategy,
                    "horizon_days": horizon,
                    "horizon_years": horizon / 365.25,
                    "n": len(y),
                    "treated": int(a.sum()),
                    "control": int((a == 0).sum()),
                    "aipw_ato_rmst_difference_days": theta,
                    "influence_se_days": se,
                    "influence_ci_low_days": ci_low,
                    "influence_ci_high_days": ci_high,
                    "fixed_nuisance_boot_ci_low_days": boot_low,
                    "fixed_nuisance_boot_ci_high_days": boot_high,
                    "outcome_nuisance_oof_mse": mse,
                    "mean_mu0_days": float(np.mean(mu0)),
                    "mean_mu1_days": float(np.mean(mu1)),
                    "pseudo_mean_days": float(np.mean(y)),
                    "pseudo_sd_days": float(np.std(y, ddof=1)),
                    "pseudo_p99_days": float(np.quantile(y, 0.99)),
                    "pseudo_max_days": float(np.max(y)),
                }
                rows.append(row)
                print("\n" + "-" * 90)
                print(
                    f"Strategy={strategy}; horizon={horizon/365.25:.2f} years"
                )
                print(pd.DataFrame([row]).to_string(index=False))

    results = pd.DataFrame(rows)
    results.to_csv(
        table_dir / "26_paperA_ai_aipw_feasibility.csv",
        index=False,
    )

    decision_rows = []
    for cohort, group in results.groupby("cohort"):
        g3 = group[group["horizon_days"] == 1095.0]
        directions = np.sign(
            g3["aipw_ato_rmst_difference_days"].to_numpy()
        )
        stable_direction = int(
            len(directions) >= 2 and np.all(directions == directions[0])
        )
        spread = float(
            g3["aipw_ato_rmst_difference_days"].max()
            - g3["aipw_ato_rmst_difference_days"].min()
        )
        widest_halfwidth = float(
            (
                g3["influence_ci_high_days"]
                - g3["influence_ci_low_days"]
            ).max()
            / 2
        )
        if (
            stable_direction
            and spread <= 90
            and widest_halfwidth <= 180
        ):
            status = "PAPER_A_FEASIBLE_AT_3Y"
        elif stable_direction:
            status = "PAPER_A_DIRECTION_STABLE_BUT_IMPRECISE"
        else:
            status = "PAPER_A_UNSTABLE"
        decision_rows.append(
            {
                "cohort": cohort,
                "three_year_direction_stable": stable_direction,
                "three_year_strategy_spread_days": spread,
                "three_year_widest_ci_halfwidth_days": widest_halfwidth,
                "paperA_feasibility_status": status,
            }
        )
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(
        table_dir / "26_paperA_feasibility_gate.csv", index=False
    )

    print("\n" + "=" * 110)
    print("FINAL PAPER A FEASIBILITY GATE")
    print("=" * 110)
    print(decisions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
