from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs
from _stage10_utils import (
    PRIMARY_COHORT,
    PRIMARY_LANDMARK,
    build_primary_model_table,
    primary_split,
)


SIM_REPS = int(os.environ.get("STAGE10_SIM_REPS", "60"))
PCA_COMPONENTS = 5
SCENARIOS = {
    "null": 0.0,
    "rna_50d_sd": 50.0,
    "rna_100d_sd": 100.0,
    "rna_150d_sd": 150.0,
    "rna_200d_sd": 200.0,
}


def make_propensity(seed: int):
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


def ridge_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ]
    )


def boosted_model(seed: int):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=120,
                    max_leaf_nodes=15,
                    min_samples_leaf=25,
                    l2_regularization=3.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def prepare_fold_features(
    W_train: pd.DataFrame,
    W_test: pd.DataFrame,
    RNA_train: pd.DataFrame,
    RNA_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W_imp = SimpleImputer(strategy="median")
    W_scale = StandardScaler()
    W_train_z = W_scale.fit_transform(
        W_imp.fit_transform(W_train)
    )
    W_test_z = W_scale.transform(W_imp.transform(W_test))

    R_imp = SimpleImputer(strategy="median")
    R_scale = StandardScaler()
    R_train_z = R_scale.fit_transform(
        R_imp.fit_transform(RNA_train)
    )
    R_test_z = R_scale.transform(R_imp.transform(RNA_test))

    components = min(
        PCA_COMPONENTS,
        R_train_z.shape[1],
        max(1, R_train_z.shape[0] - 1),
    )
    pca = PCA(n_components=components, random_state=0)
    R_train_pca = pca.fit_transform(R_train_z)
    R_test_pca = pca.transform(R_test_z)

    X_clin_train = W_train_z
    X_clin_test = W_test_z
    X_rna_train = np.column_stack(
        [W_train_z, R_train_pca]
    )
    X_rna_test = np.column_stack(
        [W_test_z, R_test_pca]
    )
    return (
        X_clin_train,
        X_clin_test,
        X_rna_train,
        X_rna_test,
    )


def fit_tau_model(
    X: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    learner: str,
    seed: int,
):
    if learner == "ridge":
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X, z, sample_weight=weights)
        return model
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=seed,
    )
    model.fit(X, z, sample_weight=weights)
    return model


def oof_comparison(
    W: pd.DataFrame,
    RNA: pd.DataFrame,
    y: np.ndarray,
    a: np.ndarray,
    fold: np.ndarray,
    tau_true: np.ndarray,
    learner: str,
    seed: int,
) -> dict[str, object]:
    n = len(y)
    tau_clin = np.full(n, np.nan)
    tau_rna = np.full(n, np.nan)
    residual_y_oof = np.full(n, np.nan)
    residual_a_oof = np.full(n, np.nan)

    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f

        e_model = make_propensity(seed + int(f))
        e_model.fit(W.loc[train], a[train])
        e_train = e_model.predict_proba(W.loc[train])[:, 1]
        e_test = e_model.predict_proba(W.loc[test])[:, 1]

        m_model = ridge_model()
        m_model.fit(W.loc[train], y[train])
        m_train = m_model.predict(W.loc[train])
        m_test = m_model.predict(W.loc[test])

        residual_y_train = y[train] - m_train
        residual_a_train = a[train] - e_train
        denominator = np.where(
            np.abs(residual_a_train) < 0.05,
            np.where(
                residual_a_train >= 0, 0.05, -0.05
            ),
            residual_a_train,
        )
        z_train = residual_y_train / denominator
        sample_weight = residual_a_train**2

        (
            Xc_train,
            Xc_test,
            Xr_train,
            Xr_test,
        ) = prepare_fold_features(
            W.loc[train],
            W.loc[test],
            RNA.loc[train],
            RNA.loc[test],
        )

        clin_model = fit_tau_model(
            Xc_train,
            z_train,
            sample_weight,
            learner,
            seed + 100 + int(f),
        )
        rna_model = fit_tau_model(
            Xr_train,
            z_train,
            sample_weight,
            learner,
            seed + 200 + int(f),
        )
        tau_clin[test] = clin_model.predict(Xc_test)
        tau_rna[test] = rna_model.predict(Xr_test)
        residual_y_oof[test] = y[test] - m_test
        residual_a_oof[test] = a[test] - e_test

    loss_clin_i = (
        residual_y_oof - residual_a_oof * tau_clin
    ) ** 2
    loss_rna_i = (
        residual_y_oof - residual_a_oof * tau_rna
    ) ** 2
    improvement_i = loss_clin_i - loss_rna_i

    p_value = float(
        ttest_1samp(
            improvement_i,
            popmean=0.0,
            alternative="greater",
        ).pvalue
    )
    pehe_clin = float(
        np.mean((tau_clin - tau_true) ** 2)
    )
    pehe_rna = float(
        np.mean((tau_rna - tau_true) ** 2)
    )
    return {
        "mean_rloss_improvement": float(
            np.mean(improvement_i)
        ),
        "median_rloss_improvement": float(
            np.median(improvement_i)
        ),
        "positive_patient_fraction": float(
            np.mean(improvement_i > 0)
        ),
        "paired_patient_p_value": p_value,
        "detected": int(
            np.isfinite(p_value)
            and p_value < 0.05
            and np.mean(improvement_i) > 0
        ),
        "pehe_clinical": pehe_clin,
        "pehe_clinical_plus_rna": pehe_rna,
        "pehe_improvement": pehe_clin - pehe_rna,
        "tau_clinical_sd": float(
            np.std(tau_clin, ddof=1)
        ),
        "tau_rna_sd": float(np.std(tau_rna, ddof=1)),
    }


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    checkpoint_path = (
        table_dir / "35_paperB_repaired_power_CHECKPOINT.csv"
    )

    base, W_cols, RNA_cols, metadata = (
        build_primary_model_table()
    )
    split = primary_split(base, repeat=1)
    fold = split["fold"].astype(int).to_numpy()

    W = base[W_cols].apply(pd.to_numeric, errors="coerce")
    RNA = base[RNA_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    a = pd.to_numeric(
        base["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    y_real = pd.to_numeric(
        base["rmst_ipcw"], errors="raise"
    ).to_numpy(float)

    # Truth-generating RNA score uses a fixed standardized combination.
    RNA_imputed = SimpleImputer(
        strategy="median"
    ).fit_transform(RNA)
    RNA_z = StandardScaler().fit_transform(RNA_imputed)
    truth_score = RNA_z[:, : min(5, RNA_z.shape[1])].mean(
        axis=1
    )
    truth_score = (
        truth_score - truth_score.mean()
    ) / max(truth_score.std(), 1e-8)

    baseline_model = ridge_model()
    baseline_model.fit(W, y_real)
    baseline = baseline_model.predict(W)
    residual_sd = float(
        np.std(y_real - baseline, ddof=1)
    )

    propensity = make_propensity(3500)
    propensity.fit(W, a)
    e = np.clip(
        propensity.predict_proba(W)[:, 1],
        0.05,
        0.95,
    )

    print("=" * 115)
    print("STAGE 35 — REPAIRED PAPER B POWER DIAGNOSTIC")
    print("=" * 115)
    print(pd.DataFrame([metadata]).to_string(index=False))
    print(
        f"Residual SD={residual_sd:.1f} days; "
        f"simulations per scenario={SIM_REPS}; "
        f"RNA PCA components={PCA_COMPONENTS}"
    )

    # Real-data repaired pilot.
    real_rows = []
    tau_unknown = np.full(len(base), np.nan)
    for learner in ("ridge", "ai_boosted"):
        result = oof_comparison(
            W,
            RNA,
            y_real,
            a,
            fold,
            np.zeros(len(base)),
            learner,
            seed=3510,
        )
        real_rows.append(
            {
                "learner": learner,
                "mean_rloss_improvement": result[
                    "mean_rloss_improvement"
                ],
                "median_rloss_improvement": result[
                    "median_rloss_improvement"
                ],
                "positive_patient_fraction": result[
                    "positive_patient_fraction"
                ],
                "paired_patient_p_value": result[
                    "paired_patient_p_value"
                ],
                "prescriptive_signal_detected": result[
                    "detected"
                ],
                "note": (
                    "PEHE fields are not interpreted for real data"
                ),
            }
        )
    real_df = pd.DataFrame(real_rows)
    real_df.to_csv(
        table_dir / "35_paperB_repaired_observed_pilot.csv",
        index=False,
    )
    print("\nRepaired observed clinical versus clinical+RNA")
    print(real_df.to_string(index=False))

    completed_rows = []
    if checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        if not prior.empty:
            completed_rows = prior.to_dict("records")
            print(
                f"\nResuming from checkpoint with "
                f"{len(completed_rows)} completed result rows."
            )

    completed_keys = {
        (
            row["scenario"],
            int(row["simulation"]),
            row["learner"],
        )
        for row in completed_rows
    }

    for scenario, signal_sd in SCENARIOS.items():
        for sim in range(SIM_REPS):
            rng = np.random.default_rng(
                350000
                + sim
                + int(signal_sd) * 100
            )
            tau_true = (
                np.full(len(base), 50.0)
                if scenario == "null"
                else 50.0 + signal_sd * truth_score
            )
            noise = rng.normal(
                0.0, residual_sd, len(base)
            )
            y = baseline + (a - e) * tau_true + noise

            for learner in ("ridge", "ai_boosted"):
                key = (scenario, sim, learner)
                if key in completed_keys:
                    continue
                result = oof_comparison(
                    W,
                    RNA,
                    y,
                    a,
                    fold,
                    tau_true,
                    learner,
                    seed=35200 + sim,
                )
                completed_rows.append(
                    {
                        "scenario": scenario,
                        "signal_sd_days": signal_sd,
                        "signal_to_noise": signal_sd
                        / max(residual_sd, 1e-8),
                        "simulation": sim,
                        "learner": learner,
                        **result,
                    }
                )
                completed_keys.add(key)

            pd.DataFrame(completed_rows).to_csv(
                checkpoint_path, index=False
            )

        scenario_rows = pd.DataFrame(completed_rows)
        scenario_rows = scenario_rows[
            scenario_rows["scenario"] == scenario
        ]
        print(
            f"\nScenario '{scenario}' completed: "
            f"{len(scenario_rows)} learner-simulation rows."
        )
        print(
            scenario_rows.groupby("learner")
            .agg(
                detection_rate=("detected", "mean"),
                mean_rloss_improvement=(
                    "mean_rloss_improvement",
                    "mean",
                ),
                mean_pehe_improvement=(
                    "pehe_improvement",
                    "mean",
                ),
            )
            .reset_index()
            .to_string(index=False)
        )

    simulation_df = pd.DataFrame(completed_rows)
    simulation_df.to_csv(
        table_dir / "35_paperB_repaired_simulations.csv",
        index=False,
    )

    power = (
        simulation_df.groupby(
            ["scenario", "signal_sd_days", "learner"]
        )
        .agg(
            simulations=("simulation", "nunique"),
            detection_rate=("detected", "mean"),
            mean_rloss_improvement=(
                "mean_rloss_improvement",
                "mean",
            ),
            median_rloss_improvement=(
                "mean_rloss_improvement",
                "median",
            ),
            mean_pehe_improvement=(
                "pehe_improvement",
                "mean",
            ),
            positive_pehe_fraction=(
                "pehe_improvement",
                lambda s: float(np.mean(s > 0)),
            ),
        )
        .reset_index()
    )
    power.to_csv(
        table_dir / "35_paperB_repaired_power_summary.csv",
        index=False,
    )

    gate_rows = []
    for learner, group in power.groupby("learner"):
        null = group[group["scenario"] == "null"].iloc[0]
        detectable = group[
            (group["scenario"] != "null")
            & (group["detection_rate"] >= 0.60)
        ].sort_values("signal_sd_days")
        min_detectable = (
            float(detectable.iloc[0]["signal_sd_days"])
            if not detectable.empty
            else np.nan
        )
        if (
            null["detection_rate"] <= 0.10
            and np.isfinite(min_detectable)
            and min_detectable <= 100
        ):
            status = "ADEQUATE_FOR_100D_SIGNAL"
        elif (
            null["detection_rate"] <= 0.10
            and np.isfinite(min_detectable)
        ):
            status = "ONLY_LARGE_SIGNALS_DETECTABLE"
        else:
            status = "METHOD_NOT_READY"
        gate_rows.append(
            {
                "learner": learner,
                "null_detection_rate": float(
                    null["detection_rate"]
                ),
                "minimum_signal_sd_for_60pct_power": (
                    min_detectable
                ),
                "power_status": status,
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(
        table_dir / "35_paperB_repaired_power_gate.csv",
        index=False,
    )

    print("\nFinal repaired power summary")
    print(power.to_string(index=False))
    print("\nFinal repaired power gate")
    print(gate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
