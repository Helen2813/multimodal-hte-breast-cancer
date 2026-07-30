from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _stage8_utils import load_compact_with_year


COHORT = "outer_hormone_hrpos_her2neg"
HORIZONS = (1095.0,)
MAX_REPEATS = 1 if os.environ.get('STAGE8_SMOKE') == '1' else 3


def ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ]
    )


def boosted_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=90,
                    max_leaf_nodes=15,
                    min_samples_leaf=25,
                    l2_regularization=3.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def tau_model(kind: str, seed: int):
    return ridge_pipeline() if kind == "linear_ridge" else boosted_pipeline(seed)


def prognostic_model(kind: str, seed: int):
    return ridge_pipeline() if kind == "linear_ridge" else boosted_pipeline(seed)


def fit_tau(
    X_train: pd.DataFrame,
    residual_y: np.ndarray,
    residual_a: np.ndarray,
    kind: str,
    seed: int,
):
    denominator = np.where(
        np.abs(residual_a) < 0.05,
        np.sign(residual_a) * 0.05,
        residual_a,
    )
    z = residual_y / denominator
    sample_weight = residual_a**2
    model = tau_model(kind, seed)
    model.fit(X_train, z, model__sample_weight=sample_weight)
    return model


def crossfit_shared_nuisance(
    W: pd.DataFrame,
    y: np.ndarray,
    a: np.ndarray,
    fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from _stage8_utils import make_propensity_model

    e = np.full(len(y), np.nan)
    m = np.full(len(y), np.nan)
    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f
        e_model = make_propensity_model(2700 + int(f))
        e_model.fit(W.loc[train], a[train])
        e[test] = e_model.predict_proba(W.loc[test])[:, 1]

        m_model = ridge_pipeline()
        m_model.fit(W.loc[train], y[train])
        m[test] = m_model.predict(W.loc[test])
    return np.clip(e, 0.02, 0.98), m


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    cohort_path = (
        DERIVED_DIR / "verified_cohorts" / f"{COHORT}_verified.csv"
    )
    pseudo_path = (
        DERIVED_DIR
        / "ipcw_pseudooutcomes"
        / f"25_ipcw_rmst_{COHORT}_classical.csv"
    )
    split_path = (
        DERIVED_DIR
        / "verified_splits"
        / "23_verified_repeated_fold_assignments.csv"
    )
    if not cohort_path.exists() or not pseudo_path.exists():
        raise FileNotFoundError("Stage 25 outer-hormone inputs are missing.")

    df = read_table(cohort_path)
    pseudo = read_table(pseudo_path)
    compact, W_cols = load_compact_with_year(COHORT)
    splits = read_table(split_path)
    splits = splits[splits["cohort"] == COHORT].copy()

    merged = (
        df.merge(
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
        .merge(
            compact[
                ["patient_id_normalized"] + W_cols
            ],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
    )

    rna_cols = [
        c
        for c in merged.columns
        if c.startswith("RNA_")
        and not any(
            token in c.lower()
            for token in ("missing", "available", "indicator")
        )
    ]
    if not rna_cols:
        raise ValueError("No biological RNA columns were found.")

    print("=" * 110)
    print("STAGE 27 — PAPER B CLINICAL VS CLINICAL+RNA AI FEASIBILITY")
    print("=" * 110)
    print(
        f"Cohort={COHORT}; n={len(merged)}; "
        f"RNA features={len(rna_cols)}; compact clinical features={len(W_cols)}; "
        f"pilot repeats={MAX_REPEATS}; horizon=3 years"
    )
    print(
        "Primary pilot: paired repeated-fold comparisons of prognostic "
        "IPCW-RMST prediction and prescriptive R-loss."
    )

    a = pd.to_numeric(
        merged["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    W = merged[W_cols].apply(pd.to_numeric, errors="coerce")
    X_sets = {
        "clinical": W,
        "clinical_plus_RNA": merged[W_cols + rna_cols].apply(
            pd.to_numeric, errors="coerce"
        ),
    }

    fold_rows = []
    cate_rows = []

    for horizon in HORIZONS:
        y = pd.to_numeric(
            merged[f"rmst_ipcw_{int(horizon)}d"], errors="raise"
        ).to_numpy(float)

        # Shared nuisance estimates are generated once on the first verified
        # split and then held fixed across repeated effect-model comparisons.
        # This preserves the same adjustment nuisance space for clinical and RNA.
        first_repeat = int(sorted(splits["repeat"].unique())[0])
        nuisance_assignment = splits[splits["repeat"] == first_repeat][
            ["patient_id_normalized", "fold"]
        ]
        nuisance_temp = merged[["patient_id_normalized"]].merge(
            nuisance_assignment,
            on="patient_id_normalized",
            how="left",
            validate="one_to_one",
        )
        nuisance_fold = nuisance_temp["fold"].astype(int).to_numpy()
        e_oof, m_oof = crossfit_shared_nuisance(
            W, y, a, nuisance_fold
        )

        repeats_to_run = sorted(splits["repeat"].unique())[:MAX_REPEATS]
        for repeat in repeats_to_run:
            assignment = splits[splits["repeat"] == repeat][
                ["patient_id_normalized", "fold"]
            ]
            temp = merged[["patient_id_normalized"]].merge(
                assignment,
                on="patient_id_normalized",
                how="left",
                validate="one_to_one",
            )
            fold = temp["fold"].astype(int).to_numpy()

            for learner in ("linear_ridge", "ai_boosted"):
                for f in sorted(np.unique(fold)):
                    train = fold != f
                    test = fold == f

                    predictions = {}
                    prog_predictions = {}
                    for feature_set, X in X_sets.items():
                        pmodel = prognostic_model(
                            learner,
                            seed=27000
                            + int(repeat) * 100
                            + int(f),
                        )
                        pmodel.fit(X.loc[train], y[train])
                        prog_predictions[feature_set] = pmodel.predict(
                            X.loc[test]
                        )

                        tmodel = fit_tau(
                            X.loc[train],
                            y[train] - m_oof[train],
                            a[train] - e_oof[train],
                            learner,
                            seed=27500
                            + int(repeat) * 100
                            + int(f),
                        )
                        predictions[feature_set] = tmodel.predict(
                            X.loc[test]
                        )

                    prog_mse_clin = float(
                        np.mean(
                            (
                                y[test]
                                - prog_predictions["clinical"]
                            )
                            ** 2
                        )
                    )
                    prog_mse_rna = float(
                        np.mean(
                            (
                                y[test]
                                - prog_predictions[
                                    "clinical_plus_RNA"
                                ]
                            )
                            ** 2
                        )
                    )

                    residual_y_test = y[test] - m_oof[test]
                    residual_a_test = a[test] - e_oof[test]
                    rloss_clin = float(
                        np.mean(
                            (
                                residual_y_test
                                - residual_a_test
                                * predictions["clinical"]
                            )
                            ** 2
                        )
                    )
                    rloss_rna = float(
                        np.mean(
                            (
                                residual_y_test
                                - residual_a_test
                                * predictions[
                                    "clinical_plus_RNA"
                                ]
                            )
                            ** 2
                        )
                    )

                    fold_rows.append(
                        {
                            "horizon_days": horizon,
                            "repeat": int(repeat),
                            "fold": int(f),
                            "learner": learner,
                            "test_n": int(test.sum()),
                            "prognostic_mse_clinical": prog_mse_clin,
                            "prognostic_mse_clinical_plus_RNA": prog_mse_rna,
                            "prognostic_mse_improvement": (
                                prog_mse_clin - prog_mse_rna
                            ),
                            "prescriptive_rloss_clinical": rloss_clin,
                            "prescriptive_rloss_clinical_plus_RNA": rloss_rna,
                            "prescriptive_rloss_improvement": (
                                rloss_clin - rloss_rna
                            ),
                        }
                    )
                    for feature_set, pred in predictions.items():
                        cate_rows.append(
                            {
                                "horizon_days": horizon,
                                "repeat": int(repeat),
                                "fold": int(f),
                                "learner": learner,
                                "feature_set": feature_set,
                                "cate_mean": float(np.mean(pred)),
                                "cate_sd": float(np.std(pred, ddof=1))
                                if len(pred) > 1
                                else np.nan,
                                "cate_positive_fraction": float(
                                    np.mean(pred > 0)
                                ),
                            }
                        )

    folds_df = pd.DataFrame(fold_rows)
    cate_df = pd.DataFrame(cate_rows)
    folds_df.to_csv(
        table_dir / "27_paperB_fold_results.csv", index=False
    )
    cate_df.to_csv(
        table_dir / "27_paperB_cate_diagnostics.csv", index=False
    )

    repeat_summary = (
        folds_df.groupby(["horizon_days", "repeat", "learner"])
        .agg(
            prognostic_mse_improvement=(
                "prognostic_mse_improvement",
                "mean",
            ),
            prescriptive_rloss_improvement=(
                "prescriptive_rloss_improvement",
                "mean",
            ),
            clinical_prog_mse=(
                "prognostic_mse_clinical",
                "mean",
            ),
            clinical_r_loss=(
                "prescriptive_rloss_clinical",
                "mean",
            ),
        )
        .reset_index()
    )
    repeat_summary[
        "prognostic_relative_improvement"
    ] = (
        repeat_summary["prognostic_mse_improvement"]
        / repeat_summary["clinical_prog_mse"]
    )
    repeat_summary[
        "prescriptive_relative_improvement"
    ] = (
        repeat_summary["prescriptive_rloss_improvement"]
        / repeat_summary["clinical_r_loss"]
    )
    repeat_summary.to_csv(
        table_dir / "27_paperB_repeat_summary.csv", index=False
    )

    aggregate_rows = []
    for (horizon, learner), group in repeat_summary.groupby(
        ["horizon_days", "learner"]
    ):
        prog_positive = int(
            (group["prognostic_mse_improvement"] > 0).sum()
        )
        pres_positive = int(
            (group["prescriptive_rloss_improvement"] > 0).sum()
        )
        mean_prog_rel = float(
            group["prognostic_relative_improvement"].mean()
        )
        mean_pres_rel = float(
            group["prescriptive_relative_improvement"].mean()
        )
        required_positive = max(2, int(np.ceil(0.67 * len(group))))
        if pres_positive >= required_positive and mean_pres_rel >= 0.01:
            status = "PROMISING_PRESCRIPTIVE_SIGNAL"
        elif prog_positive >= required_positive and mean_prog_rel >= 0.01:
            status = "PROGNOSTIC_ONLY_SIGNAL"
        else:
            status = "NO_STABLE_INCREMENTAL_RNA_SIGNAL"
        aggregate_rows.append(
            {
                "horizon_days": horizon,
                "learner": learner,
                "repeats": len(group),
                "prognostic_positive_repeats": prog_positive,
                "prescriptive_positive_repeats": pres_positive,
                "mean_prognostic_relative_improvement": mean_prog_rel,
                "mean_prescriptive_relative_improvement": mean_pres_rel,
                "min_prescriptive_relative_improvement": float(
                    group[
                        "prescriptive_relative_improvement"
                    ].min()
                ),
                "max_prescriptive_relative_improvement": float(
                    group[
                        "prescriptive_relative_improvement"
                    ].max()
                ),
                "paperB_pilot_status": status,
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        table_dir / "27_paperB_ai_feasibility_summary.csv",
        index=False,
    )

    print("\nRepeated-fold paired improvements")
    print(repeat_summary.to_string(index=False))
    print("\nFinal Paper B AI feasibility summary")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
