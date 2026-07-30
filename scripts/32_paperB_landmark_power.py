from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _stage9_utils import make_propensity, ridge_regression


SIM_REPS = int(
    os.environ.get(
        "STAGE9_SIM_REPS",
        "10" if os.environ.get("STAGE9_SMOKE") == "1" else "60",
    )
)
SCENARIOS = {
    "null": 0.0,
    "weak_RNA": 25.0,
    "moderate_RNA": 50.0,
    "strong_RNA": 100.0,
}


def boosted(seed: int):
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


def rlearner_fold_improvements(
    W: pd.DataFrame,
    X_rna: pd.DataFrame,
    y: np.ndarray,
    a: np.ndarray,
    fold: np.ndarray,
    learner: str,
    seed: int,
) -> list[float]:
    improvements = []
    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f

        e_model = make_propensity(seed + int(f))
        e_model.fit(W.loc[train], a[train])
        e_train = e_model.predict_proba(W.loc[train])[:, 1]
        e_test = e_model.predict_proba(W.loc[test])[:, 1]

        m_model = ridge_regression()
        m_model.fit(W.loc[train], y[train])
        m_train = m_model.predict(W.loc[train])
        m_test = m_model.predict(W.loc[test])

        r_y_train = y[train] - m_train
        r_a_train = a[train] - e_train
        z = r_y_train / np.where(
            np.abs(r_a_train) < 0.05,
            np.sign(r_a_train) * 0.05,
            r_a_train,
        )
        sw = r_a_train**2

        model_clin = ridge_regression() if learner == "ridge" else boosted(seed + 100 + int(f))
        model_rna = ridge_regression() if learner == "ridge" else boosted(seed + 200 + int(f))
        model_clin.fit(W.loc[train], z, model__sample_weight=sw)
        model_rna.fit(X_rna.loc[train], z, model__sample_weight=sw)

        tau_clin = model_clin.predict(W.loc[test])
        tau_rna = model_rna.predict(X_rna.loc[test])
        r_y_test = y[test] - m_test
        r_a_test = a[test] - e_test

        loss_clin = np.mean(
            (r_y_test - r_a_test * tau_clin) ** 2
        )
        loss_rna = np.mean(
            (r_y_test - r_a_test * tau_rna) ** 2
        )
        improvements.append(float(loss_clin - loss_rna))
    return improvements


def choose_design() -> tuple[str, int, int]:
    gate = read_table(
        RESULTS_DIR / "tables" / "31_landmark_paperA_gate.csv"
    )
    balance = read_table(
        RESULTS_DIR / "tables" / "30_landmark_balance_summary.csv"
    )
    candidates = gate[
        gate["cohort"] == "outer_hormone_hrpos_her2neg"
    ].merge(
        balance[
            balance["cohort"] == "outer_hormone_hrpos_her2neg"
        ][
            [
                "landmark_day",
                "balance_status",
                "max_abs_smd_overlap",
            ]
        ],
        on="landmark_day",
        how="left",
    )
    priority = {
        "LANDMARK_PAPER_A_FEASIBLE": 0,
        "DIRECTION_STABLE_BUT_SENSITIVE": 1,
        "UNSTABLE": 2,
    }
    candidates["priority"] = candidates["feasibility_status"].map(priority)
    candidates = candidates.sort_values(
        [
            "priority",
            "horizon_days_post_landmark",
            "max_abs_smd_overlap",
        ],
        ascending=[True, True, True],
    )
    row = candidates.iloc[0]
    return (
        "outer_hormone_hrpos_her2neg",
        int(row["landmark_day"]),
        int(row["horizon_days_post_landmark"]),
    )


def resolve_pseudooutcome_path(
    key: str,
    horizon: int,
    preferred_g_min: float = 0.10,
) -> tuple[Path, float]:
    """
    Resolve the actual Stage 31 pseudo-outcome file.

    Earlier Stage 31 versions encoded G=0.10 as `g01`, while one Stage 32
    version expected `g010`. This resolver reads the files that truly exist
    instead of hard-coding one naming convention.
    """
    pseudo_dir = DERIVED_DIR / "landmark_pseudooutcomes"
    prefix = f"{key}_{horizon}d_classical_g"
    candidates = sorted(pseudo_dir.glob(f"{prefix}*.csv"))

    if not candidates:
        available = sorted(p.name for p in pseudo_dir.glob(f"{key}_*.csv"))
        raise FileNotFoundError(
            "No classical Stage 31 pseudo-outcome file was found for "
            f"{key}, horizon={horizon}. Available matching files: {available}"
        )

    def parse_g(path: Path) -> float | None:
        token = path.stem.rsplit("_g", 1)[-1]
        known = {
            "005": 0.05,
            "05": 0.05,
            "050": 0.05,
            "01": 0.10,
            "010": 0.10,
            "10": 0.10,
            "100": 0.10,
        }
        if token in known:
            return known[token]
        try:
            value = float(token)
            if value > 1:
                value /= 100.0
            return value
        except ValueError:
            return None

    parsed = [(path, parse_g(path)) for path in candidates]
    usable = [(path, g) for path, g in parsed if g is not None]
    if not usable:
        raise ValueError(
            "Pseudo-outcome files were found, but their G-min suffixes "
            f"could not be parsed: {[p.name for p in candidates]}"
        )

    chosen_path, chosen_g = min(
        usable,
        key=lambda item: abs(float(item[1]) - preferred_g_min),
    )
    return chosen_path, float(chosen_g)


def build_stage32_model_table(
    cohort_df: pd.DataFrame,
    compact_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build the exact Stage 32 modeling table without merge suffix collisions.

    The landmark cohort can already contain diagnosis_year, while the compact
    adjustment table contains the authoritative compact/era representation.
    We therefore remove compact feature names from the cohort side before the
    one-to-one merge and take all adjustment features from compact_df.
    """
    id_col = "patient_id_normalized"

    compact_feature_cols = [
        c for c in compact_df.columns if c.startswith("W_")
    ]
    for col in ("diagnosis_year", "diagnosis_year_missing"):
        if col in compact_df.columns:
            compact_feature_cols.append(col)

    compact_feature_cols = list(dict.fromkeys(compact_feature_cols))
    if not compact_feature_cols:
        raise ValueError(
            "No compact clinical adjustment features were found."
        )

    required_compact = [id_col] + compact_feature_cols
    compact_selected = compact_df[required_compact].copy()

    # Remove the same feature names from the cohort side to prevent
    # diagnosis_year_x / diagnosis_year_y and any future collisions.
    cohort_selected = cohort_df.drop(
        columns=[
            c for c in compact_feature_cols if c in cohort_df.columns
        ],
        errors="ignore",
    ).copy()

    if cohort_selected.columns.duplicated().any():
        duplicates = cohort_selected.columns[
            cohort_selected.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicate columns in cohort input before merge: {duplicates}"
        )
    if compact_selected.columns.duplicated().any():
        duplicates = compact_selected.columns[
            compact_selected.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicate columns in compact input before merge: {duplicates}"
        )

    base = (
        cohort_selected.merge(
            compact_selected,
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pseudo_df[[id_col, "rmst_ipcw"]],
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
    )

    if base.columns.duplicated().any():
        duplicates = base.columns[base.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate columns in assembled Stage 32 table: {duplicates}"
        )

    missing_adjustment = [
        c for c in compact_feature_cols if c not in base.columns
    ]
    if missing_adjustment:
        raise ValueError(
            "Adjustment columns disappeared during assembly: "
            f"{missing_adjustment}"
        )

    suffix_artifacts = [
        c
        for c in base.columns
        if c.endswith("_x") or c.endswith("_y")
    ]
    if suffix_artifacts:
        raise ValueError(
            "Unexpected merge-suffix columns remain in Stage 32 table: "
            f"{suffix_artifacts}"
        )

    rna_cols = [
        c
        for c in base.columns
        if c.startswith("RNA_")
        and not any(
            token in c.lower()
            for token in ("missing", "available", "indicator")
        )
    ]
    if not rna_cols:
        raise ValueError(
            "No biological RNA columns were found after model-table assembly."
        )

    required_analysis = {
        "analysis_treatment",
        "analysis_event",
        "analysis_time",
        "rmst_ipcw",
    }
    missing_analysis = required_analysis - set(base.columns)
    if missing_analysis:
        raise ValueError(
            "Assembled Stage 32 table is missing analysis columns: "
            f"{sorted(missing_analysis)}"
        )

    return base, compact_feature_cols, rna_cols


def validate_stage32_inputs(
    cohort: str,
    landmark: int,
    horizon: int,
) -> dict[str, object]:
    key = f"{cohort}_landmark{landmark}"
    paths = {
        "cohort": DERIVED_DIR / "landmark_cohorts" / f"{key}.csv",
        "compact": DERIVED_DIR / "landmark_compact" / f"{key}_compact.csv",
        "splits": DERIVED_DIR / "landmark_splits" / f"{key}_splits.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 32 preflight failed. Missing files:\n" + "\n".join(missing)
        )

    pseudo_path, selected_g_min = resolve_pseudooutcome_path(
        key, horizon, preferred_g_min=0.10
    )
    paths["pseudo"] = pseudo_path

    frames = {name: read_table(path) for name, path in paths.items()}
    required_columns = {
        "cohort": {
            "patient_id_normalized",
            "analysis_treatment",
            "analysis_event",
            "analysis_time",
        },
        "compact": {"patient_id_normalized"},
        "splits": {"patient_id_normalized", "repeat", "fold"},
        "pseudo": {"patient_id_normalized", "rmst_ipcw"},
    }
    for name, required in required_columns.items():
        missing_cols = required - set(frames[name].columns)
        if missing_cols:
            raise ValueError(
                f"{name} input is missing columns: {sorted(missing_cols)}"
            )

    id_sets = {
        name: set(frame["patient_id_normalized"].astype(str))
        for name, frame in frames.items()
    }
    reference = id_sets["cohort"]
    mismatch = {
        name: len(reference.symmetric_difference(ids))
        for name, ids in id_sets.items()
        if name != "cohort"
    }
    if any(value > 0 for value in mismatch.values()):
        raise ValueError(
            "Stage 32 patient-ID mismatch across inputs: "
            f"{mismatch}"
        )

    base, W_cols, rna_cols = build_stage32_model_table(
        frames["cohort"],
        frames["compact"],
        frames["pseudo"],
    )

    if len(base) != len(frames["cohort"]):
        raise ValueError(
            "Stage 32 assembled row count differs from cohort row count: "
            f"{len(base)} versus {len(frames['cohort'])}"
        )

    # Dry-run the exact column selections used in the modeling code.
    _ = base[W_cols].apply(pd.to_numeric, errors="coerce")
    _ = base[W_cols + rna_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    _ = pd.to_numeric(
        base["analysis_treatment"], errors="raise"
    ).astype(int)
    _ = pd.to_numeric(base["rmst_ipcw"], errors="raise")

    return {
        "cohort": cohort,
        "landmark_day": landmark,
        "horizon_days_post_landmark": horizon,
        "key": key,
        "pseudo_path": str(pseudo_path),
        "selected_g_min": selected_g_min,
        "n": len(base),
        "split_repeats": int(frames["splits"]["repeat"].nunique()),
        "split_folds": int(frames["splits"]["fold"].nunique()),
        "compact_features": len(W_cols),
        "rna_features": len(rna_cols),
        "diagnosis_year_source": (
            "compact_table"
            if "diagnosis_year" in W_cols
            else "not_available"
        ),
        "model_table_suffix_artifacts": 0,
    }


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    cohort, landmark, horizon = choose_design()
    preflight = validate_stage32_inputs(cohort, landmark, horizon)
    key = str(preflight["key"])
    pseudo_path = Path(str(preflight["pseudo_path"]))

    print("=" * 115)
    print("STAGE 32 PREFLIGHT PASSED")
    print("=" * 115)
    print(pd.DataFrame([preflight]).to_string(index=False))

    df = read_table(
        DERIVED_DIR / "landmark_cohorts" / f"{key}.csv"
    )
    compact = read_table(
        DERIVED_DIR / "landmark_compact" / f"{key}_compact.csv"
    )
    splits = read_table(
        DERIVED_DIR / "landmark_splits" / f"{key}_splits.csv"
    )
    pseudo = read_table(pseudo_path)

    base, W_cols, rna_cols = build_stage32_model_table(
        df, compact, pseudo
    )

    print("\nCanonical Stage 32 model table")
    print(
        pd.DataFrame(
            [
                {
                    "rows": len(base),
                    "columns": len(base.columns),
                    "compact_features": len(W_cols),
                    "rna_features": len(rna_cols),
                    "diagnosis_year_present": int(
                        "diagnosis_year" in W_cols
                    ),
                    "merge_suffix_columns": len(
                        [
                            c
                            for c in base.columns
                            if c.endswith("_x") or c.endswith("_y")
                        ]
                    ),
                }
            ]
        ).to_string(index=False)
    )

    W = base[W_cols].apply(pd.to_numeric, errors="coerce")
    X_rna = base[W_cols + rna_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    a = base["analysis_treatment"].astype(int).to_numpy()
    y_observed = base["rmst_ipcw"].to_numpy(float)

    assignment = splits[splits["repeat"] == 1][
        ["patient_id_normalized", "fold"]
    ]
    fold = base[["patient_id_normalized"]].merge(
        assignment,
        on="patient_id_normalized",
        how="left",
        validate="one_to_one",
    )["fold"].astype(int).to_numpy()

    print("=" * 115)
    print("STAGE 32 — PAPER B LANDMARK PILOT AND POWER GRID")
    print("=" * 115)
    print(
        f"Selected design: cohort={cohort}, landmark={landmark}, "
        f"horizon={horizon} days post-landmark"
    )
    print(
        f"n={len(base)}, treated={int(a.sum())}, controls={int((a==0).sum())}, "
        f"RNA features={len(rna_cols)}, simulations={SIM_REPS}"
    )

    observed_rows = []
    for learner in ("ridge", "ai_boosted"):
        improvements = rlearner_fold_improvements(
            W, X_rna, y_observed, a, fold, learner, seed=3200
        )
        row = {
            "learner": learner,
            "mean_fold_rloss_improvement": float(np.mean(improvements)),
            "median_fold_rloss_improvement": float(np.median(improvements)),
            "positive_folds": int(np.sum(np.asarray(improvements) > 0)),
            "folds": len(improvements),
            "relative_improvement": float(
                np.mean(improvements) / np.var(y_observed)
            ),
        }
        observed_rows.append(row)
    observed = pd.DataFrame(observed_rows)
    observed.to_csv(
        table_dir / "32_paperB_landmark_observed_pilot.csv",
        index=False,
    )
    print("\nObserved landmark clinical versus clinical+RNA")
    print(observed.to_string(index=False))

    # Simulation-generating components.
    rng0 = np.random.default_rng(2026)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Wz = scaler.fit_transform(imputer.fit_transform(W))
    Rz = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(
            base[rna_cols]
        )
    )
    z_rna = Rz[:, : min(5, Rz.shape[1])].mean(axis=1)
    z_rna = (z_rna - z_rna.mean()) / max(z_rna.std(), 1e-8)
    z_clin = Wz[:, : min(3, Wz.shape[1])].mean(axis=1)
    z_clin = (z_clin - z_clin.mean()) / max(z_clin.std(), 1e-8)

    m_model = ridge_regression()
    m_model.fit(W, y_observed)
    baseline = m_model.predict(W)
    residual_sd = float(
        np.std(y_observed - baseline, ddof=1)
    )

    e_model = make_propensity(3210)
    e_model.fit(W, a)
    e = np.clip(e_model.predict_proba(W)[:, 1], 0.05, 0.95)

    sim_rows = []
    for scenario, delta in SCENARIOS.items():
        detections = {"ridge": 0, "ai_boosted": 0}
        mean_improvements = {"ridge": [], "ai_boosted": []}
        for sim in range(SIM_REPS):
            rng = np.random.default_rng(50000 + sim + int(delta))
            if scenario == "null":
                tau = np.full(len(base), 50.0)
            elif scenario == "weak_RNA":
                tau = 50.0 + delta * z_rna
            elif scenario == "moderate_RNA":
                tau = 50.0 + delta * z_rna
            elif scenario == "strong_RNA":
                tau = 50.0 + delta * z_rna
            else:
                tau = 50.0 + delta * z_clin
            noise = rng.normal(0.0, residual_sd, len(base))
            y = baseline + tau * (a - e) + noise

            for learner in ("ridge", "ai_boosted"):
                improvements = rlearner_fold_improvements(
                    W,
                    X_rna,
                    y,
                    a,
                    fold,
                    learner,
                    seed=33000 + sim,
                )
                mean_imp = float(np.mean(improvements))
                mean_improvements[learner].append(mean_imp)
                p = ttest_1samp(
                    improvements,
                    popmean=0.0,
                    alternative="greater",
                ).pvalue
                if np.isfinite(p) and p < 0.05:
                    detections[learner] += 1

        for learner in ("ridge", "ai_boosted"):
            vals = np.asarray(mean_improvements[learner])
            sim_rows.append(
                {
                    "scenario": scenario,
                    "rna_effect_sd_days": delta,
                    "learner": learner,
                    "simulations": SIM_REPS,
                    "detection_rate": detections[learner] / SIM_REPS,
                    "mean_rloss_improvement": float(vals.mean()),
                    "median_rloss_improvement": float(np.median(vals)),
                    "positive_mean_fraction": float(np.mean(vals > 0)),
                }
            )
            print(
                f"scenario={scenario}, learner={learner}, "
                f"detection={detections[learner]}/{SIM_REPS}, "
                f"mean improvement={vals.mean():.3f}"
            )

        checkpoint = pd.DataFrame(sim_rows)
        checkpoint.to_csv(
            table_dir / "32_paperB_power_grid_CHECKPOINT.csv",
            index=False,
        )
        print(
            f"Checkpoint saved after scenario '{scenario}': "
            f"{table_dir / '32_paperB_power_grid_CHECKPOINT.csv'}"
        )

    power = pd.DataFrame(sim_rows)
    power.to_csv(
        table_dir / "32_paperB_power_grid.csv", index=False
    )

    gate_rows = []
    for learner, group in power.groupby("learner"):
        null = group[group["scenario"] == "null"].iloc[0]
        moderate = group[group["scenario"] == "moderate_RNA"].iloc[0]
        strong = group[group["scenario"] == "strong_RNA"].iloc[0]
        if (
            null["detection_rate"] <= 0.10
            and moderate["detection_rate"] >= 0.60
        ):
            status = "POWER_ADEQUATE_FOR_MODERATE_SIGNAL"
        elif (
            null["detection_rate"] <= 0.10
            and strong["detection_rate"] >= 0.60
        ):
            status = "POWER_ONLY_FOR_STRONG_SIGNAL"
        else:
            status = "POWER_OR_TYPE1_INADEQUATE"
        gate_rows.append(
            {
                "learner": learner,
                "null_detection_rate": null["detection_rate"],
                "moderate_detection_rate": moderate["detection_rate"],
                "strong_detection_rate": strong["detection_rate"],
                "power_status": status,
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(
        table_dir / "32_paperB_power_gate.csv", index=False
    )

    print("\nFinal Paper B power grid")
    print(power.to_string(index=False))
    print("\nFinal Paper B power gate")
    print(gate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
