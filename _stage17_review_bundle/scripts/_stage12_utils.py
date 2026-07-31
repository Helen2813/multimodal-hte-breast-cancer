from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import (
    PROCESSED_DIR,
    DERIVED_DIR,
    RESULTS_DIR,
    normalize_patient_id,
    read_table,
)

COHORT = "outer_hormone_hrpos_her2neg"
LANDMARK_DAY = 180
LANDMARK_HORIZON = 730.0
CCW_HORIZON = LANDMARK_DAY + LANDMARK_HORIZON
G_MIN = 0.10
LANDMARK_INTERVAL = 90.0
CCW_INTERVAL = 30.0
N_FOLDS = 5
BASE_SEED = 12000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_clinical_path() -> Path:
    for path in (
        PROCESSED_DIR / "01_Clinical" / "clinical.tsv",
        PROCESSED_DIR / "01_Clinical" / "drags" / "clinical.tsv",
    ):
        if path.exists():
            return path
    return PROCESSED_DIR / "01_Clinical" / "clinical.tsv"


def project_paths() -> dict[str, Path]:
    key = f"{COHORT}_landmark{LANDMARK_DAY}"
    return {
        "landmark_cohort": DERIVED_DIR / "landmark_cohorts" / f"{key}.csv",
        "landmark_compact": DERIVED_DIR / "landmark_compact" / f"{key}_compact.csv",
        "landmark_splits": DERIVED_DIR / "landmark_splits" / f"{key}_splits.csv",
        "landmark_weights": DERIVED_DIR / "landmark_weights" / f"{key}_weights.csv",
        "source_cohort": DERIVED_DIR / "verified_cohorts" / f"{COHORT}_verified.csv",
        "source_compact": DERIVED_DIR / "verified_compact_adjustment" / f"{COHORT}_compact_verified.csv",
        "diagnosis_year": DERIVED_DIR / "manifests" / "20_patient_diagnosis_year.csv",
        "candidate": RESULTS_DIR / "tables" / "34_paperA_candidate_summary.csv",
        "candidate_v2": RESULTS_DIR / "tables" / "40_stage11_design_decision.csv",
        "clinical": _find_clinical_path(),
    }


def require_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 12 inputs:\n" + "\n".join(missing))


def compact_features(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("W_")]
    for col in ("diagnosis_year", "diagnosis_year_missing"):
        if col in frame.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))


def assemble_landmark_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, object]]:
    paths = project_paths()
    require_paths({name: paths[name] for name in ("landmark_cohort", "landmark_compact", "landmark_splits", "landmark_weights", "candidate")})
    cohort = read_table(paths["landmark_cohort"])
    compact = read_table(paths["landmark_compact"])
    splits = read_table(paths["landmark_splits"])
    weights = read_table(paths["landmark_weights"])
    features = compact_features(compact)
    if not features:
        raise ValueError("No compact landmark features found.")

    cohort_side = cohort.drop(columns=[c for c in features if c in cohort.columns], errors="ignore")
    required_weight_columns = {"patient_id_normalized", "propensity_score_oof"}
    missing_weight_columns = required_weight_columns - set(weights.columns)
    if missing_weight_columns:
        raise ValueError(
            "Stage 30 landmark weights are missing columns: "
            f"{sorted(missing_weight_columns)}"
        )
    base = (
        cohort_side.merge(
            compact[["patient_id_normalized"] + features],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            weights[["patient_id_normalized", "propensity_score_oof"]].rename(
                columns={"propensity_score_oof": "propensity_score_oof_stage30"}
            ),
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
    )
    suffixes = [c for c in base.columns if c.endswith("_x") or c.endswith("_y")]
    if suffixes:
        raise ValueError(f"Unexpected merge suffixes: {suffixes}")
    if len(base) != len(cohort):
        raise ValueError("Landmark assembly lost rows.")

    assignment = splits[splits["repeat"] == 1][["patient_id_normalized", "fold"]]
    assignment = base[["patient_id_normalized"]].merge(assignment, on="patient_id_normalized", how="left", validate="one_to_one")
    if assignment["fold"].isna().any():
        raise ValueError("Missing landmark fold assignments.")

    required = {"analysis_treatment", "analysis_event", "analysis_time"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Landmark analysis columns missing: {sorted(missing)}")

    candidate = read_table(paths["candidate"])
    candidate_row = candidate[
        (candidate["cohort"] == COHORT)
        & (candidate["landmark_day"] == LANDMARK_DAY)
        & (candidate["post_landmark_horizon_days"] == int(LANDMARK_HORIZON))
    ]
    if len(candidate_row) != 1:
        raise ValueError("Expected one candidate Paper A row.")

    metadata = {
        "cohort": COHORT,
        "landmark_day": LANDMARK_DAY,
        "horizon_days": LANDMARK_HORIZON,
        "n": len(base),
        "treated": int(pd.to_numeric(base["analysis_treatment"], errors="raise").sum()),
        "control": int((1 - pd.to_numeric(base["analysis_treatment"], errors="raise")).sum()),
        "events": int(pd.to_numeric(base["analysis_event"], errors="raise").sum()),
        "features": len(features),
        "candidate_effect_days": float(candidate_row.iloc[0]["primary_effect_days"]),
        "cohort_sha256": sha256_file(paths["landmark_cohort"]),
        "compact_sha256": sha256_file(paths["landmark_compact"]),
        "weights_sha256": sha256_file(paths["landmark_weights"]),
    }
    return base, features, assignment, metadata


def extract_hormone_timing(patient_ids: pd.Series) -> pd.DataFrame:
    clinical_path = project_paths()["clinical"]
    if not clinical_path.exists():
        raise FileNotFoundError(clinical_path)
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    required = {"cases.submitter_id", "treatments.treatment_type", "treatments.days_to_treatment_start"}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Clinical timing columns missing: {sorted(missing)}")

    text = clinical["treatments.treatment_type"].astype(str).str.lower()
    mask = text.str.contains("hormone|endocrine", regex=True, na=False)
    temp = pd.DataFrame({
        "patient_id_normalized": clinical.loc[mask, "cases.submitter_id"].map(normalize_patient_id),
        "start_day": pd.to_numeric(clinical.loc[mask, "treatments.days_to_treatment_start"], errors="coerce"),
    })
    timing = temp.groupby("patient_id_normalized", as_index=False).agg(
        treatment_records=("start_day", "size"),
        nonmissing_start_records=("start_day", "count"),
        earliest_start_nonnegative=("start_day", lambda s: s[s >= 0].min() if (s >= 0).any() else np.nan),
        negative_start_records=("start_day", lambda s: int((s < 0).sum())),
    )
    return pd.DataFrame({"patient_id_normalized": patient_ids.astype(str)}).merge(
        timing, on="patient_id_normalized", how="left", validate="one_to_one"
    )


def assemble_source_data() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    paths = project_paths()
    require_paths({name: paths[name] for name in ("source_cohort", "source_compact", "diagnosis_year", "clinical")})
    cohort = read_table(paths["source_cohort"])
    compact = read_table(paths["source_compact"])
    years = read_table(paths["diagnosis_year"])[["patient_id_normalized", "diagnosis_year"]]

    if "diagnosis_year" not in compact.columns:
        compact = compact.merge(years, on="patient_id_normalized", how="left", validate="one_to_one")
    if "diagnosis_year_missing" not in compact.columns:
        compact["diagnosis_year_missing"] = compact["diagnosis_year"].isna().astype(float)
    features = compact_features(compact)
    cohort_side = cohort.drop(columns=[c for c in features if c in cohort.columns], errors="ignore")
    base = cohort_side.merge(compact[["patient_id_normalized"] + features], on="patient_id_normalized", how="inner", validate="one_to_one")
    timing = extract_hormone_timing(base["patient_id_normalized"])
    base = base.merge(timing, on="patient_id_normalized", how="left", validate="one_to_one")

    ever = pd.to_numeric(base["analysis_treatment"], errors="raise").astype(int)
    start = pd.to_numeric(base["earliest_start_nonnegative"], errors="coerce")
    ambiguous = ever.eq(1) & (start.isna() | start.gt(3650))
    base = base.loc[~ambiguous].copy().reset_index(drop=True)
    start = pd.to_numeric(base["earliest_start_nonnegative"], errors="coerce")
    base["early_initiation"] = start.between(0, LANDMARK_DAY).astype(int)

    metadata = {
        "source_n": len(cohort),
        "ccw_eligible_n": len(base),
        "excluded_ambiguous": int(ambiguous.sum()),
        "early_initiators": int(base["early_initiation"].sum()),
        "no_initiation_by_180": int((1 - base["early_initiation"]).sum()),
        "source_sha256": sha256_file(paths["source_cohort"]),
        "source_compact_sha256": sha256_file(paths["source_compact"]),
    }
    return base, features, metadata


def make_propensity_model(seed: int, cv: int = 4) -> Pipeline:
    """Exact Stage 30 propensity specification."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
            cv=cv,
            scoring="neg_log_loss",
            penalty="l2",
            solver="lbfgs",
            max_iter=6000,
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def make_logistic(seed: int, cv: int = 3) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=[0.003, 0.01, 0.03, 0.1, 0.3],
            cv=cv,
            scoring="neg_log_loss",
            max_iter=6000,
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def make_ridge() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
    ])


def make_bootstrap_sample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(frame), len(frame))
    sample = frame.iloc[indices].copy().reset_index(drop=True)
    sample["original_patient_id"] = sample["patient_id_normalized"].astype(str)
    sample["bootstrap_instance_id"] = [f"B{seed}_{i:05d}" for i in range(len(sample))]
    return sample


def bootstrap_folds(
    frame: pd.DataFrame,
    treatment_col: str,
    event_col: str,
    seed: int,
    n_folds: int = N_FOLDS,
) -> np.ndarray:
    treatment = pd.to_numeric(frame[treatment_col], errors="raise").astype(int)
    event = pd.to_numeric(frame[event_col], errors="raise").astype(int)
    strata = 2 * treatment + event
    groups = frame["original_patient_id"].astype(str)

    group_frame = pd.DataFrame({"group": groups, "stratum": strata}).drop_duplicates("group")
    counts = group_frame["stratum"].value_counts()
    folds = min(n_folds, int(counts.min()))
    if folds < 3:
        strata = treatment
        group_frame = pd.DataFrame({"group": groups, "stratum": strata}).drop_duplicates("group")
        counts = group_frame["stratum"].value_counts()
        folds = min(n_folds, int(counts.min()))
    if folds < 3:
        raise ValueError("Fewer than three grouped folds are possible.")

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.full(len(frame), -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(frame)), strata, groups), start=1):
        assignment[test] = fold
    if np.any(assignment < 1):
        raise RuntimeError("Bootstrap fold assignment failed.")
    return assignment


def crossfit_propensity(
    frame: pd.DataFrame,
    features: list[str],
    fold: np.ndarray,
    treatment_col: str,
    seed: int,
) -> np.ndarray:
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    a = pd.to_numeric(frame[treatment_col], errors="raise").astype(int).to_numpy()
    ps = np.full(len(frame), np.nan)
    for f in sorted(np.unique(fold)):
        train = fold != f
        test = fold == f
        inner = max(2, min(4, int(np.bincount(a[train], minlength=2).min())))
        model = make_propensity_model(seed + int(f), cv=inner)
        model.fit(X.loc[train], a[train])
        ps[test] = model.predict_proba(X.loc[test])[:, 1]
    return np.clip(ps, 0.01, 0.99)


def interval_grid(horizon: float, interval_days: float) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0.0, horizon, interval_days)
    ends = np.minimum(starts + interval_days, horizon)
    return starts, ends


def build_censor_long(
    frame: pd.DataFrame,
    features: list[str],
    horizon: float,
    interval_days: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    starts, ends = interval_grid(horizon, interval_days)
    time = pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float)
    event = pd.to_numeric(frame["analysis_event"], errors="raise").astype(int).to_numpy()
    treatment = pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy()
    rows = []
    X = frame[features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    for i in range(len(frame)):
        for k, (start, end) in enumerate(zip(starts, ends)):
            if not np.isfinite(time[i]) or time[i] < start:
                continue
            row = X.iloc[i].to_dict()
            row.update({
                "row_id": i,
                "interval": k,
                "interval_start": start,
                "treatment": treatment[i],
                "censor_event": int(event[i] == 0 and start <= time[i] <= end),
            })
            rows.append(row)
    return pd.DataFrame(rows), starts, ends


def crossfit_censor_survival(
    frame: pd.DataFrame,
    features: list[str],
    fold: np.ndarray,
    horizon: float,
    interval_days: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    long, starts, ends = build_censor_long(frame, features, horizon, interval_days)
    model_features = features + ["treatment", "interval", "interval_start"]
    n = len(frame)
    K = len(starts)
    q = np.full((n, K), np.nan)
    true, pred_all = [], []

    for f in sorted(np.unique(fold)):
        train_rows = set(np.where(fold != f)[0].tolist())
        test_rows = set(np.where(fold == f)[0].tolist())
        train_mask = long["row_id"].isin(train_rows)
        test_mask = long["row_id"].isin(test_rows)
        y_train = long.loc[train_mask, "censor_event"].astype(int)
        if y_train.nunique() < 2:
            raise ValueError(f"Censoring model fold {f} has one outcome class.")
        model = make_logistic(seed + int(f), cv=3)
        model.fit(long.loc[train_mask, model_features], y_train)
        pred = np.clip(model.predict_proba(long.loc[test_mask, model_features])[:, 1], 1e-5, 0.95)
        info = long.loc[test_mask, ["row_id", "interval"]]
        for (_, row), p in zip(info.iterrows(), pred):
            q[int(row["row_id"]), int(row["interval"])] = p
        true.extend(long.loc[test_mask, "censor_event"].astype(int).tolist())
        pred_all.extend(pred.tolist())

    q = np.nan_to_num(q, nan=0.0)
    G = np.ones((n, K), dtype=float)
    for k in range(1, K):
        G[:, k] = G[:, k - 1] * (1.0 - q[:, k - 1])
    G = np.clip(G, 0.005, 1.0)

    true_arr = np.asarray(true, dtype=int)
    pred_arr = np.asarray(pred_all, dtype=float)
    eps = 1e-12
    logloss = float(-np.mean(true_arr * np.log(np.clip(pred_arr, eps, 1 - eps)) + (1 - true_arr) * np.log(np.clip(1 - pred_arr, eps, 1 - eps))))
    brier = float(np.mean((true_arr - pred_arr) ** 2))
    return G, starts, ends, {
        "censor_log_loss": logloss,
        "censor_brier": brier,
        "G_min": float(G.min()),
        "G_p01": float(np.quantile(G, 0.01)),
    }


def ipcw_rmst_pseudo(
    observed_time: np.ndarray,
    G_start: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    horizon: float,
    g_min: float,
) -> np.ndarray:
    out = np.zeros(len(observed_time), dtype=float)
    for k, (start, end) in enumerate(zip(starts, ends)):
        if start >= horizon:
            break
        effective_end = min(float(end), horizon)
        length = np.maximum(0.0, np.minimum(observed_time, effective_end) - start)
        at_risk = observed_time > start
        out += np.where(at_risk, length / np.clip(G_start[:, k], g_min, 1.0), 0.0)
    return out


def crossfit_arm_outcomes(
    frame: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    fold: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    a = pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy()
    mu0 = np.full(len(frame), np.nan)
    mu1 = np.full(len(frame), np.nan)
    for f in sorted(np.unique(fold)):
        test = fold == f
        train = ~test
        for arm in (0, 1):
            arm_train = train & (a == arm)
            if int(arm_train.sum()) < 20:
                raise ValueError(f"Outcome model arm={arm}, fold={f}: only {int(arm_train.sum())} training rows.")
            model = make_ridge()
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
    score = h * (mu1 - mu0) + h * a / e * (y - mu1) - h * (1 - a) / (1 - e) * (y - mu0)
    theta = float(score.sum() / h.sum())
    influence = (score - theta * h) / np.mean(h)
    return theta, influence


def landmark_estimate(
    frame: pd.DataFrame,
    features: list[str],
    fold: np.ndarray,
    seed: int,
    propensity_scores: np.ndarray | None = None,
) -> dict[str, float]:
    a = pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy()
    time = pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float)
    if propensity_scores is None:
        ps = crossfit_propensity(frame, features, fold, "analysis_treatment", seed)
        propensity_source = "refitted_exact_stage30_specification"
    else:
        ps = np.asarray(propensity_scores, dtype=float)
        if len(ps) != len(frame):
            raise ValueError("Frozen propensity-score length does not match landmark data.")
        if not np.isfinite(ps).all():
            raise ValueError("Frozen Stage 30 propensity scores contain non-finite values.")
        ps = np.clip(ps, 0.01, 0.99)
        propensity_source = "frozen_stage30_oof_scores"
    G, starts, ends, censor_metrics = crossfit_censor_survival(frame, features, fold, LANDMARK_HORIZON, LANDMARK_INTERVAL, seed + 100)
    y = ipcw_rmst_pseudo(time, G, starts, ends, LANDMARK_HORIZON, G_MIN)
    mu0, mu1 = crossfit_arm_outcomes(frame, features, y, fold, seed + 200)
    theta, influence = aipw_ato(y, a, ps, mu0, mu1)
    return {
        "estimate_days": theta,
        "if_se_days": float(np.std(influence, ddof=1) / np.sqrt(len(frame))),
        "ps_min": float(ps.min()),
        "ps_p01": float(np.quantile(ps, 0.01)),
        "ps_p99": float(np.quantile(ps, 0.99)),
        "pseudo_mean": float(y.mean()),
        "pseudo_sd": float(y.std(ddof=1)),
        "pseudo_p99": float(np.quantile(y, 0.99)),
        "pseudo_max": float(y.max()),
        "propensity_source": propensity_source,
        **censor_metrics,
    }


def _clone_long_rows(
    frame: pd.DataFrame,
    features: list[str],
    horizon: float,
    interval_days: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    starts, ends = interval_grid(horizon, interval_days)
    X = frame[features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    natural_time = pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float)
    natural_event = pd.to_numeric(frame["analysis_event"], errors="raise").astype(int).to_numpy()
    initiation = pd.to_numeric(frame["earliest_start_nonnegative"], errors="coerce").to_numpy(float)

    rows = []
    for i in range(len(frame)):
        for strategy in (0, 1):
            if strategy == 0:
                artificial_time = initiation[i] if np.isfinite(initiation[i]) and initiation[i] <= LANDMARK_DAY else np.inf
            else:
                artificial_time = LANDMARK_DAY if not (np.isfinite(initiation[i]) and initiation[i] <= LANDMARK_DAY) else np.inf
            observed_end = min(natural_time[i], artificial_time, horizon)
            for k, (start, end) in enumerate(zip(starts, ends)):
                if observed_end < start:
                    continue
                row = X.iloc[i].to_dict()
                row.update({
                    "row_id": i,
                    "strategy": strategy,
                    "interval": k,
                    "interval_start": start,
                    "interval_end": end,
                    "event": int(natural_event[i] == 1 and start <= natural_time[i] <= end and natural_time[i] <= artificial_time and natural_time[i] <= horizon),
                    "natural_censor": int(natural_event[i] == 0 and start <= natural_time[i] <= end and natural_time[i] <= artificial_time and natural_time[i] <= horizon),
                    "artificial_censor": int(np.isfinite(artificial_time) and start <= artificial_time <= end and artificial_time < natural_time[i] and artificial_time <= horizon),
                })
                rows.append(row)
    return pd.DataFrame(rows), starts, ends


def _crossfit_hazard(
    long: pd.DataFrame,
    features_den: list[str],
    features_num: list[str],
    outcome: str,
    patient_fold: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_den = np.full(len(long), np.nan)
    pred_num = np.full(len(long), np.nan)
    row_fold = patient_fold[long["row_id"].astype(int).to_numpy()]
    y_all = long[outcome].astype(int).to_numpy()

    for f in sorted(np.unique(patient_fold)):
        train = row_fold != f
        test = row_fold == f
        y_train = y_all[train]
        if len(np.unique(y_train)) < 2:
            grouped = long.loc[train].groupby(["strategy", "interval"])[outcome].agg(["sum", "count"])
            global_rate = (y_train.sum() + 0.5) / (len(y_train) + 1.0)
            values = []
            for _, row in long.loc[test, ["strategy", "interval"]].iterrows():
                key = (row["strategy"], row["interval"])
                if key in grouped.index:
                    s = grouped.loc[key, "sum"]
                    n = grouped.loc[key, "count"]
                    values.append((s + 0.5) / (n + 1.0))
                else:
                    values.append(global_rate)
            values = np.clip(np.asarray(values), 1e-5, 0.95)
            pred_den[test] = values
            pred_num[test] = values
            continue
        den = make_logistic(seed + int(f), cv=3)
        num = make_logistic(seed + 100 + int(f), cv=3)
        den.fit(long.loc[train, features_den], y_train)
        num.fit(long.loc[train, features_num], y_train)
        pred_den[test] = den.predict_proba(long.loc[test, features_den])[:, 1]
        pred_num[test] = num.predict_proba(long.loc[test, features_num])[:, 1]
    return np.clip(pred_den, 1e-5, 0.95), np.clip(pred_num, 1e-5, 0.95)


def ccw_estimate(
    frame: pd.DataFrame,
    features: list[str],
    patient_fold: np.ndarray,
    seed: int,
) -> dict[str, float]:
    long, starts, ends = _clone_long_rows(frame, features, CCW_HORIZON, CCW_INTERVAL)
    base_features = features + ["strategy", "interval", "interval_start"]
    numerator_features = ["strategy", "interval", "interval_start"]

    art_den, art_num = _crossfit_hazard(long, base_features, numerator_features, "artificial_censor", patient_fold, seed)
    nat_den, nat_num = _crossfit_hazard(long, base_features, numerator_features, "natural_censor", patient_fold, seed + 500)
    long = long.copy()
    long["art_den"] = art_den
    long["art_num"] = art_num
    long["nat_den"] = nat_den
    long["nat_num"] = nat_num
    long["weight"] = np.nan

    for (_, _), idx in long.groupby(["row_id", "strategy"]).groups.items():
        ordered = long.loc[idx].sort_values("interval")
        cumulative = 1.0
        weights = []
        for _, row in ordered.iterrows():
            weights.append(cumulative)
            cumulative *= (1.0 - row["art_num"]) / (1.0 - row["art_den"])
            cumulative *= (1.0 - row["nat_num"]) / (1.0 - row["nat_den"])
            cumulative = float(np.clip(cumulative, 0.01, 25.0))
        long.loc[ordered.index, "weight"] = weights

    rmst, survival_at_end = {}, {}
    for strategy in (0, 1):
        subset = long[long["strategy"] == strategy]
        survival = 1.0
        area = 0.0
        for k, (start, end) in enumerate(zip(starts, ends)):
            rows = subset[subset["interval"] == k]
            area += survival * (end - start)
            if rows.empty:
                continue
            risk_weight = float(rows["weight"].sum())
            event_weight = float((rows["weight"] * rows["event"]).sum())
            if risk_weight > 0:
                survival *= max(0.0, 1.0 - event_weight / risk_weight)
        rmst[strategy] = area
        survival_at_end[strategy] = survival

    weights = pd.to_numeric(long["weight"], errors="coerce").to_numpy(float)
    return {
        "estimate_days": float(rmst[1] - rmst[0]),
        "rmst_initiate_by_180": float(rmst[1]),
        "rmst_no_initiation_by_180": float(rmst[0]),
        "survival_initiate": float(survival_at_end[1]),
        "survival_no_initiation": float(survival_at_end[0]),
        "weight_mean": float(np.mean(weights)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(np.max(weights)),
        "fraction_weight_gt5": float(np.mean(weights > 5)),
        "clone_rows": len(long),
    }


def checkpoint_config(analysis: str, reps: int, input_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "analysis": analysis,
        "target_reps": reps,
        "cohort": COHORT,
        "landmark_day": LANDMARK_DAY,
        "landmark_horizon": LANDMARK_HORIZON,
        "ccw_horizon": CCW_HORIZON,
        "g_min": G_MIN,
        "base_seed": BASE_SEED,
        "input_hashes": input_hashes,
    }


def validate_or_write_config(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_new = dict(config)
        comparable_existing.pop("target_reps", None)
        comparable_new.pop("target_reps", None)
        if comparable_existing != comparable_new:
            raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
