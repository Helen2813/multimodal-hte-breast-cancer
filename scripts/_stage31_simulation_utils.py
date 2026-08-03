from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def project_root() -> Path:
    return Path.cwd().resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(data: object) -> str:
    raw = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_logit_fit(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    covariates: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    numeric = frame[covariates].apply(
        pd.to_numeric,
        errors="raise",
    )
    means = numeric.mean(axis=0)
    standard_deviations = numeric.std(axis=0, ddof=0)
    active = [
        column
        for column in covariates
        if float(standard_deviations[column]) > 0
    ]
    standardized = (
        numeric[active] - means[active]
    ) / standard_deviations[active]
    design = sm.add_constant(
        standardized,
        has_constant="add",
    )
    model = sm.GLM(
        np.asarray(treatment, dtype=int),
        design,
        family=sm.families.Binomial(),
    )
    result = model.fit(
        maxiter=1000,
        tol=1e-10,
        disp=0,
    )
    propensity = np.asarray(
        result.predict(design),
        dtype=float,
    )
    return propensity, {
        "converged": bool(result.converged),
        "maximum_absolute_coefficient": float(
            np.max(np.abs(np.asarray(result.params, dtype=float)))
        ),
        "active_covariates": active,
    }


def make_folds(
    treatment: np.ndarray,
    event: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=int)
    event = np.asarray(event, dtype=int)
    labels = treatment.astype(str) + "_" + event.astype(str)
    counts = pd.Series(labels).value_counts()

    if len(counts) >= 2 and int(counts.min()) >= folds:
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(
            np.zeros(len(labels)),
            labels,
        )
    else:
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(np.zeros(len(labels)))

    fold = np.empty(len(labels), dtype=int)
    for fold_number, (_, test_index) in enumerate(iterator):
        fold[test_index] = fold_number
    return fold


def simulate_dataset(
    n: int,
    sequencing_strength: float,
    seed: int,
    config: dict,
) -> pd.DataFrame:
    settings = config["simulation"]
    parameters = settings["data_generating_parameters"]
    intervals = int(settings["intervals"])
    interval_days = float(settings["interval_days"])
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.binomial(1, 0.45, n)
    x3 = rng.binomial(1, expit(-0.2 + 0.55 * x1), n)

    chemo_linear = (
        parameters["chemo_logit_intercept"]
        + parameters["chemo_x1"] * x1
        + parameters["chemo_x3"] * x3
    )
    chemo_probability = expit(chemo_linear)
    chemo = rng.binomial(1, chemo_probability)

    treatment_linear = (
        parameters["treatment_logit_intercept"]
        + parameters["treatment_x1"] * x1
        + parameters["treatment_x2"] * x2
        - sequencing_strength * chemo
    )
    true_propensity = expit(treatment_linear)
    treatment = rng.binomial(1, true_propensity)

    event_linear = (
        parameters["event_logit_intercept"]
        + parameters["event_x1"] * x1
        + parameters["event_x2"] * x2
        + parameters["event_x3"] * x3
        + parameters["event_chemo"] * chemo
        + settings["true_treatment_log_hazard_effect"]
        * treatment
    )
    event_hazard = expit(event_linear)

    censor_linear = (
        parameters["censor_logit_intercept"]
        + parameters["censor_x1"] * x1
        + parameters["censor_chemo"] * chemo
    )
    censor_hazard = expit(censor_linear)

    event_interval = np.full(n, intervals, dtype=int)
    censor_interval = np.full(n, intervals, dtype=int)
    active = np.ones(n, dtype=bool)

    for interval in range(intervals):
        event_draw = rng.binomial(1, event_hazard)
        event_now = active & (event_draw == 1)
        event_interval[event_now] = interval
        active[event_now] = False

        censor_draw = rng.binomial(1, censor_hazard)
        censor_now = active & (censor_draw == 1)
        censor_interval[censor_now] = interval
        active[censor_now] = False

    observed_event = (
        (event_interval < censor_interval)
        & (event_interval < intervals)
    ).astype(int)
    observed_end_interval = np.minimum(
        np.minimum(event_interval, censor_interval),
        intervals - 1,
    )
    observed_time = (
        observed_end_interval + 1
    ) * interval_days

    event_hazard_a0 = expit(
        parameters["event_logit_intercept"]
        + parameters["event_x1"] * x1
        + parameters["event_x2"] * x2
        + parameters["event_x3"] * x3
        + parameters["event_chemo"] * chemo
    )
    event_hazard_a1 = expit(
        parameters["event_logit_intercept"]
        + parameters["event_x1"] * x1
        + parameters["event_x2"] * x2
        + parameters["event_x3"] * x3
        + parameters["event_chemo"] * chemo
        + settings["true_treatment_log_hazard_effect"]
    )

    powers = np.arange(intervals, dtype=float)
    rmst0 = interval_days * np.sum(
        (1.0 - event_hazard_a0[:, None]) ** powers[None, :],
        axis=1,
    )
    rmst1 = interval_days * np.sum(
        (1.0 - event_hazard_a1[:, None]) ** powers[None, :],
        axis=1,
    )

    frame = pd.DataFrame({
        "patient_id": np.arange(n, dtype=int),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "chemo_by_day180": chemo,
        "analysis_treatment": treatment,
        "analysis_event": observed_event,
        "analysis_time": observed_time,
        "event_interval": event_interval,
        "censor_interval": censor_interval,
        "true_propensity": true_propensity,
        "true_rmst0": rmst0,
        "true_rmst1": rmst1,
    })
    return frame


def ato_truth(
    frame: pd.DataFrame,
) -> float:
    propensity = frame["true_propensity"].to_numpy(dtype=float)
    h = propensity * (1.0 - propensity)
    contrast = (
        frame["true_rmst1"].to_numpy(dtype=float)
        - frame["true_rmst0"].to_numpy(dtype=float)
    )
    return float(np.sum(h * contrast) / np.sum(h))


def build_person_period(
    frame: pd.DataFrame,
    covariates: list[str],
    intervals: int,
) -> pd.DataFrame:
    rows = []
    for row_index, row in frame.iterrows():
        event_interval = int(row["event_interval"])
        censor_interval = int(row["censor_interval"])
        final_interval = min(
            event_interval,
            censor_interval,
            intervals - 1,
        )
        for interval in range(final_interval + 1):
            censored_now = int(
                censor_interval == interval
                and censor_interval < event_interval
                and censor_interval < intervals
            )
            record = {
                "row_index": int(row_index),
                "interval": interval,
                "censored_now": censored_now,
                "analysis_treatment": int(
                    row["analysis_treatment"]
                ),
            }
            for column in covariates:
                record[column] = float(row[column])
            rows.append(record)
            if censored_now or (
                event_interval == interval
                and event_interval < censor_interval
            ):
                break
    return pd.DataFrame(rows)


def crossfit_censoring_survival(
    frame: pd.DataFrame,
    covariates: list[str],
    fold: np.ndarray,
    intervals: int,
) -> np.ndarray:
    n = len(frame)
    G = np.ones((n, intervals), dtype=float)

    for fold_number in sorted(np.unique(fold)):
        train_index = np.flatnonzero(fold != fold_number)
        test_index = np.flatnonzero(fold == fold_number)

        train_period = build_person_period(
            frame.iloc[train_index].reset_index(drop=True),
            covariates,
            intervals,
        )
        feature_columns = (
            covariates
            + ["analysis_treatment", "interval"]
        )

        if train_period["censored_now"].nunique() < 2:
            constant_hazard = float(
                train_period["censored_now"].mean()
            )
            test_hazard = np.full(
                (len(test_index), intervals),
                constant_hazard,
                dtype=float,
            )
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(
                train_period[feature_columns]
            )
            model = LogisticRegression(
                penalty="l2",
                C=10.0,
                solver="lbfgs",
                max_iter=2000,
            )
            model.fit(
                X_train,
                train_period["censored_now"].to_numpy(dtype=int),
            )

            prediction_rows = []
            for local_position, patient_index in enumerate(test_index):
                for interval in range(intervals):
                    record = {
                        column: float(
                            frame.iloc[patient_index][column]
                        )
                        for column in covariates
                    }
                    record["analysis_treatment"] = int(
                        frame.iloc[patient_index][
                            "analysis_treatment"
                        ]
                    )
                    record["interval"] = interval
                    record["local_position"] = local_position
                    prediction_rows.append(record)

            prediction_frame = pd.DataFrame(prediction_rows)
            X_test = scaler.transform(
                prediction_frame[feature_columns]
            )
            predicted = model.predict_proba(X_test)[:, 1]
            test_hazard = predicted.reshape(
                len(test_index),
                intervals,
            )

        test_hazard = np.clip(test_hazard, 1e-6, 0.95)
        survival = np.ones_like(test_hazard)
        for interval in range(1, intervals):
            survival[:, interval] = (
                survival[:, interval - 1]
                * (1.0 - test_hazard[:, interval - 1])
            )
        G[test_index, :] = survival

    return G


def ipcw_rmst_pseudo(
    frame: pd.DataFrame,
    G: np.ndarray,
    intervals: int,
    interval_days: float,
    g_min: float,
) -> np.ndarray:
    event_interval = frame[
        "event_interval"
    ].to_numpy(dtype=int)
    censor_interval = frame[
        "censor_interval"
    ].to_numpy(dtype=int)

    pseudo = np.zeros(len(frame), dtype=float)
    for interval in range(intervals):
        observed_at_start = (
            (event_interval >= interval)
            & (censor_interval >= interval)
        )
        pseudo += (
            interval_days
            * observed_at_start
            / np.maximum(G[:, interval], g_min)
        )
    return pseudo


def crossfit_outcomes(
    frame: pd.DataFrame,
    covariates: list[str],
    pseudo: np.ndarray,
    fold: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    mu0 = np.empty(n, dtype=float)
    mu1 = np.empty(n, dtype=float)
    treatment = frame[
        "analysis_treatment"
    ].to_numpy(dtype=int)

    for fold_number in sorted(np.unique(fold)):
        train = fold != fold_number
        test = fold == fold_number

        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            frame.loc[train, covariates]
        )
        X_test = scaler.transform(
            frame.loc[test, covariates]
        )

        for arm, target in [(0, mu0), (1, mu1)]:
            arm_train = train.copy()
            arm_train[train] = (
                treatment[train] == arm
            )
            if int(arm_train.sum()) < 5:
                raise RuntimeError(
                    f"Too few training patients in arm {arm}."
                )

            scaler_arm = StandardScaler()
            X_arm = scaler_arm.fit_transform(
                frame.loc[arm_train, covariates]
            )
            X_test_arm = scaler_arm.transform(
                frame.loc[test, covariates]
            )
            model = RidgeCV(
                alphas=np.logspace(-3, 3, 25)
            )
            model.fit(X_arm, pseudo[arm_train])
            target[test] = np.clip(
                model.predict(X_test_arm),
                0.0,
                horizon,
            )

    return mu0, mu1


def weighted_balance(
    frame: pd.DataFrame,
    covariates: list[str],
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> tuple[float, float, float]:
    treated = treatment == 1
    control = treatment == 0
    w1 = 1.0 - propensity[treated]
    w0 = propensity[control]

    smds = []
    for column in covariates:
        values = frame[column].to_numpy(dtype=float)
        value1 = values[treated]
        value0 = values[control]

        mean1 = float(np.sum(w1 * value1) / np.sum(w1))
        mean0 = float(np.sum(w0 * value0) / np.sum(w0))
        variance1 = float(
            np.sum(w1 * (value1 - mean1) ** 2)
            / np.sum(w1)
        )
        variance0 = float(
            np.sum(w0 * (value0 - mean0) ** 2)
            / np.sum(w0)
        )
        pooled = (variance1 + variance0) / 2.0
        smd = (
            0.0
            if pooled <= 0 and mean1 == mean0
            else (mean1 - mean0) / math.sqrt(pooled)
        )
        smds.append(abs(float(smd)))

    ess1 = float(
        np.sum(w1) ** 2 / np.sum(w1 ** 2)
    )
    ess0 = float(
        np.sum(w0) ** 2 / np.sum(w0 ** 2)
    )
    return (
        float(max(smds)),
        float(ess1 / treated.sum()),
        float(ess0 / control.sum()),
    )


def estimate_method(
    frame: pd.DataFrame,
    covariates: list[str],
    truth: float,
    seed: int,
    config: dict,
) -> dict[str, Any]:
    settings = config["simulation"]
    treatment = frame[
        "analysis_treatment"
    ].to_numpy(dtype=int)
    event = frame[
        "analysis_event"
    ].to_numpy(dtype=int)

    if treatment.sum() < 10 or (1 - treatment).sum() < 10:
        raise RuntimeError("Insufficient treatment-arm size.")

    propensity, fit_summary = stable_logit_fit(
        frame,
        treatment,
        covariates,
    )
    fold = make_folds(
        treatment,
        event,
        int(settings["folds"]),
        seed,
    )
    G = crossfit_censoring_survival(
        frame,
        covariates,
        fold,
        int(settings["intervals"]),
    )
    pseudo = ipcw_rmst_pseudo(
        frame,
        G,
        int(settings["intervals"]),
        float(settings["interval_days"]),
        float(settings["censoring_g_min"]),
    )
    mu0, mu1 = crossfit_outcomes(
        frame,
        covariates,
        pseudo,
        fold,
        float(settings["horizon_days"]),
    )

    h = propensity * (1.0 - propensity)
    numerator = (
        h * (mu1 - mu0)
        + treatment
        * (1.0 - propensity)
        * (pseudo - mu1)
        - (1 - treatment)
        * propensity
        * (pseudo - mu0)
    )
    denominator = float(np.sum(h))
    estimate = float(np.sum(numerator) / denominator)
    mean_h = float(np.mean(h))
    influence = (
        numerator - estimate * h
    ) / mean_h
    standard_error = float(
        np.std(influence, ddof=1)
        / math.sqrt(len(frame))
    )
    ci_low = estimate - 1.96 * standard_error
    ci_high = estimate + 1.96 * standard_error

    max_smd, ess_treated, ess_control = (
        weighted_balance(
            frame,
            covariates,
            treatment,
            propensity,
        )
    )

    return {
        "n_analysis": len(frame),
        "treated": int(treatment.sum()),
        "control": int((1 - treatment).sum()),
        "events": int(event.sum()),
        "treated_events": int(
            ((treatment == 1) & (event == 1)).sum()
        ),
        "control_events": int(
            ((treatment == 0) & (event == 1)).sum()
        ),
        "truth_days": float(truth),
        "estimate_days": estimate,
        "bias_days": estimate - truth,
        "squared_error": (estimate - truth) ** 2,
        "diagnostic_if_se_days": standard_error,
        "ci_low_days": ci_low,
        "ci_high_days": ci_high,
        "covered": bool(ci_low <= truth <= ci_high),
        "max_abs_weighted_smd": max_smd,
        "ato_ess_fraction_treated": ess_treated,
        "ato_ess_fraction_control": ess_control,
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(
            np.quantile(propensity, 0.01)
        ),
        "propensity_p99": float(
            np.quantile(propensity, 0.99)
        ),
        "propensity_max": float(np.max(propensity)),
        "propensity_fit_converged": bool(
            fit_summary["converged"]
        ),
        "maximum_absolute_propensity_coefficient": float(
            fit_summary["maximum_absolute_coefficient"]
        ),
        "pseudo_p99": float(np.quantile(pseudo, 0.99)),
        "pseudo_max": float(np.max(pseudo)),
    }
