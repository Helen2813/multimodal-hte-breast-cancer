from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from _stage31_simulation_utils import (
    crossfit_censoring_survival,
    crossfit_outcomes,
    ipcw_rmst_pseudo,
    make_folds,
    stable_logit_fit,
)


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


def binary_smd(
    values: np.ndarray,
    treatment: np.ndarray,
    weights_treated: np.ndarray | None = None,
    weights_control: np.ndarray | None = None,
) -> float:
    values = np.asarray(values, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    treated = treatment == 1
    control = treatment == 0

    if weights_treated is None:
        weights_treated = np.ones(int(treated.sum()), dtype=float)
    if weights_control is None:
        weights_control = np.ones(int(control.sum()), dtype=float)

    p1 = float(
        np.sum(weights_treated * values[treated])
        / np.sum(weights_treated)
    )
    p0 = float(
        np.sum(weights_control * values[control])
        / np.sum(weights_control)
    )
    pooled = (p1 * (1.0 - p1) + p0 * (1.0 - p0)) / 2.0
    if pooled <= 0:
        return 0.0 if p1 == p0 else float("inf")
    return float((p1 - p0) / math.sqrt(pooled))


def continuous_smd(
    values: np.ndarray,
    treatment: np.ndarray,
    weights_treated: np.ndarray | None = None,
    weights_control: np.ndarray | None = None,
) -> float:
    values = np.asarray(values, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    treated = treatment == 1
    control = treatment == 0

    if weights_treated is None:
        weights_treated = np.ones(int(treated.sum()), dtype=float)
    if weights_control is None:
        weights_control = np.ones(int(control.sum()), dtype=float)

    value1 = values[treated]
    value0 = values[control]
    mean1 = float(np.sum(weights_treated * value1) / np.sum(weights_treated))
    mean0 = float(np.sum(weights_control * value0) / np.sum(weights_control))
    var1 = float(
        np.sum(weights_treated * (value1 - mean1) ** 2)
        / np.sum(weights_treated)
    )
    var0 = float(
        np.sum(weights_control * (value0 - mean0) ** 2)
        / np.sum(weights_control)
    )
    pooled = (var1 + var0) / 2.0
    if pooled <= 0:
        return 0.0 if mean1 == mean0 else float("inf")
    return float((mean1 - mean0) / math.sqrt(pooled))


def maximum_weighted_smd(
    frame: pd.DataFrame,
    covariates: list[str],
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> float:
    treatment = np.asarray(treatment, dtype=int)
    treated = treatment == 1
    control = treatment == 0
    weights_treated = 1.0 - propensity[treated]
    weights_control = propensity[control]

    smds = []
    for column in covariates:
        values = frame[column].to_numpy(dtype=float)
        unique = np.unique(values[~np.isnan(values)])
        if set(unique).issubset({0.0, 1.0}):
            smd = binary_smd(
                values,
                treatment,
                weights_treated,
                weights_control,
            )
        else:
            smd = continuous_smd(
                values,
                treatment,
                weights_treated,
                weights_control,
            )
        smds.append(abs(float(smd)))
    return float(max(smds))


def ato_ess_fractions(
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> tuple[float, float]:
    treatment = np.asarray(treatment, dtype=int)
    treated = treatment == 1
    control = treatment == 0
    w1 = 1.0 - propensity[treated]
    w0 = propensity[control]
    ess1 = float(np.sum(w1) ** 2 / np.sum(w1 ** 2))
    ess0 = float(np.sum(w0) ** 2 / np.sum(w0 ** 2))
    return (
        ess1 / int(treated.sum()),
        ess0 / int(control.sum()),
    )


def rmst_from_hazard(
    hazard: np.ndarray,
    intervals: int,
    interval_days: float,
) -> np.ndarray:
    hazard = np.asarray(hazard, dtype=float)
    powers = np.arange(intervals, dtype=float)
    return interval_days * np.sum(
        (1.0 - hazard[:, None]) ** powers[None, :],
        axis=1,
    )


def simulate_dataset_v2(
    n: int,
    sequencing_strength: float,
    treatment_effect: float,
    seed: int,
    calibrated: dict[str, float],
    fixed: dict[str, float],
    censor: dict[str, float],
    config: dict,
) -> pd.DataFrame:
    settings = config["simulation"]
    intervals = int(settings["intervals"])
    interval_days = float(settings["interval_days"])
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.binomial(1, float(fixed["x2_probability"]), n)
    x3 = rng.binomial(
        1,
        expit(
            float(fixed["x3_intercept"])
            + float(fixed["x3_x1"]) * x1
        ),
    )

    p_chemo = expit(
        float(calibrated["chemo_logit_intercept"])
        + float(fixed["chemo_x1"]) * x1
        + float(fixed["chemo_x3"]) * x3
    )
    chemo = rng.binomial(1, p_chemo)

    e0 = expit(
        float(calibrated["treatment_logit_intercept"])
        + float(fixed["treatment_x1"]) * x1
        + float(fixed["treatment_x2"]) * x2
    )
    e1 = expit(
        float(calibrated["treatment_logit_intercept"])
        + float(fixed["treatment_x1"]) * x1
        + float(fixed["treatment_x2"]) * x2
        - float(sequencing_strength)
    )
    true_propensity = np.where(chemo == 1, e1, e0)
    marginal_propensity = (
        (1.0 - p_chemo) * e0 + p_chemo * e1
    )
    treatment = rng.binomial(1, true_propensity)

    ascertainable = rng.binomial(
        1,
        float(settings["timing_ascertainability_probability"]),
        n,
    )
    strict_eligible = (
        (chemo == 0) & (ascertainable == 1)
    ).astype(int)

    base_event_linear = (
        float(calibrated["event_logit_intercept"])
        + float(fixed["event_x1"]) * x1
        + float(fixed["event_x2"]) * x2
        + float(fixed["event_x3"]) * x3
        + float(fixed["event_chemo"]) * chemo
    )
    event_hazard = expit(
        base_event_linear + float(treatment_effect) * treatment
    )
    censor_hazard = expit(
        float(censor["intercept"])
        + float(censor["x1"]) * x1
        + float(censor["chemo"]) * chemo
    )

    event_interval = np.full(n, intervals, dtype=int)
    censor_interval = np.full(n, intervals, dtype=int)
    active = np.ones(n, dtype=bool)

    for interval in range(intervals):
        event_now = active & (
            rng.binomial(1, event_hazard) == 1
        )
        event_interval[event_now] = interval
        active[event_now] = False

        censor_now = active & (
            rng.binomial(1, censor_hazard) == 1
        )
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

    hazard_a0 = expit(base_event_linear)
    hazard_a1 = expit(
        base_event_linear + float(treatment_effect)
    )
    rmst0 = rmst_from_hazard(
        hazard_a0,
        intervals,
        interval_days,
    )
    rmst1 = rmst_from_hazard(
        hazard_a1,
        intervals,
        interval_days,
    )

    return pd.DataFrame({
        "patient_id": np.arange(n, dtype=int),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "chemo_probability": p_chemo,
        "chemo_by_day180": chemo,
        "timing_ascertainable": ascertainable,
        "strict_sequence_eligible": strict_eligible,
        "analysis_treatment": treatment,
        "analysis_event": observed_event,
        "analysis_time": observed_time,
        "event_interval": event_interval,
        "censor_interval": censor_interval,
        "true_propensity": true_propensity,
        "marginal_propensity": marginal_propensity,
        "no_chemo_propensity": e0,
        "true_rmst0": rmst0,
        "true_rmst1": rmst1,
    })


def weighted_truth(
    frame: pd.DataFrame,
    propensity_column: str,
) -> float:
    propensity = frame[propensity_column].to_numpy(dtype=float)
    h = propensity * (1.0 - propensity)
    contrast = (
        frame["true_rmst1"].to_numpy(dtype=float)
        - frame["true_rmst0"].to_numpy(dtype=float)
    )
    return float(np.sum(h * contrast) / np.sum(h))


def truth_bundle(
    full_frame: pd.DataFrame,
) -> dict[str, float]:
    strict = full_frame[
        full_frame["strict_sequence_eligible"] == 1
    ].reset_index(drop=True)
    return {
        "intended_full_ato_truth": weighted_truth(
            full_frame,
            "true_propensity",
        ),
        "naive_implied_overlap_truth": weighted_truth(
            full_frame,
            "marginal_propensity",
        ),
        "strict_no_chemo_ato_truth": weighted_truth(
            strict,
            "no_chemo_propensity",
        ),
    }


def estimate_method_v2(
    frame: pd.DataFrame,
    covariates: list[str],
    primary_truth: float,
    seed: int,
    config: dict,
    secondary_truth: float | None = None,
) -> dict[str, Any]:
    settings = config["simulation"]
    treatment = frame[
        "analysis_treatment"
    ].to_numpy(dtype=int)
    event = frame[
        "analysis_event"
    ].to_numpy(dtype=int)

    if int(treatment.sum()) < 10 or int((1 - treatment).sum()) < 10:
        raise RuntimeError("Insufficient treatment-arm size.")

    propensity, fit = stable_logit_fit(
        frame,
        treatment,
        covariates,
    )
    fold = make_folds(
        treatment,
        event,
        int(settings["folds"]),
        int(seed),
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
    estimate = float(np.sum(numerator) / np.sum(h))
    influence = (
        numerator - estimate * h
    ) / float(np.mean(h))
    standard_error = float(
        np.std(influence, ddof=1) / math.sqrt(len(frame))
    )
    ci_low = estimate - 1.96 * standard_error
    ci_high = estimate + 1.96 * standard_error

    treated = treatment == 1
    control = treatment == 0
    w1 = 1.0 - propensity[treated]
    w0 = propensity[control]
    chemo = frame[
        "chemo_by_day180"
    ].to_numpy(dtype=float)

    unweighted_chemo_smd = binary_smd(
        chemo,
        treatment,
    )
    weighted_chemo_smd = binary_smd(
        chemo,
        treatment,
        w1,
        w0,
    )
    included_smd = maximum_weighted_smd(
        frame,
        covariates,
        treatment,
        propensity,
    )
    ess_treated, ess_control = ato_ess_fractions(
        treatment,
        propensity,
    )

    result = {
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
        "primary_truth_days": float(primary_truth),
        "estimate_days": estimate,
        "primary_bias_days": estimate - float(primary_truth),
        "primary_squared_error": (
            estimate - float(primary_truth)
        ) ** 2,
        "diagnostic_if_se_days": standard_error,
        "primary_ci_low_days": ci_low,
        "primary_ci_high_days": ci_high,
        "primary_covered": bool(
            ci_low <= float(primary_truth) <= ci_high
        ),
        "included_covariate_max_abs_weighted_smd": included_smd,
        "unweighted_chemo_smd": unweighted_chemo_smd,
        "weighted_chemo_smd": weighted_chemo_smd,
        "chemo_fraction_treated": float(
            chemo[treated].mean()
        ),
        "chemo_fraction_control": float(
            chemo[control].mean()
        ),
        "ato_ess_fraction_treated": ess_treated,
        "ato_ess_fraction_control": ess_control,
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(np.max(propensity)),
        "propensity_fit_converged": bool(fit["converged"]),
        "maximum_absolute_propensity_coefficient": float(
            fit["maximum_absolute_coefficient"]
        ),
        "pseudo_p99": float(np.quantile(pseudo, 0.99)),
        "pseudo_max": float(np.max(pseudo)),
    }

    if secondary_truth is not None:
        secondary_truth = float(secondary_truth)
        result.update({
            "secondary_truth_days": secondary_truth,
            "secondary_bias_days": estimate - secondary_truth,
            "secondary_ci_low_days": ci_low,
            "secondary_ci_high_days": ci_high,
            "secondary_covered": bool(
                ci_low <= secondary_truth <= ci_high
            ),
            "target_drift_days": (
                secondary_truth - float(primary_truth)
            ),
            "residual_omitted_sequence_bias_days": (
                estimate - secondary_truth
            ),
        })
    else:
        result.update({
            "secondary_truth_days": np.nan,
            "secondary_bias_days": np.nan,
            "secondary_ci_low_days": np.nan,
            "secondary_ci_high_days": np.nan,
            "secondary_covered": np.nan,
            "target_drift_days": np.nan,
            "residual_omitted_sequence_bias_days": np.nan,
        })

    return result
