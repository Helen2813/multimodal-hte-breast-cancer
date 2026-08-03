from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import norm, qmc


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


def empirical_target_values(config: dict) -> dict[str, float]:
    target = config["empirical_targets"]
    full_n = float(target["full_cohort_n"])
    treated = float(target["full_treated"])
    control = float(target["full_control"])
    no_chemo_n = float(target["broad_no_chemo_n"])
    no_chemo_treated = float(target["broad_no_chemo_treated"])
    v10_treated = float(target["v10_treated"])
    v10_control = float(target["v10_control"])

    return {
        "full_treated_fraction": treated / full_n,
        "chemo_fraction_treated": (
            float(target["chemo_by180_treated"]) / treated
        ),
        "chemo_fraction_control": (
            float(target["chemo_by180_control"]) / control
        ),
        "no_chemo_fraction": no_chemo_n / full_n,
        "treated_fraction_no_chemo": no_chemo_treated / no_chemo_n,
        "event_risk_treated_no_chemo": (
            float(target["v10_treated_events"]) / v10_treated
        ),
        "event_risk_control_no_chemo": (
            float(target["v10_control_events"]) / v10_control
        ),
    }


def sobol_baseline(
    power: int,
    seed: int,
    config: dict,
) -> pd.DataFrame:
    fixed = config["calibration"]["fixed_covariate_parameters"]
    sampler = qmc.Sobol(
        d=3,
        scramble=True,
        seed=int(seed),
    )
    u = sampler.random_base2(m=int(power))
    u0 = np.clip(u[:, 0], 1e-12, 1.0 - 1e-12)
    x1 = norm.ppf(u0)
    x2 = (
        u[:, 1] < float(fixed["x2_probability"])
    ).astype(float)
    x3_probability = expit(
        float(fixed["x3_intercept"])
        + float(fixed["x3_x1"]) * x1
    )
    x3 = (u[:, 2] < x3_probability).astype(float)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
    })


def treatment_metrics(
    frame: pd.DataFrame,
    chemo_intercept: float,
    treatment_intercept: float,
    sequencing_strength: float,
    config: dict,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    fixed = config["calibration"]["fixed_covariate_parameters"]
    x1 = frame["x1"].to_numpy(dtype=float)
    x2 = frame["x2"].to_numpy(dtype=float)
    x3 = frame["x3"].to_numpy(dtype=float)

    p_chemo = expit(
        float(chemo_intercept)
        + float(fixed["chemo_x1"]) * x1
        + float(fixed["chemo_x3"]) * x3
    )
    e0 = expit(
        float(treatment_intercept)
        + float(fixed["treatment_x1"]) * x1
        + float(fixed["treatment_x2"]) * x2
    )
    e1 = expit(
        float(treatment_intercept)
        + float(fixed["treatment_x1"]) * x1
        + float(fixed["treatment_x2"]) * x2
        - float(sequencing_strength)
    )

    p_a1 = (1.0 - p_chemo) * e0 + p_chemo * e1
    p_a0 = 1.0 - p_a1

    chemo_and_treated = p_chemo * e1
    chemo_and_control = p_chemo * (1.0 - e1)
    no_chemo_and_treated = (1.0 - p_chemo) * e0

    metrics = {
        "full_treated_fraction": float(np.mean(p_a1)),
        "chemo_fraction_treated": float(
            np.mean(chemo_and_treated) / np.mean(p_a1)
        ),
        "chemo_fraction_control": float(
            np.mean(chemo_and_control) / np.mean(p_a0)
        ),
        "no_chemo_fraction": float(np.mean(1.0 - p_chemo)),
        "treated_fraction_no_chemo": float(
            np.mean(no_chemo_and_treated)
            / np.mean(1.0 - p_chemo)
        ),
    }
    arrays = {
        "p_chemo": p_chemo,
        "e0": e0,
        "e1": e1,
        "p_a1": p_a1,
    }
    return metrics, arrays


def cumulative_event_probability(
    event_hazard: np.ndarray,
    censor_hazard: np.ndarray,
    intervals: int,
) -> np.ndarray:
    event_hazard = np.asarray(event_hazard, dtype=float)
    censor_hazard = np.asarray(censor_hazard, dtype=float)
    stay = (
        (1.0 - event_hazard)
        * (1.0 - censor_hazard)
    )
    denominator = 1.0 - stay
    geometric_sum = np.where(
        np.abs(denominator) > 1e-12,
        (1.0 - stay ** int(intervals)) / denominator,
        float(intervals),
    )
    return event_hazard * geometric_sum


def outcome_metrics(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    event_intercept: float,
    treatment_effect: float,
    config: dict,
) -> dict[str, float]:
    fixed = config["calibration"]["fixed_covariate_parameters"]
    censor = config["calibration"]["censoring_parameters"]
    intervals = int(config["calibration"]["intervals"])

    x1 = frame["x1"].to_numpy(dtype=float)
    x2 = frame["x2"].to_numpy(dtype=float)
    x3 = frame["x3"].to_numpy(dtype=float)
    p_chemo = arrays["p_chemo"]
    e0 = arrays["e0"]
    e1 = arrays["e1"]

    base_no_chemo = (
        float(event_intercept)
        + float(fixed["event_x1"]) * x1
        + float(fixed["event_x2"]) * x2
        + float(fixed["event_x3"]) * x3
    )
    censor_no_chemo = expit(
        float(censor["intercept"])
        + float(censor["x1"]) * x1
    )
    risk_a0_c0 = cumulative_event_probability(
        expit(base_no_chemo),
        censor_no_chemo,
        intervals,
    )
    risk_a1_c0 = cumulative_event_probability(
        expit(base_no_chemo + float(treatment_effect)),
        censor_no_chemo,
        intervals,
    )

    weight_no_chemo_treated = (1.0 - p_chemo) * e0
    weight_no_chemo_control = (
        (1.0 - p_chemo) * (1.0 - e0)
    )

    treated_risk = float(
        np.sum(weight_no_chemo_treated * risk_a1_c0)
        / np.sum(weight_no_chemo_treated)
    )
    control_risk = float(
        np.sum(weight_no_chemo_control * risk_a0_c0)
        / np.sum(weight_no_chemo_control)
    )
    total_risk = float(
        (
            np.sum(weight_no_chemo_treated * risk_a1_c0)
            + np.sum(weight_no_chemo_control * risk_a0_c0)
        )
        / np.sum(1.0 - p_chemo)
    )

    return {
        "event_risk_treated_no_chemo": treated_risk,
        "event_risk_control_no_chemo": control_risk,
        "event_risk_total_no_chemo": total_risk,
    }


def standardized_residuals(
    observed: dict[str, float],
    targets: dict[str, float],
    tolerances: dict[str, float],
    names: list[str],
) -> np.ndarray:
    return np.asarray([
        (float(observed[name]) - float(targets[name]))
        / float(tolerances[name])
        for name in names
    ], dtype=float)


def optimize_treatment(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[dict[str, float], pd.DataFrame]:
    targets = empirical_target_values(config)
    tolerances = config["target_tolerances"]
    bounds = config["calibration"]["parameter_bounds"]
    starts = int(config["calibration"]["multi_start_count"])
    max_nfev = int(config["calibration"]["optimizer_max_nfev"])
    base_start = np.asarray(
        config["calibration"]["starting_points"]["treatment"],
        dtype=float,
    )
    lower = np.asarray([
        bounds["chemo_logit_intercept"][0],
        bounds["treatment_logit_intercept"][0],
        bounds["sequencing_strength"][0],
    ], dtype=float)
    upper = np.asarray([
        bounds["chemo_logit_intercept"][1],
        bounds["treatment_logit_intercept"][1],
        bounds["sequencing_strength"][1],
    ], dtype=float)
    names = [
        "full_treated_fraction",
        "chemo_fraction_treated",
        "chemo_fraction_control",
        "no_chemo_fraction",
        "treated_fraction_no_chemo",
    ]

    rng = np.random.default_rng(32032)
    start_points = [np.clip(base_start, lower, upper)]
    for _ in range(starts - 1):
        start_points.append(rng.uniform(lower, upper))

    run_rows = []
    best = None
    for run_number, start in enumerate(start_points, start=1):
        def objective(parameters: np.ndarray) -> np.ndarray:
            metrics, _ = treatment_metrics(
                frame,
                parameters[0],
                parameters[1],
                parameters[2],
                config,
            )
            return standardized_residuals(
                metrics,
                targets,
                tolerances,
                names,
            )

        result = least_squares(
            objective,
            x0=start,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        residual = objective(result.x)
        row = {
            "optimization_component": "treatment",
            "run_number": run_number,
            "success": bool(result.success),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "maximum_absolute_standardized_residual": float(
                np.max(np.abs(residual))
            ),
            "rms_standardized_residual": float(
                np.sqrt(np.mean(residual ** 2))
            ),
            "chemo_logit_intercept": float(result.x[0]),
            "treatment_logit_intercept": float(result.x[1]),
            "sequencing_strength": float(result.x[2]),
            "message": str(result.message),
        }
        run_rows.append(row)
        if best is None or row["cost"] < best["row"]["cost"]:
            best = {
                "row": row,
                "result": result,
            }

    parameters = {
        "chemo_logit_intercept": float(best["result"].x[0]),
        "treatment_logit_intercept": float(best["result"].x[1]),
        "sequencing_strength": float(best["result"].x[2]),
    }
    return parameters, pd.DataFrame(run_rows)


def optimize_outcome(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    config: dict,
) -> tuple[dict[str, float], pd.DataFrame]:
    targets = empirical_target_values(config)
    tolerances = config["target_tolerances"]
    bounds = config["calibration"]["parameter_bounds"]
    starts = int(config["calibration"]["multi_start_count"])
    max_nfev = int(config["calibration"]["optimizer_max_nfev"])
    base_start = np.asarray(
        config["calibration"]["starting_points"]["outcome"],
        dtype=float,
    )
    lower = np.asarray([
        bounds["event_logit_intercept"][0],
        bounds["true_treatment_log_hazard_effect"][0],
    ], dtype=float)
    upper = np.asarray([
        bounds["event_logit_intercept"][1],
        bounds["true_treatment_log_hazard_effect"][1],
    ], dtype=float)
    names = [
        "event_risk_treated_no_chemo",
        "event_risk_control_no_chemo",
    ]

    rng = np.random.default_rng(32033)
    start_points = [np.clip(base_start, lower, upper)]
    for _ in range(starts - 1):
        start_points.append(rng.uniform(lower, upper))

    run_rows = []
    best = None
    for run_number, start in enumerate(start_points, start=1):
        def objective(parameters: np.ndarray) -> np.ndarray:
            metrics = outcome_metrics(
                frame,
                arrays,
                parameters[0],
                parameters[1],
                config,
            )
            return standardized_residuals(
                metrics,
                targets,
                tolerances,
                names,
            )

        result = least_squares(
            objective,
            x0=start,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        residual = objective(result.x)
        row = {
            "optimization_component": "outcome",
            "run_number": run_number,
            "success": bool(result.success),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "maximum_absolute_standardized_residual": float(
                np.max(np.abs(residual))
            ),
            "rms_standardized_residual": float(
                np.sqrt(np.mean(residual ** 2))
            ),
            "event_logit_intercept": float(result.x[0]),
            "true_treatment_log_hazard_effect": float(result.x[1]),
            "message": str(result.message),
        }
        run_rows.append(row)
        if best is None or row["cost"] < best["row"]["cost"]:
            best = {
                "row": row,
                "result": result,
            }

    parameters = {
        "event_logit_intercept": float(best["result"].x[0]),
        "true_treatment_log_hazard_effect": float(
            best["result"].x[1]
        ),
    }
    return parameters, pd.DataFrame(run_rows)


def evaluate_parameter_set(
    frame: pd.DataFrame,
    parameters: dict[str, float],
    config: dict,
) -> dict[str, float]:
    treatment, arrays = treatment_metrics(
        frame,
        parameters["chemo_logit_intercept"],
        parameters["treatment_logit_intercept"],
        parameters["sequencing_strength"],
        config,
    )
    outcome = outcome_metrics(
        frame,
        arrays,
        parameters["event_logit_intercept"],
        parameters["true_treatment_log_hazard_effect"],
        config,
    )
    return {**treatment, **outcome}
