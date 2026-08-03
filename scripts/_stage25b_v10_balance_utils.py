from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def compact_features(frame: pd.DataFrame) -> list[str]:
    features = [
        column
        for column in frame.columns
        if str(column).startswith("W_")
    ]
    for column in ("diagnosis_year", "diagnosis_year_missing"):
        if column in frame.columns:
            features.append(column)
    return list(dict.fromkeys(features))


def dataframe_console(
    frame: pd.DataFrame,
    max_rows: int | None = None,
) -> str:
    if frame.empty:
        return "<empty table>"
    view = frame if max_rows is None else frame.head(max_rows)
    with pd.option_context(
        "display.max_rows",
        None if max_rows is None else max_rows,
        "display.max_columns",
        None,
        "display.width",
        360,
        "display.max_colwidth",
        100,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        return view.to_string(index=False)


def verify_stage25_inputs(root: Path, config: dict) -> dict[str, Any]:
    source = config["source"]
    summary = load_json(root / source["stage25_cohort_summary"])

    cohort_path = root / source["v10_cohort"]
    compact_path = root / source["v10_compact"]
    if not cohort_path.exists() or not compact_path.exists():
        raise FileNotFoundError("Frozen Stage 25 V10 cohort files are missing.")

    observed_cohort_hash = sha256_file(cohort_path)
    observed_compact_hash = sha256_file(compact_path)
    if observed_cohort_hash != summary["candidate_v10_cohort_sha256"]:
        raise RuntimeError("Frozen V10 cohort hash mismatch.")
    if observed_compact_hash != summary["candidate_v10_compact_sha256"]:
        raise RuntimeError("Frozen V10 compact-table hash mismatch.")

    failed_gates = pd.read_csv(
        root / source["stage25_failed_gates"],
        low_memory=False,
    )
    failed = failed_gates[
        failed_gates["pass"].astype(str).str.lower() != "true"
    ]
    if len(failed) != 1:
        raise RuntimeError(
            "Expected exactly one Stage 25 pre-effect gate failure.\n"
            + dataframe_console(failed)
        )
    if str(failed.iloc[0]["check"]) != "maximum absolute ATO-weighted SMD":
        raise RuntimeError(
            "The sole Stage 25 failure is not the expected balance gate."
        )

    return {
        "cohort_sha256": observed_cohort_hash,
        "compact_sha256": observed_compact_hash,
        "stage25_summary": summary,
        "stage25_failed_gate": failed.iloc[0].to_dict(),
    }


def prepare_design_matrix(
    compact: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = compact[features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    medians = numeric.median(axis=0).fillna(0.0)
    imputed = numeric.fillna(medians)
    means = imputed.mean(axis=0)
    standard_deviations = (
        imputed.std(axis=0, ddof=0)
        .replace(0.0, np.nan)
    )
    active_features = [
        feature
        for feature in features
        if np.isfinite(standard_deviations[feature])
        and float(standard_deviations[feature]) > 0
    ]
    standardized = (
        imputed[active_features] - means[active_features]
    ) / standard_deviations[active_features]
    design = sm.add_constant(
        standardized,
        has_constant="add",
    )
    metadata = {
        "features": features,
        "active_features": active_features,
        "zero_variance_features": [
            feature
            for feature in features
            if feature not in active_features
        ],
        "medians": {
            key: float(value)
            for key, value in medians.items()
        },
        "means": {
            key: float(value)
            for key, value in means.items()
        },
        "standard_deviations": {
            key: (
                float(value)
                if np.isfinite(value)
                else None
            )
            for key, value in standard_deviations.items()
        },
    }
    return design, metadata


def fit_unpenalized_propensity(
    compact: pd.DataFrame,
    treatment: np.ndarray,
    features: list[str],
    config: dict,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    design, metadata = prepare_design_matrix(
        compact,
        features,
    )
    repair = config["design_repair"]
    model = sm.GLM(
        np.asarray(treatment, dtype=int),
        design,
        family=sm.families.Binomial(),
    )
    result = model.fit(
        maxiter=int(repair["maximum_iterations"]),
        tol=float(repair["tolerance"]),
        disp=0,
    )
    propensity = np.asarray(
        result.predict(design),
        dtype=float,
    )
    coefficient_table = pd.DataFrame({
        "term": design.columns.astype(str),
        "coefficient": np.asarray(result.params, dtype=float),
        "standard_error": np.asarray(result.bse, dtype=float),
        "z_value": np.asarray(result.tvalues, dtype=float),
        "p_value_descriptive_only": np.asarray(
            result.pvalues,
            dtype=float,
        ),
    })
    fit_summary = {
        "converged": bool(result.converged),
        "iterations": int(
            result.fit_history.get("iteration", -1)
        ),
        "deviance": float(result.deviance),
        "pearson_chi2": float(result.pearson_chi2),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "maximum_absolute_coefficient": float(
            np.max(np.abs(np.asarray(result.params, dtype=float)))
        ),
        **metadata,
    }
    return propensity, coefficient_table, fit_summary


def weighted_mean_variance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0 or float(weights.sum()) <= 0:
        return float("nan"), float("nan")
    normalized = weights / weights.sum()
    mean = float(np.sum(normalized * values))
    variance = float(
        np.sum(normalized * (values - mean) ** 2)
    )
    return mean, variance


def smd(
    mean_1: float,
    variance_1: float,
    mean_0: float,
    variance_0: float,
) -> float:
    pooled = (variance_1 + variance_0) / 2.0
    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0 if mean_1 == mean_0 else float("inf")
    return (mean_1 - mean_0) / math.sqrt(pooled)


def balance_table(
    compact: pd.DataFrame,
    features: list[str],
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> pd.DataFrame:
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    treated = treatment == 1
    control = treatment == 0
    weight_treated = np.where(treated, 1.0 - propensity, 0.0)
    weight_control = np.where(control, propensity, 0.0)

    rows = []
    for feature in features:
        raw = pd.to_numeric(
            compact[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        median = float(np.nanmedian(raw))
        if not np.isfinite(median):
            median = 0.0
        values = np.where(np.isfinite(raw), raw, median)

        mean_t = float(np.mean(values[treated]))
        mean_c = float(np.mean(values[control]))
        var_t = float(np.var(values[treated]))
        var_c = float(np.var(values[control]))

        weighted_mean_t, weighted_var_t = weighted_mean_variance(
            values[treated],
            weight_treated[treated],
        )
        weighted_mean_c, weighted_var_c = weighted_mean_variance(
            values[control],
            weight_control[control],
        )

        unweighted = smd(mean_t, var_t, mean_c, var_c)
        weighted = smd(
            weighted_mean_t,
            weighted_var_t,
            weighted_mean_c,
            weighted_var_c,
        )
        rows.append({
            "feature": feature,
            "missing_fraction": float(np.mean(~np.isfinite(raw))),
            "unweighted_mean_treated": mean_t,
            "unweighted_mean_control": mean_c,
            "unweighted_smd": unweighted,
            "ato_weighted_mean_treated": weighted_mean_t,
            "ato_weighted_mean_control": weighted_mean_c,
            "ato_weighted_smd": weighted,
            "abs_ato_weighted_smd": abs(weighted),
        })
    return pd.DataFrame(rows).sort_values(
        ["abs_ato_weighted_smd", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def propensity_summary(
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> dict[str, float]:
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    treated = treatment == 1
    control = treatment == 0
    weight_treated = 1.0 - propensity[treated]
    weight_control = propensity[control]

    ess_treated = float(
        weight_treated.sum() ** 2
        / np.sum(weight_treated ** 2)
    )
    ess_control = float(
        weight_control.sum() ** 2
        / np.sum(weight_control ** 2)
    )
    return {
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_p05": float(np.quantile(propensity, 0.05)),
        "propensity_median": float(np.median(propensity)),
        "propensity_p95": float(np.quantile(propensity, 0.95)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(np.max(propensity)),
        "fraction_propensity_below_0_05": float(
            np.mean(propensity < 0.05)
        ),
        "fraction_propensity_above_0_95": float(
            np.mean(propensity > 0.95)
        ),
        "ato_ess_treated": ess_treated,
        "ato_ess_control": ess_control,
        "ato_ess_fraction_treated": float(
            ess_treated / treated.sum()
        ),
        "ato_ess_fraction_control": float(
            ess_control / control.sum()
        ),
    }
