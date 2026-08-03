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


def prepare_design(
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
    active = [
        feature
        for feature in features
        if np.isfinite(standard_deviations[feature])
        and float(standard_deviations[feature]) > 0
    ]
    standardized = (
        imputed[active] - means[active]
    ) / standard_deviations[active]
    design = sm.add_constant(
        standardized,
        has_constant="add",
    )
    metadata = {
        "features": features,
        "active_features": active,
        "zero_variance_features": [
            feature for feature in features if feature not in active
        ],
        "medians": {key: float(value) for key, value in medians.items()},
        "means": {key: float(value) for key, value in means.items()},
        "standard_deviations": {
            key: (
                float(value) if np.isfinite(value) else None
            )
            for key, value in standard_deviations.items()
        },
    }
    return design, metadata


def fit_propensity(
    compact: pd.DataFrame,
    treatment: np.ndarray,
    features: list[str],
    config: dict,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    design, metadata = prepare_design(compact, features)
    settings = config["propensity"]
    model = sm.GLM(
        np.asarray(treatment, dtype=int),
        design,
        family=sm.families.Binomial(),
    )
    result = model.fit(
        maxiter=int(settings["maximum_iterations"]),
        tol=float(settings["tolerance"]),
        disp=0,
    )
    propensity = np.asarray(
        result.predict(design),
        dtype=float,
    )
    coefficients = pd.DataFrame({
        "term": design.columns.astype(str),
        "coefficient": np.asarray(result.params, dtype=float),
        "standard_error": np.asarray(result.bse, dtype=float),
    })
    summary = {
        "converged": bool(result.converged),
        "iterations": int(result.fit_history.get("iteration", -1)),
        "maximum_absolute_coefficient": float(
            np.max(np.abs(np.asarray(result.params, dtype=float)))
        ),
        **metadata,
    }
    return propensity, coefficients, summary


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
    variance = float(np.sum(normalized * (values - mean) ** 2))
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
    weight_treated = 1.0 - propensity[treated]
    weight_control = propensity[control]

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
            weight_treated,
        )
        weighted_mean_c, weighted_var_c = weighted_mean_variance(
            values[control],
            weight_control,
        )

        raw_smd = smd(mean_t, var_t, mean_c, var_c)
        weighted_smd = smd(
            weighted_mean_t,
            weighted_var_t,
            weighted_mean_c,
            weighted_var_c,
        )
        rows.append({
            "feature": feature,
            "missing_fraction": float(np.mean(~np.isfinite(raw))),
            "unweighted_smd": raw_smd,
            "ato_weighted_smd": weighted_smd,
            "abs_ato_weighted_smd": abs(weighted_smd),
        })

    return pd.DataFrame(rows).sort_values(
        ["abs_ato_weighted_smd", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def propensity_metrics(
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> dict[str, float]:
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    treated = treatment == 1
    control = treatment == 0

    w1 = 1.0 - propensity[treated]
    w0 = propensity[control]
    ess1 = float(w1.sum() ** 2 / np.sum(w1 ** 2))
    ess0 = float(w0.sum() ** 2 / np.sum(w0 ** 2))
    h = propensity * (1.0 - propensity)

    return {
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_p05": float(np.quantile(propensity, 0.05)),
        "propensity_median": float(np.median(propensity)),
        "propensity_p95": float(np.quantile(propensity, 0.95)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(np.max(propensity)),
        "fraction_propensity_below_0_01": float(
            np.mean(propensity < 0.01)
        ),
        "fraction_propensity_above_0_99": float(
            np.mean(propensity > 0.99)
        ),
        "ato_ess_treated": ess1,
        "ato_ess_control": ess0,
        "ato_ess_fraction_treated": float(
            ess1 / treated.sum()
        ),
        "ato_ess_fraction_control": float(
            ess0 / control.sum()
        ),
        "overlap_mass": float(np.sum(h)),
        "normalized_overlap_mass": float(
            np.mean(h) / 0.25
        ),
    }


def bootstrap_propensity_feasibility(
    compact: pd.DataFrame,
    treatment: np.ndarray,
    features: list[str],
    config: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["propensity"]
    gates = config["gates"]
    repetitions = int(
        settings["bootstrap_repetitions_for_feasibility"]
    )
    base_seed = int(settings["bootstrap_base_seed"])
    n = len(compact)

    rows = []
    for repetition in range(1, repetitions + 1):
        rng = np.random.default_rng(base_seed + repetition)
        indices = rng.integers(0, n, n)
        sample_compact = compact.iloc[indices].reset_index(drop=True)
        sample_treatment = np.asarray(treatment, dtype=int)[indices]

        row = {
            "repetition": repetition,
            "seed": base_seed + repetition,
            "success": False,
            "error": "",
        }
        try:
            propensity, _, fit_summary = fit_propensity(
                sample_compact,
                sample_treatment,
                features,
                config,
            )
            balance = balance_table(
                sample_compact,
                features,
                sample_treatment,
                propensity,
            )
            metrics = propensity_metrics(
                sample_treatment,
                propensity,
            )
            max_smd = float(
                balance["abs_ato_weighted_smd"].max()
            )
            tail_pass = (
                metrics["fraction_propensity_below_0_01"]
                <= float(gates[
                    "maximum_fraction_propensity_below_0_01"
                ])
                and metrics["fraction_propensity_above_0_99"]
                <= float(gates[
                    "maximum_fraction_propensity_above_0_99"
                ])
            )
            balance_pass = (
                max_smd
                <= float(gates[
                    "maximum_abs_ato_weighted_smd"
                ])
            )
            row.update({
                "success": bool(fit_summary["converged"]),
                "maximum_absolute_coefficient": fit_summary[
                    "maximum_absolute_coefficient"
                ],
                "max_abs_ato_weighted_smd": max_smd,
                "tail_gate_pass": tail_pass,
                "balance_gate_pass": balance_pass,
                **metrics,
            })
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = {
        "repetitions": repetitions,
        "successful_fits": int(frame["success"].fillna(False).sum()),
        "success_fraction": float(
            frame["success"].fillna(False).mean()
        ),
        "tail_gate_failure_fraction": float(
            1.0 - frame.loc[
                frame["success"] == True,
                "tail_gate_pass",
            ].fillna(False).mean()
        ) if bool((frame["success"] == True).any()) else 1.0,
        "balance_gate_failure_fraction": float(
            1.0 - frame.loc[
                frame["success"] == True,
                "balance_gate_pass",
            ].fillna(False).mean()
        ) if bool((frame["success"] == True).any()) else 1.0,
        "maximum_abs_coefficient_over_successful_fits": float(
            pd.to_numeric(
                frame.loc[
                    frame["success"] == True,
                    "maximum_absolute_coefficient",
                ],
                errors="coerce",
            ).max()
        ) if bool((frame["success"] == True).any()) else None,
        "maximum_abs_smd_over_successful_fits": float(
            pd.to_numeric(
                frame.loc[
                    frame["success"] == True,
                    "max_abs_ato_weighted_smd",
                ],
                errors="coerce",
            ).max()
        ) if bool((frame["success"] == True).any()) else None,
    }
    return frame, summary
