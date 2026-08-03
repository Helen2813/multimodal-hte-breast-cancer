from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage12_utils import (
    crossfit_arm_outcomes,
    crossfit_censor_survival,
    ipcw_rmst_pseudo,
)
from _stage18_utils import make_grouped_bootstrap_folds
from _stage25c_v10_utils import fit_propensity


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


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
        120,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        return view.to_string(index=False)


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


def verify_manifest(root: Path, config: dict) -> tuple[dict, pd.DataFrame]:
    manifest_path = root / config["source"]["v10_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = load_json(manifest_path)
    expected_id = config["expected"]["protocol_id"]
    if manifest.get("protocol_id") != expected_id:
        raise RuntimeError(
            f"Unexpected V10 protocol ID: {manifest.get('protocol_id')} != {expected_id}"
        )
    if bool(manifest.get("candidate_v10_effect_computed")):
        raise RuntimeError(
            "The locked V10 manifest says the effect was already computed."
        )

    rows = []
    for item in manifest["locked_files"]:
        path = root / item["path"]
        observed = sha256_file(path) if path.exists() else ""
        rows.append({
            "path": item["path"],
            "exists": path.exists(),
            "expected_sha256": item["sha256"],
            "observed_sha256": observed,
            "match": observed == item["sha256"],
        })

    checks = pd.DataFrame(rows)
    if checks.empty or not bool(checks["match"].all()):
        raise RuntimeError(
            "Candidate V10 lock integrity failed.\n"
            + dataframe_console(
                checks[checks["match"] != True],
                max_rows=200,
            )
        )
    return manifest, checks


def assemble_v10_frame(
    root: Path,
    config: dict,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    cohort_path = root / config["source"]["v10_cohort"]
    compact_path = root / config["source"]["v10_compact"]
    cohort = read_csv(cohort_path)
    compact = read_csv(compact_path)

    features = compact_features(compact)
    cohort_side = cohort.drop(
        columns=[
            column for column in features
            if column in cohort.columns
        ],
        errors="ignore",
    )
    frame = cohort_side.merge(
        compact[["patient_id_normalized"] + features],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)

    metadata = {
        "cohort_sha256": sha256_file(cohort_path),
        "compact_sha256": sha256_file(compact_path),
        "n": len(frame),
        "treated": int(
            pd.to_numeric(
                frame["analysis_treatment"],
                errors="raise",
            ).sum()
        ),
        "control": int(
            (
                1 - pd.to_numeric(
                    frame["analysis_treatment"],
                    errors="raise",
                )
            ).sum()
        ),
        "events": int(
            pd.to_numeric(
                frame["analysis_event"],
                errors="raise",
            ).sum()
        ),
        "features": len(features),
    }
    return frame, features, metadata


def stabilized_ato_components(
    y: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)

    h = propensity * (1.0 - propensity)
    denominator = float(np.sum(h))
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Invalid overlap-score denominator.")

    plugin = h * (mu1 - mu0)
    treated_residual = (
        treatment
        * (1.0 - propensity)
        * (y - mu1)
    )
    control_residual = (
        -(1 - treatment)
        * propensity
        * (y - mu0)
    )
    score_numerator = (
        plugin + treated_residual + control_residual
    )
    theta = float(
        np.sum(score_numerator) / denominator
    )
    mean_h = float(np.mean(h))
    influence = (
        score_numerator - theta * h
    ) / mean_h
    if_se = float(
        np.std(influence, ddof=1)
        / math.sqrt(len(influence))
    )

    treated_weights = (
        treatment * (1.0 - propensity)
    )
    control_weights = (
        (1 - treatment) * propensity
    )
    direct_treated = float(
        np.sum(treated_weights * y)
        / np.sum(treated_weights)
    )
    direct_control = float(
        np.sum(control_weights * y)
        / np.sum(control_weights)
    )

    patient = pd.DataFrame({
        "h": h,
        "plugin_numerator": plugin,
        "treated_residual_numerator": treated_residual,
        "control_residual_numerator": control_residual,
        "score_numerator": score_numerator,
        "influence": influence,
    })
    summary = {
        "estimate_days": theta,
        "if_se_days": if_se,
        "if_ci_low_days": theta - 1.96 * if_se,
        "if_ci_high_days": theta + 1.96 * if_se,
        "ato_denominator": denominator,
        "plugin_component_days": float(
            np.sum(plugin) / denominator
        ),
        "treated_residual_component_days": float(
            np.sum(treated_residual) / denominator
        ),
        "control_residual_component_days": float(
            np.sum(control_residual) / denominator
        ),
        "total_residual_augmentation_days": float(
            np.sum(
                treated_residual + control_residual
            ) / denominator
        ),
        "direct_ato_ipw_treated_mean_days": direct_treated,
        "direct_ato_ipw_control_mean_days": direct_control,
        "direct_ato_ipw_effect_days": (
            direct_treated - direct_control
        ),
        "aipw_minus_direct_ato_ipw_days": (
            theta - (direct_treated - direct_control)
        ),
    }
    return summary, patient


def fit_partition(
    frame: pd.DataFrame,
    features: list[str],
    treatment: np.ndarray,
    event: np.ndarray,
    observed_time: np.ndarray,
    propensity: np.ndarray,
    partition_number: int,
    base_seed: int,
    config: dict,
) -> tuple[dict[str, Any], pd.DataFrame]:
    estimator = config["estimator"]
    groups = np.arange(len(frame), dtype=int)
    last_error: Exception | None = None

    for retry in range(
        int(estimator["maximum_partition_retries"])
    ):
        seed = int(base_seed) + retry * 100_000
        try:
            fold, stratification, fold_retry = (
                make_grouped_bootstrap_folds(
                    treatment,
                    event,
                    groups,
                    seed,
                    int(estimator["n_folds"]),
                )
            )
            G, starts, ends, censor_metrics = (
                crossfit_censor_survival(
                    frame,
                    features,
                    fold,
                    float(estimator["horizon_days"]),
                    float(estimator["interval_days"]),
                    seed + 100,
                )
            )
            pseudo = ipcw_rmst_pseudo(
                observed_time,
                G,
                starts,
                ends,
                float(estimator["horizon_days"]),
                float(estimator["censoring_g_min"]),
            )
            mu0_raw, mu1_raw = crossfit_arm_outcomes(
                frame,
                features,
                pseudo,
                fold,
                seed + 200,
            )
            mu0 = np.clip(
                mu0_raw,
                float(estimator["outcome_bound_low"]),
                float(estimator["outcome_bound_high"]),
            )
            mu1 = np.clip(
                mu1_raw,
                float(estimator["outcome_bound_low"]),
                float(estimator["outcome_bound_high"]),
            )
            summary, patient = stabilized_ato_components(
                pseudo,
                treatment,
                propensity,
                mu0,
                mu1,
            )
            patient.insert(
                0,
                "patient_id_normalized",
                frame["patient_id_normalized"].astype(str).to_numpy(),
            )
            patient.insert(
                0,
                "row_index",
                np.arange(len(frame), dtype=int),
            )
            patient.insert(0, "partition", partition_number)

            row = {
                "partition": partition_number,
                "base_seed": int(base_seed),
                "seed_used": int(seed),
                "nuisance_retry": int(retry),
                "fold_retry": int(fold_retry),
                "fold_stratification": str(stratification),
                **summary,
                "censor_log_loss": float(
                    censor_metrics["censor_log_loss"]
                ),
                "censor_brier": float(
                    censor_metrics["censor_brier"]
                ),
                "G_min_raw": float(
                    censor_metrics["G_min"]
                ),
                "G_p01_raw": float(
                    censor_metrics["G_p01"]
                ),
                "pseudo_mean": float(np.mean(pseudo)),
                "pseudo_sd": float(
                    np.std(pseudo, ddof=1)
                ),
                "pseudo_p99": float(
                    np.quantile(pseudo, 0.99)
                ),
                "pseudo_max": float(np.max(pseudo)),
                "fraction_mu0_outside_before_bounding": float(
                    np.mean(
                        (mu0_raw < 0.0)
                        | (
                            mu0_raw
                            > float(
                                estimator[
                                    "outcome_bound_high"
                                ]
                            )
                        )
                    )
                ),
                "fraction_mu1_outside_before_bounding": float(
                    np.mean(
                        (mu1_raw < 0.0)
                        | (
                            mu1_raw
                            > float(
                                estimator[
                                    "outcome_bound_high"
                                ]
                            )
                        )
                    )
                ),
            }
            return row, patient

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Partition {partition_number} failed after "
        f"{estimator['maximum_partition_retries']} retries: "
        f"{last_error}"
    ) from last_error


def aggregate_partition_scores(
    scores: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    grouped = (
        scores.groupby(
            ["row_index", "patient_id_normalized"],
            as_index=False,
        )
        .agg(
            h=("h", "mean"),
            plugin_numerator=("plugin_numerator", "mean"),
            treated_residual_numerator=(
                "treated_residual_numerator",
                "mean",
            ),
            control_residual_numerator=(
                "control_residual_numerator",
                "mean",
            ),
            score_numerator=("score_numerator", "mean"),
        )
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    denominator = float(grouped["h"].sum())
    theta = float(
        grouped["score_numerator"].sum()
        / denominator
    )
    mean_h = float(grouped["h"].mean())
    influence = (
        grouped["score_numerator"].to_numpy(dtype=float)
        - theta * grouped["h"].to_numpy(dtype=float)
    ) / mean_h
    grouped["influence"] = influence
    grouped["absolute_influence"] = np.abs(influence)

    if_se = float(
        np.std(influence, ddof=1)
        / math.sqrt(len(grouped))
    )
    summary = {
        "estimate_days": theta,
        "if_se_days": if_se,
        "if_ci_low_days": theta - 1.96 * if_se,
        "if_ci_high_days": theta + 1.96 * if_se,
        "ato_denominator": denominator,
        "plugin_component_days": float(
            grouped["plugin_numerator"].sum()
            / denominator
        ),
        "treated_residual_component_days": float(
            grouped[
                "treated_residual_numerator"
            ].sum()
            / denominator
        ),
        "control_residual_component_days": float(
            grouped[
                "control_residual_numerator"
            ].sum()
            / denominator
        ),
    }
    summary["total_residual_augmentation_days"] = (
        summary["treated_residual_component_days"]
        + summary["control_residual_component_days"]
    )
    return summary, grouped
