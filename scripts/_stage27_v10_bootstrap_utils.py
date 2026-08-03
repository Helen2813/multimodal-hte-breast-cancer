from __future__ import annotations

import hashlib
import json
import math
import traceback
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
from _stage25c_v10_utils import (
    balance_table,
    fit_propensity,
    propensity_metrics,
)
from _stage26_v10_utils import stabilized_ato_components


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
        380,
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


def assemble_frame(
    root: Path,
    config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    cohort = read_csv(root / config["source"]["v10_cohort"])
    compact = read_csv(root / config["source"]["v10_compact"])
    features = compact_features(compact)
    frame = (
        cohort.drop(
            columns=[
                column
                for column in features
                if column in cohort.columns
            ],
            errors="ignore",
        )
        .merge(
            compact[["patient_id_normalized"] + features],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    return frame, features


def verify_hash_manifest(
    root: Path,
    manifest_path: Path,
) -> tuple[dict, pd.DataFrame]:
    manifest = load_json(manifest_path)
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
            "Locked-file integrity failed.\n"
            + dataframe_console(
                checks[checks["match"] != True],
                max_rows=200,
            )
        )
    return manifest, checks



def locked_bootstrap_settings(
    root: Path,
    config: dict,
) -> dict[str, Any]:
    spec = load_json(
        root / config["source"]["v10_estimator_spec"]
    )
    bootstrap = spec["bootstrap"]
    repetitions = int(bootstrap["repetitions"])
    base_seed = int(bootstrap["base_seed"])
    if repetitions != int(config["bootstrap"]["repetitions"]):
        raise RuntimeError(
            "Configured bootstrap repetitions do not match the locked "
            "Candidate V10 estimator specification."
        )
    return {
        "repetitions": repetitions,
        "base_seed": base_seed,
        "primary_interval": bootstrap["primary_interval"],
        "sensitivity_intervals": bootstrap[
            "sensitivity_intervals"
        ],
        "sampling": bootstrap["sampling"],
        "duplicate_fold_rule": bootstrap["duplicate_fold_rule"],
    }


def fit_partition_bootstrap(
    sample: pd.DataFrame,
    features: list[str],
    treatment: np.ndarray,
    event: np.ndarray,
    observed_time: np.ndarray,
    source_groups: np.ndarray,
    propensity: np.ndarray,
    partition_number: int,
    base_seed: int,
    config: dict,
) -> tuple[dict[str, Any], pd.DataFrame]:
    estimator = config["estimator"]
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
                    source_groups,
                    seed,
                    int(estimator["n_folds"]),
                )
            )
            G, starts, ends, censor_metrics = (
                crossfit_censor_survival(
                    sample,
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
                sample,
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
                "bootstrap_row",
                np.arange(len(sample), dtype=int),
            )
            patient.insert(
                0,
                "source_group",
                source_groups,
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
                "G_min_raw": float(censor_metrics["G_min"]),
                "G_p01_raw": float(censor_metrics["G_p01"]),
                "censor_log_loss": float(
                    censor_metrics["censor_log_loss"]
                ),
                "censor_brier": float(
                    censor_metrics["censor_brier"]
                ),
                "pseudo_mean": float(np.mean(pseudo)),
                "pseudo_sd": float(np.std(pseudo, ddof=1)),
                "pseudo_p99": float(np.quantile(pseudo, 0.99)),
                "pseudo_max": float(np.max(pseudo)),
                "fraction_mu0_outside_before_bounding": float(
                    np.mean(
                        (mu0_raw < 0.0)
                        | (
                            mu0_raw
                            > float(estimator["outcome_bound_high"])
                        )
                    )
                ),
                "fraction_mu1_outside_before_bounding": float(
                    np.mean(
                        (mu1_raw < 0.0)
                        | (
                            mu1_raw
                            > float(estimator["outcome_bound_high"])
                        )
                    )
                ),
            }
            return row, patient
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Partition {partition_number} failed after "
        f"{estimator['maximum_partition_retries']} retries: {last_error}"
    ) from last_error


def aggregate_partition_scores(
    scores: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    grouped = (
        scores.groupby(
            ["bootstrap_row", "source_group"],
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
        .sort_values("bootstrap_row")
        .reset_index(drop=True)
    )
    denominator = float(grouped["h"].sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Invalid aggregated overlap denominator.")
    theta = float(
        grouped["score_numerator"].sum() / denominator
    )
    mean_h = float(grouped["h"].mean())
    influence = (
        grouped["score_numerator"].to_numpy(dtype=float)
        - theta * grouped["h"].to_numpy(dtype=float)
    ) / mean_h
    grouped["influence"] = influence
    if_se = float(
        np.std(influence, ddof=1) / math.sqrt(len(influence))
    )
    summary = {
        "estimate_days": theta,
        "diagnostic_if_se_days": if_se,
        "diagnostic_if_ci_low_days": theta - 1.96 * if_se,
        "diagnostic_if_ci_high_days": theta + 1.96 * if_se,
        "ato_denominator": denominator,
        "plugin_component_days": float(
            grouped["plugin_numerator"].sum() / denominator
        ),
        "treated_residual_component_days": float(
            grouped[
                "treated_residual_numerator"
            ].sum() / denominator
        ),
        "control_residual_component_days": float(
            grouped[
                "control_residual_numerator"
            ].sum() / denominator
        ),
    }
    summary["total_residual_augmentation_days"] = (
        summary["treated_residual_component_days"]
        + summary["control_residual_component_days"]
    )
    return summary, grouped


def run_resample(
    frame: pd.DataFrame,
    features: list[str],
    indices: np.ndarray,
    repetition: int,
    config: dict,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    sample = frame.iloc[indices].reset_index(drop=True)
    source_groups = np.asarray(indices, dtype=int)
    treatment = pd.to_numeric(
        sample["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        sample["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()
    observed_time = pd.to_numeric(
        sample["analysis_time"],
        errors="coerce",
    ).to_numpy(dtype=float)

    stage25c_config = load_json(
        project_root() / config["source"]["stage25c_config"]
    )
    propensity, _, propensity_fit = fit_propensity(
        sample,
        treatment,
        features,
        stage25c_config,
    )
    balance = balance_table(
        sample,
        features,
        treatment,
        propensity,
    )
    pmetrics = propensity_metrics(
        treatment,
        propensity,
    )

    partition_rows = []
    patient_rows = []
    for partition_number, base_seed in enumerate(
        config["estimator"]["partition_base_seeds"],
        start=1,
    ):
        row, patient = fit_partition_bootstrap(
            sample,
            features,
            treatment,
            event,
            observed_time,
            source_groups,
            propensity,
            partition_number,
            int(base_seed),
            config,
        )
        partition_rows.append(row)
        patient_rows.append(patient)

    partitions = pd.DataFrame(partition_rows)
    scores = pd.concat(patient_rows, ignore_index=True)
    aggregate, _ = aggregate_partition_scores(scores)
    partition_effects = pd.to_numeric(
        partitions["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)

    result = {
        "repetition": int(repetition),
        "seed": (
            None if repetition == 0 else int(bootstrap_seed)
        ),
        "success": True,
        "error_type": "",
        "error_message": "",
        "n": len(sample),
        "unique_source_patients": int(
            pd.Series(source_groups).nunique()
        ),
        "treated": int(treatment.sum()),
        "control": int((1 - treatment).sum()),
        "events": int(event.sum()),
        "treated_events": int(
            ((treatment == 1) & (event == 1)).sum()
        ),
        "control_events": int(
            ((treatment == 0) & (event == 1)).sum()
        ),
        "estimate_days": aggregate["estimate_days"],
        "diagnostic_if_se_days": aggregate[
            "diagnostic_if_se_days"
        ],
        "plugin_component_days": aggregate[
            "plugin_component_days"
        ],
        "treated_residual_component_days": aggregate[
            "treated_residual_component_days"
        ],
        "control_residual_component_days": aggregate[
            "control_residual_component_days"
        ],
        "total_residual_augmentation_days": aggregate[
            "total_residual_augmentation_days"
        ],
        "partition_mean_effect_days": float(
            np.mean(partition_effects)
        ),
        "partition_sd_effect_days": float(
            np.std(partition_effects, ddof=1)
        ),
        "partition_mcse_effect_days": float(
            np.std(partition_effects, ddof=1)
            / math.sqrt(len(partition_effects))
        ),
        "partition_min_effect_days": float(
            np.min(partition_effects)
        ),
        "partition_max_effect_days": float(
            np.max(partition_effects)
        ),
        "partition_range_effect_days": float(
            np.max(partition_effects)
            - np.min(partition_effects)
        ),
        "minimum_G_min_raw": float(
            pd.to_numeric(
                partitions["G_min_raw"],
                errors="raise",
            ).min()
        ),
        "minimum_G_p01_raw": float(
            pd.to_numeric(
                partitions["G_p01_raw"],
                errors="raise",
            ).min()
        ),
        "maximum_pseudo_max": float(
            pd.to_numeric(
                partitions["pseudo_max"],
                errors="raise",
            ).max()
        ),
        "maximum_nuisance_retry": int(
            pd.to_numeric(
                partitions["nuisance_retry"],
                errors="raise",
            ).max()
        ),
        "propensity_converged": bool(
            propensity_fit["converged"]
        ),
        "maximum_absolute_propensity_coefficient": float(
            propensity_fit["maximum_absolute_coefficient"]
        ),
        "propensity_min": pmetrics["propensity_min"],
        "propensity_p01": pmetrics["propensity_p01"],
        "propensity_p99": pmetrics["propensity_p99"],
        "propensity_max": pmetrics["propensity_max"],
        "fraction_propensity_below_0_01": pmetrics[
            "fraction_propensity_below_0_01"
        ],
        "fraction_propensity_above_0_99": pmetrics[
            "fraction_propensity_above_0_99"
        ],
        "ato_ess_fraction_treated": pmetrics[
            "ato_ess_fraction_treated"
        ],
        "ato_ess_fraction_control": pmetrics[
            "ato_ess_fraction_control"
        ],
        "normalized_overlap_mass": pmetrics[
            "normalized_overlap_mass"
        ],
        "max_abs_ato_weighted_smd": float(
            balance["abs_ato_weighted_smd"].max()
        ),
    }
    return result


def append_checkpoint(
    row: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    if checkpoint_path.exists():
        frame = pd.read_csv(checkpoint_path, low_memory=False)
        frame = frame[
            pd.to_numeric(
                frame["repetition"],
                errors="coerce",
            )
            != int(row["repetition"])
        ]
        frame = pd.concat(
            [frame, pd.DataFrame([row])],
            ignore_index=True,
        )
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("repetition").reset_index(drop=True)
    write_csv(frame, checkpoint_path)


def error_row(
    repetition: int,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "repetition": int(repetition),
        "seed": int(seed),
        "success": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
