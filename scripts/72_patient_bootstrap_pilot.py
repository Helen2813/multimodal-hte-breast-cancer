#!/usr/bin/env python3
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from _stage12_utils import (
    assemble_landmark_data,
    crossfit_arm_outcomes,
    crossfit_censor_survival,
    crossfit_propensity,
    ipcw_rmst_pseudo,
)
from _stage17_utils import effect_and_patient_components
from _stage18_utils import (
    aggregate_partition_patient_scores,
    append_replace_repetition,
    checkpoint_identity,
    dataframe_console,
    ensure_stage18_dirs,
    load_stage18_config,
    make_grouped_bootstrap_folds,
    project_root,
    read_csv,
    validate_or_create_checkpoint_identity,
    write_csv,
)


def empty_or_read(path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def bootstrap_payload() -> dict[str, object]:
    frame, features, _, metadata = assemble_landmark_data()
    frame = frame.copy().reset_index(drop=True)
    return {
        "frame": frame,
        "features": list(features),
        "a": pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy(),
        "event": pd.to_numeric(frame["analysis_event"], errors="raise").astype(int).to_numpy(),
        "observed_time": pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float),
        "metadata": metadata,
    }


def fit_partition(
    frame: pd.DataFrame,
    features: list[str],
    a: np.ndarray,
    event: np.ndarray,
    observed_time: np.ndarray,
    original_groups: np.ndarray,
    partition_number: int,
    nominal_seed: int,
    cfg: dict,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    bcfg = cfg["bootstrap_pilot"]
    n_folds = int(bcfg["n_folds"])
    g_min = float(bcfg["primary_g_min"])
    horizon = float(bcfg["horizon_days"])
    interval = float(bcfg["interval_days"])
    maximum_retries = int(bcfg["maximum_partition_retries"])

    last_error: Exception | None = None
    for retry in range(maximum_retries):
        seed = nominal_seed + retry * 100_000
        try:
            fold, stratification, fold_retry = make_grouped_bootstrap_folds(
                a, event, original_groups, seed, n_folds
            )
            e = crossfit_propensity(
                frame, features, fold, "analysis_treatment", seed + 10
            )
            G, starts, ends, censor_metrics = crossfit_censor_survival(
                frame, features, fold, horizon, interval, seed + 100
            )
            y = ipcw_rmst_pseudo(
                observed_time, G, starts, ends, horizon, g_min
            )
            mu0_raw, mu1_raw = crossfit_arm_outcomes(
                frame, features, y, fold, seed + 200
            )
            mu0 = np.clip(mu0_raw, 0.0, horizon)
            mu1 = np.clip(mu1_raw, 0.0, horizon)
            summary, patient = effect_and_patient_components(y, a, e, mu0, mu1)
            patient = patient[["h", "score_numerator"]].copy()
            patient.insert(0, "original_patient_group", original_groups)
            patient.insert(0, "bootstrap_row_index", np.arange(len(frame)))
            patient.insert(0, "partition", partition_number)
            row = {
                "partition": partition_number,
                "nominal_seed": nominal_seed,
                "seed_used": seed,
                "nuisance_retry": retry,
                "fold_retry": fold_retry,
                "fold_stratification": stratification,
                **summary,
                "propensity_min": float(np.min(e)),
                "propensity_p01": float(np.quantile(e, 0.01)),
                "propensity_p99": float(np.quantile(e, 0.99)),
                "propensity_max": float(np.max(e)),
                "censor_log_loss": float(censor_metrics["censor_log_loss"]),
                "censor_brier": float(censor_metrics["censor_brier"]),
                "G_min_raw": float(censor_metrics["G_min"]),
                "G_p01_raw": float(censor_metrics["G_p01"]),
                "pseudo_mean": float(np.mean(y)),
                "pseudo_sd": float(np.std(y, ddof=1)),
                "pseudo_p99": float(np.quantile(y, 0.99)),
                "pseudo_max": float(np.max(y)),
                "fraction_mu0_outside_before_bounding": float(
                    np.mean((mu0_raw < 0.0) | (mu0_raw > horizon))
                ),
                "fraction_mu1_outside_before_bounding": float(
                    np.mean((mu1_raw < 0.0) | (mu1_raw > horizon))
                ),
            }
            return row, patient
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Partition {partition_number} failed after {maximum_retries} retries: {last_error}"
    ) from last_error


def fit_bootstrap_repetition(
    payload: dict[str, object],
    repetition: int,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bcfg = cfg["bootstrap_pilot"]
    n = len(payload["frame"])
    bootstrap_seed = int(bcfg["base_seed"]) + repetition
    rng = np.random.default_rng(bootstrap_seed)
    sampled_original_rows = rng.integers(0, n, size=n, endpoint=False)

    frame = payload["frame"].iloc[sampled_original_rows].copy().reset_index(drop=True)
    features = list(payload["features"])
    a = np.asarray(payload["a"], dtype=int)[sampled_original_rows]
    event = np.asarray(payload["event"], dtype=int)[sampled_original_rows]
    observed_time = np.asarray(payload["observed_time"], dtype=float)[sampled_original_rows]
    original_groups = sampled_original_rows.astype(int)

    multiplicity = pd.Series(original_groups).value_counts()
    partition_rows: list[dict] = []
    patient_scores: list[pd.DataFrame] = []
    seeds = [int(x) for x in bcfg["crossfit_partition_seeds"]]
    if len(seeds) != int(bcfg["n_crossfit_partitions"]):
        raise ValueError("Cross-fit partition seed count does not match configuration.")

    for partition_number, base_seed in enumerate(seeds, start=1):
        nominal_seed = base_seed + repetition * 1_000_000
        row, patient = fit_partition(
            frame,
            features,
            a,
            event,
            observed_time,
            original_groups,
            partition_number,
            nominal_seed,
            cfg,
        )
        row["bootstrap_repetition"] = repetition
        row["bootstrap_seed"] = bootstrap_seed
        partition_rows.append(row)
        patient.insert(0, "bootstrap_repetition", repetition)
        patient_scores.append(patient)

    partition_df = pd.DataFrame(partition_rows)
    score_df = pd.concat(patient_scores, ignore_index=True)
    aggregated = aggregate_partition_patient_scores(score_df)
    estimates = partition_df["estimate_days"].to_numpy(float)

    summary = pd.DataFrame(
        [
            {
                "bootstrap_repetition": repetition,
                "bootstrap_seed": bootstrap_seed,
                "n": n,
                "unique_original_patients": int(multiplicity.size),
                "unique_original_patient_fraction": float(multiplicity.size / n),
                "maximum_patient_multiplicity": int(multiplicity.max()),
                "treated": int(a.sum()),
                "control": int((1 - a).sum()),
                "events": int(event.sum()),
                "treated_events": int(np.sum((a == 1) & (event == 1))),
                "control_events": int(np.sum((a == 0) & (event == 1))),
                "aggregated_effect_days": float(aggregated["estimate_days"]),
                "aggregated_if_se_days": float(aggregated["if_se_days"]),
                "aggregated_if_ci_low_days": float(aggregated["if_ci_low_days"]),
                "aggregated_if_ci_high_days": float(aggregated["if_ci_high_days"]),
                "partition_mean_effect_days": float(np.mean(estimates)),
                "partition_median_effect_days": float(np.median(estimates)),
                "partition_sd_effect_days": float(np.std(estimates, ddof=1)),
                "partition_min_effect_days": float(np.min(estimates)),
                "partition_max_effect_days": float(np.max(estimates)),
                "partition_range_effect_days": float(np.max(estimates) - np.min(estimates)),
                "mean_censor_log_loss": float(partition_df["censor_log_loss"].mean()),
                "mean_censor_brier": float(partition_df["censor_brier"].mean()),
                "minimum_G_min_raw": float(partition_df["G_min_raw"].min()),
                "minimum_G_p01_raw": float(partition_df["G_p01_raw"].min()),
                "minimum_propensity": float(partition_df["propensity_min"].min()),
                "minimum_propensity_p01": float(partition_df["propensity_p01"].min()),
                "maximum_propensity_p99": float(partition_df["propensity_p99"].max()),
                "maximum_propensity": float(partition_df["propensity_max"].max()),
                "median_pseudo_p99": float(partition_df["pseudo_p99"].median()),
                "maximum_pseudo_max": float(partition_df["pseudo_max"].max()),
                "maximum_nuisance_retry": int(partition_df["nuisance_retry"].max()),
            }
        ]
    )
    return summary, partition_df


def main() -> int:
    root = project_root()
    ensure_stage18_dirs(root)
    cfg = load_stage18_config(root)
    bcfg = cfg["bootstrap_pilot"]
    tables = root / "results/tables"
    local = root / "data/derived/stage18"

    repetition_path = tables / "72_bootstrap_pilot_repetitions_checkpoint.csv"
    partition_path = tables / "72_bootstrap_pilot_partitions_checkpoint.csv"
    errors_path = tables / "72_bootstrap_pilot_errors.csv"
    identity_path = local / "72_bootstrap_checkpoint_identity.json"

    validate_or_create_checkpoint_identity(identity_path, checkpoint_identity(root, cfg))
    repetitions = empty_or_read(repetition_path)
    partitions = empty_or_read(partition_path)
    errors = empty_or_read(
        errors_path,
        ["bootstrap_repetition", "bootstrap_seed", "error_type", "error_message", "traceback"],
    )
    completed = (
        set(pd.to_numeric(repetitions["bootstrap_repetition"], errors="coerce").dropna().astype(int))
        if not repetitions.empty and "bootstrap_repetition" in repetitions.columns
        else set()
    )

    payload = bootstrap_payload()
    metadata = payload["metadata"]
    target = int(bcfg["n_repetitions"])

    print("=" * 124)
    print("STAGE 72 - PATIENT-LEVEL BOOTSTRAP PILOT WITH FULL NUISANCE REFITTING")
    print("=" * 124)
    print(f"Original landmark cohort: n={metadata['n']}; treated={metadata['treated']}; control={metadata['control']}; events={metadata['events']}")
    print(f"Target bootstrap repetitions: {target}")
    print(f"Already complete: {len(completed)} -> {sorted(completed)}")
    print(f"Cross-fit partitions inside each bootstrap: {bcfg['n_crossfit_partitions']}")
    print("Sampling: ordinary patient bootstrap with replacement.")
    print("Leakage protection: all copies of the same original patient stay in the same nuisance fold.")
    print("Primary nuisance pipeline: refitted Stage 30 propensity, refitted censoring model, bounded arm-specific ridge outcomes.")
    print("This is a pilot. The full publication bootstrap is NOT started.")

    for repetition in range(1, target + 1):
        if repetition in completed:
            print(f"Bootstrap repetition {repetition:02d}/{target} already complete; skipping.")
            continue
        print("-" * 124)
        print(f"RUNNING BOOTSTRAP REPETITION {repetition:02d}/{target}")
        try:
            summary, partition_df = fit_bootstrap_repetition(payload, repetition, cfg)
            repetitions = append_replace_repetition(repetitions, summary, repetition)
            partitions = append_replace_repetition(partitions, partition_df, repetition)
            if not errors.empty:
                keep = pd.to_numeric(errors["bootstrap_repetition"], errors="coerce") != repetition
                errors = errors.loc[keep].copy()
            write_csv(repetitions.sort_values("bootstrap_repetition"), repetition_path)
            write_csv(partitions.sort_values(["bootstrap_repetition", "partition"]), partition_path)
            write_csv(errors, errors_path)
            print("Bootstrap repetition summary")
            print(dataframe_console(summary))
            print("Partition estimates")
            print(dataframe_console(partition_df[[
                "partition", "estimate_days", "if_se_days", "direct_ato_ipw_effect_days",
                "pseudo_p99", "pseudo_max", "G_min_raw", "propensity_p01", "propensity_p99",
                "nuisance_retry"
            ]]))
            print(f"Checkpoint saved: {len(repetitions)}/{target} successful repetitions.")
        except Exception as exc:
            error_row = pd.DataFrame([
                {
                    "bootstrap_repetition": repetition,
                    "bootstrap_seed": int(bcfg["base_seed"]) + repetition,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ])
            errors = append_replace_repetition(errors, error_row, repetition)
            write_csv(errors, errors_path)
            print("Bootstrap repetition failed")
            print(dataframe_console(error_row[["bootstrap_repetition", "bootstrap_seed", "error_type", "error_message"]]))

    print("=" * 124)
    print("STAGE 72 PILOT CHECKPOINT SUMMARY")
    print("=" * 124)
    if repetitions.empty:
        print("No successful bootstrap repetitions.")
    else:
        print(dataframe_console(repetitions.sort_values("bootstrap_repetition")))
    print("\nFiles")
    for path in (repetition_path, partition_path, errors_path):
        print(f"- {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
