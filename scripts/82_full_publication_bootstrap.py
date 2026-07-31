#!/usr/bin/env python3
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from _stage21_utils import (
    aggregate_score_arrays,
    append_partition_scores,
    append_replace_repetition,
    bootstrap_payload,
    checkpoint_identity,
    dataframe_console,
    empty_or_read,
    ensure_stage21_dirs,
    fit_partition,
    load_partition_score_file,
    load_stage21_config,
    project_root,
    recreate_bootstrap_sample,
    validate_or_create_checkpoint_identity,
    verify_locked_files,
    write_csv,
)


def repetition_score_path(root: Path, repetition: int) -> Path:
    return root / f"data/derived/stage21/partition_scores/rep_{repetition:03d}_scores_LOCAL_ONLY.npz"


def main() -> int:
    root = project_root()
    ensure_stage21_dirs(root)
    cfg = load_stage21_config(root)
    bcfg = cfg["full_bootstrap"]

    lock_check = verify_locked_files(root)
    identity_path = root / "data/derived/stage21/82_publication_bootstrap_checkpoint_identity.json"
    validate_or_create_checkpoint_identity(identity_path, checkpoint_identity(root, cfg))

    tables = root / "results/tables"
    repetition_path = tables / "82_publication_bootstrap_repetitions_checkpoint.csv"
    partition_path = tables / "82_publication_bootstrap_partitions_checkpoint.csv"
    errors_path = tables / "82_publication_bootstrap_errors.csv"

    repetitions = empty_or_read(repetition_path)
    partitions = empty_or_read(partition_path)
    errors = empty_or_read(
        errors_path,
        ["bootstrap_repetition", "partition", "bootstrap_seed", "error_type", "error_message", "traceback"],
    )
    completed_repetitions = (
        set(pd.to_numeric(repetitions["bootstrap_repetition"], errors="coerce").dropna().astype(int))
        if not repetitions.empty and "bootstrap_repetition" in repetitions.columns
        else set()
    )

    payload = bootstrap_payload()
    metadata = payload["metadata"]
    target = int(bcfg["n_repetitions"])
    partition_seeds = [int(x) for x in bcfg["partition_base_seeds"]]
    expected_partitions = list(range(1, len(partition_seeds) + 1))
    if len(partition_seeds) != int(bcfg["n_crossfit_partitions"]):
        raise RuntimeError("Locked partition seed count differs from n_crossfit_partitions.")

    print("=" * 128)
    print("STAGE 82 - LOCKED 300-REPETITION PATIENT PUBLICATION BOOTSTRAP")
    print("=" * 128)
    print(f"Protocol lock integrity: PASS for {len(lock_check)} files")
    print(f"Original cohort: n={metadata['n']}; treated={metadata['treated']}; control={metadata['control']}; events={metadata['events']}")
    print(f"Target bootstrap repetitions: {target}")
    print(f"Already completed repetitions: {len(completed_repetitions)}")
    print(f"Nuisance partitions per bootstrap: {len(partition_seeds)}")
    print(f"Locked partition seeds: {partition_seeds}")
    print("All copies of the same original patient remain in one nuisance fold.")
    print("Each nuisance partition is checkpointed. No estimator setting may be changed during this run.")

    for repetition in range(1, target + 1):
        if repetition in completed_repetitions:
            print(f"Bootstrap repetition {repetition:03d}/{target} already complete; skipping.")
            continue

        print("-" * 128)
        print(f"BOOTSTRAP REPETITION {repetition:03d}/{target}")
        sample = recreate_bootstrap_sample(payload, repetition, cfg)
        n = len(sample["frame"])
        score_path = repetition_score_path(root, repetition)
        score_partitions, _, _ = load_partition_score_file(score_path, n)
        score_completed = set(score_partitions.tolist())

        csv_completed: set[int] = set()
        if not partitions.empty and {"bootstrap_repetition", "partition"}.issubset(partitions.columns):
            mask = pd.to_numeric(partitions["bootstrap_repetition"], errors="coerce") == repetition
            csv_completed = set(
                pd.to_numeric(partitions.loc[mask, "partition"], errors="coerce").dropna().astype(int)
            )
        completed_partitions = csv_completed & score_completed
        if completed_partitions:
            print(f"Resuming repetition with completed partitions: {sorted(completed_partitions)}")

        repetition_failed = False
        for partition_number, base_seed in enumerate(partition_seeds, start=1):
            if partition_number in completed_partitions:
                print(f"rep={repetition:03d} partition={partition_number:02d} already complete; skipping")
                continue
            try:
                row, score_numerator, h = fit_partition(
                    sample, repetition, partition_number, base_seed, cfg
                )
                new_row = pd.DataFrame([row])
                if not partitions.empty:
                    keep = ~(
                        (pd.to_numeric(partitions["bootstrap_repetition"], errors="coerce") == repetition)
                        & (pd.to_numeric(partitions["partition"], errors="coerce") == partition_number)
                    )
                    partitions = partitions.loc[keep].copy()
                partitions = pd.concat([partitions, new_row], ignore_index=True)
                append_partition_scores(score_path, partition_number, score_numerator, h)
                if not errors.empty:
                    keep_err = ~(
                        (pd.to_numeric(errors["bootstrap_repetition"], errors="coerce") == repetition)
                        & (pd.to_numeric(errors["partition"], errors="coerce") == partition_number)
                    )
                    errors = errors.loc[keep_err].copy()
                write_csv(partitions.sort_values(["bootstrap_repetition", "partition"]), partition_path)
                write_csv(errors, errors_path)
                print(
                    f"rep={repetition:03d} partition={partition_number:02d} "
                    f"effect={row['estimate_days']:.6f} IF_SE={row['if_se_days']:.6f} "
                    f"G_min={row['G_min_raw']:.6f} prop_p01={row['propensity_p01']:.6f} "
                    f"prop_p99={row['propensity_p99']:.6f} pseudo_p99={row['pseudo_p99']:.6f} "
                    f"pseudo_max={row['pseudo_max']:.6f} retry={row['nuisance_retry']}"
                )
            except Exception as exc:
                error_row = pd.DataFrame([{
                    "bootstrap_repetition": repetition,
                    "partition": partition_number,
                    "bootstrap_seed": int(sample["bootstrap_seed"]),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }])
                if not errors.empty:
                    keep_err = ~(
                        (pd.to_numeric(errors["bootstrap_repetition"], errors="coerce") == repetition)
                        & (pd.to_numeric(errors["partition"], errors="coerce") == partition_number)
                    )
                    errors = errors.loc[keep_err].copy()
                errors = pd.concat([errors, error_row], ignore_index=True)
                write_csv(errors, errors_path)
                repetition_failed = True
                print("Partition failed")
                print(dataframe_console(error_row[["bootstrap_repetition", "partition", "error_type", "error_message"]]))
                break

        if repetition_failed:
            print(f"Repetition {repetition:03d} is incomplete and will be retried on the next runner invocation.")
            continue

        score_partitions, score_numerators, h_values = load_partition_score_file(score_path, n)
        aggregate = aggregate_score_arrays(
            score_partitions, score_numerators, h_values, expected_partitions
        )
        rep_partitions = partitions[
            pd.to_numeric(partitions["bootstrap_repetition"], errors="coerce") == repetition
        ].sort_values("partition")
        if len(rep_partitions) != len(expected_partitions):
            raise RuntimeError(
                f"Repetition {repetition} has {len(rep_partitions)} partition summaries; expected {len(expected_partitions)}."
            )

        a = np.asarray(sample["a"], dtype=int)
        event = np.asarray(sample["event"], dtype=int)
        groups = np.asarray(sample["original_groups"], dtype=int)
        multiplicity = pd.Series(groups).value_counts()
        estimates = pd.to_numeric(rep_partitions["estimate_days"], errors="raise").to_numpy(float)
        summary = pd.DataFrame([{
            "bootstrap_repetition": repetition,
            "bootstrap_seed": int(sample["bootstrap_seed"]),
            "n": n,
            "unique_original_patients": int(multiplicity.size),
            "unique_original_patient_fraction": float(multiplicity.size / n),
            "maximum_patient_multiplicity": int(multiplicity.max()),
            "treated": int(a.sum()),
            "control": int((1 - a).sum()),
            "events": int(event.sum()),
            "treated_events": int(np.sum((a == 1) & (event == 1))),
            "control_events": int(np.sum((a == 0) & (event == 1))),
            "aggregated_effect_days": float(aggregate["estimate_days"]),
            "aggregated_if_se_days": float(aggregate["if_se_days"]),
            "aggregated_if_ci_low_days": float(aggregate["if_ci_low_days"]),
            "aggregated_if_ci_high_days": float(aggregate["if_ci_high_days"]),
            "partition_mean_effect_days": float(np.mean(estimates)),
            "partition_median_effect_days": float(np.median(estimates)),
            "partition_sd_effect_days": float(np.std(estimates, ddof=1)),
            "partition_mcse_effect_days": float(np.std(estimates, ddof=1) / np.sqrt(len(estimates))),
            "partition_min_effect_days": float(np.min(estimates)),
            "partition_max_effect_days": float(np.max(estimates)),
            "partition_range_effect_days": float(np.max(estimates) - np.min(estimates)),
            "mean_censor_log_loss": float(pd.to_numeric(rep_partitions["censor_log_loss"], errors="raise").mean()),
            "mean_censor_brier": float(pd.to_numeric(rep_partitions["censor_brier"], errors="raise").mean()),
            "minimum_G_min_raw": float(pd.to_numeric(rep_partitions["G_min_raw"], errors="raise").min()),
            "minimum_G_p01_raw": float(pd.to_numeric(rep_partitions["G_p01_raw"], errors="raise").min()),
            "minimum_propensity": float(pd.to_numeric(rep_partitions["propensity_min"], errors="raise").min()),
            "minimum_propensity_p01": float(pd.to_numeric(rep_partitions["propensity_p01"], errors="raise").min()),
            "maximum_propensity_p99": float(pd.to_numeric(rep_partitions["propensity_p99"], errors="raise").max()),
            "maximum_propensity": float(pd.to_numeric(rep_partitions["propensity_max"], errors="raise").max()),
            "median_pseudo_p99": float(pd.to_numeric(rep_partitions["pseudo_p99"], errors="raise").median()),
            "maximum_pseudo_max": float(pd.to_numeric(rep_partitions["pseudo_max"], errors="raise").max()),
            "maximum_nuisance_retry": int(pd.to_numeric(rep_partitions["nuisance_retry"], errors="raise").max()),
        }])
        repetitions = append_replace_repetition(repetitions, summary, repetition)
        write_csv(repetitions.sort_values("bootstrap_repetition"), repetition_path)
        print("Completed repetition summary")
        print(dataframe_console(summary))
        print(f"Publication-bootstrap checkpoint: {len(repetitions)}/{target} completed repetitions")

    print("=" * 128)
    print("STAGE 82 CHECKPOINT STATUS")
    print("=" * 128)
    print(f"Successful repetitions: {len(repetitions)}/{target}")
    print(f"Partition summaries: {len(partitions)}/{target * len(partition_seeds)}")
    print(f"Errors currently recorded: {len(errors)}")
    if not repetitions.empty:
        print(dataframe_console(repetitions.sort_values("bootstrap_repetition"), max_rows=300))
    print("Files")
    for path in (repetition_path, partition_path, errors_path):
        print(f"- {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
