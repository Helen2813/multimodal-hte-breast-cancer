#!/usr/bin/env python3
from __future__ import annotations

import traceback

import pandas as pd

from _stage19_utils import (
    bootstrap_payload,
    checkpoint_identity,
    dataframe_console,
    ensure_stage19_dirs,
    fit_partition_summary,
    load_stage19_config,
    project_root,
    recreate_bootstrap_sample,
    read_csv,
    validate_or_create_identity,
    write_csv,
)


def empty_or_read(path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def main() -> int:
    root = project_root()
    ensure_stage19_dirs(root)
    cfg = load_stage19_config(root)
    ext = cfg["extension"]
    tables = root / "results/tables"
    local = root / "data/derived/stage19"

    output_path = tables / "76_extended_bootstrap_partitions_checkpoint.csv"
    errors_path = tables / "76_extended_bootstrap_partition_errors.csv"
    identity_path = local / "76_stage19_checkpoint_identity.json"
    validate_or_create_identity(identity_path, checkpoint_identity(root, cfg))

    existing = empty_or_read(output_path)
    errors = empty_or_read(
        errors_path,
        [
            "bootstrap_repetition",
            "partition",
            "error_type",
            "error_message",
            "traceback",
        ],
    )
    write_csv(errors, errors_path)
    payload = bootstrap_payload()
    target_reps = int(ext["bootstrap_repetitions"])
    partition_numbers = [int(x) for x in ext["new_partition_numbers"]]
    partition_seeds = [int(x) for x in ext["new_partition_base_seeds"]]
    if len(partition_numbers) != len(partition_seeds):
        raise ValueError("Stage 19 partition number and seed lists differ in length.")

    complete_keys = set()
    if not existing.empty:
        complete_keys = set(
            zip(
                pd.to_numeric(existing["bootstrap_repetition"], errors="raise").astype(int),
                pd.to_numeric(existing["partition"], errors="raise").astype(int),
            )
        )

    print("=" * 124)
    print("STAGE 76 - EXTEND THE SAME 30 BOOTSTRAP SAMPLES TO 20 NUISANCE PARTITIONS")
    print("=" * 124)
    print(f"Bootstrap repetitions: {target_reps}")
    print(f"Existing Stage 18 partitions: 1-{ext['stage18_existing_partitions']}")
    print(f"New Stage 19 partitions: {partition_numbers[0]}-{partition_numbers[-1]}")
    print("No new bootstrap samples are drawn. The Stage 18 sampling seeds are reused exactly.")
    print("Completed Stage 19 partition fits are checkpointed and skipped on resume.")

    for repetition in range(1, target_reps + 1):
        sample = recreate_bootstrap_sample(payload, repetition, cfg)
        print("-" * 124)
        print(
            f"BOOTSTRAP REPETITION {repetition:02d}/{target_reps}; "
            f"bootstrap seed={sample['bootstrap_seed']}"
        )
        new_rows = []
        for partition_number, base_seed in zip(partition_numbers, partition_seeds):
            key = (repetition, partition_number)
            if key in complete_keys:
                print(f"Partition {partition_number:02d} already complete; skipping.")
                continue
            try:
                row = fit_partition_summary(
                    sample,
                    repetition,
                    partition_number,
                    base_seed,
                    cfg,
                )
                new_rows.append(row)
                print(
                    f"partition={partition_number:02d} "
                    f"effect={row['estimate_days']:.6f} "
                    f"IF_SE={row['if_se_days']:.6f} "
                    f"G_min_raw={row['G_min_raw']:.6f} "
                    f"pseudo_p99={row['pseudo_p99']:.6f} "
                    f"pseudo_max={row['pseudo_max']:.6f}"
                )
            except Exception as exc:
                error_row = {
                    "bootstrap_repetition": repetition,
                    "partition": partition_number,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors = pd.concat([errors, pd.DataFrame([error_row])], ignore_index=True)
                write_csv(errors, errors_path)
                print(
                    f"partition={partition_number:02d} FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )
        if new_rows:
            existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            existing = (
                existing.sort_values(["bootstrap_repetition", "partition"])
                .drop_duplicates(["bootstrap_repetition", "partition"], keep="last")
                .reset_index(drop=True)
            )
            write_csv(existing, output_path)
            complete_keys.update(
                (int(row["bootstrap_repetition"]), int(row["partition"])) for row in new_rows
            )
        rep_count = int(
            sum(1 for partition_number in partition_numbers if (repetition, partition_number) in complete_keys)
        )
        print(f"Checkpoint: new partitions complete for this repetition = {rep_count}/{len(partition_numbers)}")

    expected = target_reps * len(partition_numbers)
    completed = len(complete_keys)
    write_csv(errors, errors_path)
    print("=" * 124)
    print("STAGE 76 COMPLETION SUMMARY")
    print("=" * 124)
    print(f"Expected new partition fits: {expected}")
    print(f"Completed new partition fits: {completed}")
    print(f"Errors recorded: {len(errors)}")
    if not existing.empty:
        print("\nNew partition result summary")
        summary = (
            existing.groupby("partition", as_index=False)
            .agg(
                repetitions=("bootstrap_repetition", "nunique"),
                mean_effect_days=("estimate_days", "mean"),
                sd_effect_days=("estimate_days", "std"),
                minimum_effect_days=("estimate_days", "min"),
                maximum_effect_days=("estimate_days", "max"),
                median_G_min_raw=("G_min_raw", "median"),
                maximum_pseudo_max=("pseudo_max", "max"),
            )
        )
        print(dataframe_console(summary))
    print("\nFiles")
    print(f"- {output_path.relative_to(root)}")
    print(f"- {errors_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
