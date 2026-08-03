#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage18_utils import aggregate_partition_patient_scores
from _stage25_v10_utils import dataframe_console, load_config, load_json, project_root, write_csv
from _stage26_v10_utils import (
    assemble_v10_data,
    ensure_stage26_dirs,
    fit_v10_partition,
    verify_v10_lock,
)


def main() -> int:
    root = project_root()
    config = load_config(root)
    ensure_stage26_dirs(root)

    print("=" * 128)
    print("STAGE 97 - LOCKED CANDIDATE V10 20-PARTITION POINT ESTIMATE")
    print("=" * 128)

    lock_check = verify_v10_lock(root, config)
    manifest = load_json(root / config["output"]["manifest"])
    locked_counts = manifest["locked_counts"]

    frame, features, metadata = assemble_v10_data(root)
    treatment = pd.to_numeric(
        frame["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"], errors="raise"
    ).astype(int).to_numpy()
    observed_time = pd.to_numeric(
        frame["analysis_time"], errors="coerce"
    ).to_numpy(dtype=float)

    checks = pd.DataFrame([
        {
            "check": "n",
            "observed": metadata["n"],
            "expected": locked_counts["n"],
            "pass": metadata["n"] == int(locked_counts["n"]),
        },
        {
            "check": "treated",
            "observed": metadata["treated"],
            "expected": locked_counts["treated"],
            "pass": metadata["treated"] == int(locked_counts["treated"]),
        },
        {
            "check": "control",
            "observed": metadata["control"],
            "expected": locked_counts["control"],
            "pass": metadata["control"] == int(locked_counts["control"]),
        },
        {
            "check": "events",
            "observed": metadata["events"],
            "expected": locked_counts["events"],
            "pass": metadata["events"] == int(locked_counts["events"]),
        },
        {
            "check": "features",
            "observed": metadata["features"],
            "expected": locked_counts["features"],
            "pass": metadata["features"] == int(locked_counts["features"]),
        },
        {
            "check": "partition seed count",
            "observed": len(config["estimator"]["partition_base_seeds"]),
            "expected": 20,
            "pass": len(config["estimator"]["partition_base_seeds"]) == 20,
        },
    ])
    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Candidate V10 point-estimate preflight failed.\n"
            + dataframe_console(checks)
        )

    print(f"Protocol ID: {manifest['protocol_id']}")
    print(f"V10 lock integrity: PASS for {len(lock_check)} files")
    print("Preflight checks")
    print(dataframe_console(checks))

    rows = []
    score_frames = []
    for partition_number, base_seed in enumerate(
        config["estimator"]["partition_base_seeds"], start=1
    ):
        row, patient = fit_v10_partition(
            frame,
            features,
            treatment,
            event,
            observed_time,
            partition_number,
            int(base_seed),
            config,
        )
        rows.append(row)
        score_frames.append(patient)
        print(
            f"partition={partition_number:02d} seed={base_seed} "
            f"effect={row['estimate_days']:.6f} "
            f"IF_SE={row['if_se_days']:.6f} "
            f"G_min={row['G_min_raw']:.6f} "
            f"prop_p01={row['propensity_p01']:.6f} "
            f"prop_p99={row['propensity_p99']:.6f} "
            f"pseudo_p99={row['pseudo_p99']:.6f} "
            f"retry={row['nuisance_retry']}"
        )

    partitions = pd.DataFrame(rows)
    score_df = pd.concat(score_frames, ignore_index=True).rename(
        columns={"row_index": "bootstrap_row_index"}
    )
    score_df["original_patient_group"] = score_df["bootstrap_row_index"]

    prefix_rows = []
    for prefix in (5, 10, 15, 20):
        aggregate = aggregate_partition_patient_scores(
            score_df[score_df["partition"] <= prefix]
        )
        values = partitions.loc[
            partitions["partition"] <= prefix,
            "estimate_days",
        ].to_numpy(dtype=float)
        prefix_rows.append({
            "prefix_partitions": prefix,
            "estimate_days": float(aggregate["estimate_days"]),
            "if_se_days": float(aggregate["if_se_days"]),
            "if_ci_low_days": float(aggregate["if_ci_low_days"]),
            "if_ci_high_days": float(aggregate["if_ci_high_days"]),
            "partition_mean_days": float(np.mean(values)),
            "partition_sd_days": float(np.std(values, ddof=1)),
            "partition_mcse_days": float(
                np.std(values, ddof=1) / np.sqrt(prefix)
            ),
        })

    prefixes = pd.DataFrame(prefix_rows)
    final = prefixes[prefixes["prefix_partitions"] == 20].copy()
    final.insert(0, "protocol_id", manifest["protocol_id"])
    final.insert(1, "protocol_candidate", "PAPER_A_CANDIDATE_V10")
    final["n"] = metadata["n"]
    final["treated"] = metadata["treated"]
    final["control"] = metadata["control"]
    final["events"] = metadata["events"]
    final["treated_events"] = metadata["treated_events"]
    final["control_events"] = metadata["control_events"]
    final["horizon_days"] = float(config["timing"]["horizon_days"])
    final["landmark_day"] = int(
        config["candidate_v10_population"]["landmark_day"]
    )
    final["g_min"] = float(config["estimator"]["primary_g_min"])
    final["partitions"] = 20

    table_dir = root / "results/tables/stage26_candidate_v10"
    local_dir = root / "data/derived/stage26_candidate_v10"
    write_csv(checks, table_dir / "s26_97_point_preflight_checks.csv")
    write_csv(partitions, table_dir / "s26_97_partition_estimates.csv")
    write_csv(prefixes, table_dir / "s26_97_prefix_convergence.csv")
    write_csv(final, table_dir / "s26_97_final_point_estimate.csv")
    write_csv(
        score_df,
        local_dir / "s26_97_patient_scores_LOCAL_ONLY.csv",
    )

    print("\nAll partition estimates")
    print(dataframe_console(partitions, max_rows=30))
    print("\nPrefix convergence")
    print(dataframe_console(prefixes))
    print("\nCandidate V10 point estimate")
    print(dataframe_console(final))
    print(
        "\nPASS: locked Candidate V10 point estimate completed. "
        "Do not start a publication bootstrap until diagnostics are reviewed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
