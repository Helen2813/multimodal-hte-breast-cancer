from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from _stage30_v10_non_event_influence_utils import (
    assemble_frame,
    canonical_sha256,
    count_checks,
    dataframe_console,
    load_json,
    load_primary_influence_table,
    project_root,
    sha256_file,
    verify_inputs,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage30_v10_non_event_influence_config.json"
    )
    output = config["output"]
    manifest_path = root / output["calculation_manifest"]

    print("=" * 128)
    print("STAGE 115 - LOCK TOP-INFLUENCE NON-EVENT PATIENT SET")
    print("=" * 128)

    verification = verify_inputs(root, config)
    frame, features = assemble_frame(root, config)
    checks = count_checks(frame, features, config)
    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Frozen V10 count checks failed.\n"
            + dataframe_console(checks)
        )

    influence = load_primary_influence_table(root, config)

    event_counts = (
        influence.groupby(
            ["event_status", "arm"],
            dropna=False,
        )
        .size()
        .reset_index(name="patients")
    )

    top_overall = (
        influence.sort_values(
            ["absolute_influence", "row_index"],
            ascending=[False, True],
        )
        .head(20)
        .reset_index(drop=True)
    )
    composition = top_overall[
        [
            "row_index",
            "arm",
            "event_status",
            "analysis_time",
            "influence",
            "absolute_influence",
            "normalized_contribution_days",
            "absolute_normalized_contribution_days",
        ]
    ].copy()
    composition.insert(
        0,
        "overall_influence_rank",
        np.arange(1, len(composition) + 1),
    )
    write_csv(
        composition,
        root / output["composition_table"],
    )

    top_k = int(config["selection"]["top_k"])
    selected = (
        influence[
            influence["analysis_event"].astype(int) == 0
        ]
        .sort_values(
            ["absolute_influence", "row_index"],
            ascending=[False, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )
    if len(selected) != top_k:
        raise RuntimeError(
            f"Expected {top_k} selected non-event patients, "
            f"found {len(selected)}."
        )

    selected.insert(
        0,
        "influence_rank",
        np.arange(1, len(selected) + 1),
    )
    selected.insert(
        1,
        "influence_case_id",
        [f"N{i:03d}" for i in range(1, len(selected) + 1)],
    )

    locked_columns = [
        "influence_rank",
        "influence_case_id",
        "patient_id_normalized",
        "row_index",
        "arm",
        "analysis_time",
        "influence",
        "absolute_influence",
        "normalized_contribution_days",
        "absolute_normalized_contribution_days",
    ]
    selected_local = selected[locked_columns].copy()
    write_csv(
        selected_local,
        root / output["locked_set_local"],
    )

    selected_hash_payload = (
        selected_local.to_dict("records")
    )

    locked_inputs = [
        root / "stage30_v10_non_event_influence_config.json",
        root / "scripts/_stage30_v10_non_event_influence_utils.py",
        root / "scripts/115_stage30_lock_top_non_event_set.py",
        root / "scripts/116_stage30_run_top_non_event_leave_one_out.py",
        root / "scripts/117_stage30_summarize_non_event_influence.py",
        root / "run_stage30_candidate_v10_non_event_influence.ps1",
        root / config["source"]["v10_manifest"],
        root / config["source"]["v10_protocol"],
        root / config["source"]["v10_estimator_spec"],
        root / config["source"]["stage25c_config"],
        root / config["source"]["stage26_config"],
        root / config["source"]["stage26_calculation_manifest"],
        root / config["source"]["stage26_point_estimate"],
        root / config["source"]["stage26_patient_scores"],
        root / config["source"]["stage29_event_results"],
        root / config["source"]["v10_cohort"],
        root / config["source"]["v10_compact"],
        root / "scripts/_stage25c_v10_utils.py",
        root / "scripts/_stage26_v10_utils.py",
        root / "scripts/_stage29_v10_event_influence_utils.py",
    ]
    locked_files = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in locked_inputs
    ]

    if manifest_path.exists():
        existing = load_json(manifest_path)
        for item in existing["locked_files"]:
            path = root / item["path"]
            if (
                not path.exists()
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(
                    "Existing Stage 30 lock failed integrity: "
                    + item["path"]
                )
        print("Existing Stage 30 lock verified.")
        print(json.dumps(existing, indent=2))
        return 0

    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "STAGE30_NON_EVENT_INFLUENCE_LOCKED",
        "created_utc": created,
        "v10_protocol_id": verification["v10_protocol_id"],
        "stage26_calculation_id": verification[
            "stage26_calculation_id"
        ],
        "primary_point_estimate_days": verification[
            "stage26_point_estimate_days"
        ],
        "selection": config["selection"],
        "selected_patients": top_k,
        "selected_set_path_LOCAL_ONLY": output[
            "locked_set_local"
        ],
        "selected_set_sha256": canonical_sha256(
            selected_hash_payload
        ),
        "primary_influence_composition": (
            event_counts.to_dict("records")
        ),
        "boundary": config["boundary"],
        "locked_files": locked_files,
        "v10_integrity_hash": verification[
            "v10_integrity_hash"
        ],
    }
    manifest["influence_id"] = (
        "PAPER_A_V10_NON_EVENT_INFLUENCE_"
        + canonical_sha256(manifest)[:16].upper()
    )
    write_json(manifest, manifest_path)

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        checks,
        table_dir / "s30_115_preflight_checks.csv",
    )

    print("Frozen V10 checks")
    print(dataframe_console(checks))
    print("\nTop-20 primary influence composition")
    print(dataframe_console(composition))
    print("\nStage 30 lock")
    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: top-10 non-event influence set locked before "
        "leave-one-out refits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
