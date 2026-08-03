from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from _stage29_v10_event_influence_utils import (
    assemble_frame,
    canonical_sha256,
    count_checks,
    dataframe_console,
    load_json,
    project_root,
    sha256_file,
    verify_inputs,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage29_v10_event_influence_config.json"
    )
    output = config["output"]
    manifest_path = root / output["calculation_manifest"]

    print("=" * 128)
    print("STAGE 112 - LOCK LEAVE-ONE-EVENT-PATIENT-OUT ANALYSIS")
    print("=" * 128)

    verification = verify_inputs(root, config)
    frame, features = assemble_frame(root, config)
    checks = count_checks(frame, features, config)

    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Frozen V10 count checks failed.\n"
            + dataframe_console(checks)
        )

    event_frame = frame[
        pd.to_numeric(
            frame["analysis_event"],
            errors="raise",
        ).astype(int)
        == 1
    ].copy()
    event_frame["omitted_arm"] = (
        pd.to_numeric(
            event_frame["analysis_treatment"],
            errors="raise",
        )
        .astype(int)
        .map({0: "control", 1: "early_hormone"})
    )
    event_frame = event_frame.sort_values(
        ["omitted_arm", "patient_id_normalized"]
    ).reset_index(drop=True)
    event_frame["event_case_id"] = [
        f"E{i:03d}" for i in range(1, len(event_frame) + 1)
    ]

    if len(event_frame) != int(
        config["expected"]["required_successful_deletions"]
    ):
        raise RuntimeError(
            f"Expected 36 event patients, found {len(event_frame)}."
        )

    locked_inputs = [
        root / "stage29_v10_event_influence_config.json",
        root / "scripts/_stage29_v10_event_influence_utils.py",
        root / "scripts/112_stage29_lock_event_influence.py",
        root / "scripts/113_stage29_run_leave_one_event_out.py",
        root / "scripts/114_stage29_summarize_event_influence.py",
        root / "run_stage29_candidate_v10_event_influence.ps1",
        root / config["source"]["v10_manifest"],
        root / config["source"]["v10_protocol"],
        root / config["source"]["v10_estimator_spec"],
        root / config["source"]["stage25c_config"],
        root / config["source"]["stage26_config"],
        root / config["source"]["stage26_calculation_manifest"],
        root / config["source"]["stage26_point_estimate"],
        root / config["source"]["v10_cohort"],
        root / config["source"]["v10_compact"],
        root / "scripts/_stage25c_v10_utils.py",
        root / "scripts/_stage26_v10_utils.py",
    ]
    locked_files = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in locked_inputs
    ]

    omitted_set_frame = event_frame[
        [
            "event_case_id",
            "patient_id_normalized",
            "omitted_arm",
            "analysis_time",
        ]
    ].copy()
    omitted_set = omitted_set_frame.to_dict("records")

    if manifest_path.exists():
        existing = load_json(manifest_path)
        for item in existing["locked_files"]:
            path = root / item["path"]
            if (
                not path.exists()
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(
                    "Existing Stage 29 lock failed integrity: "
                    + item["path"]
                )
        print("Existing Stage 29 lock verified.")
        print(json.dumps(existing, indent=2))
        return 0

    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "STAGE29_EVENT_INFLUENCE_LOCKED",
        "created_utc": created,
        "v10_protocol_id": verification["v10_protocol_id"],
        "stage26_calculation_id": verification[
            "stage26_calculation_id"
        ],
        "primary_point_estimate_days": verification[
            "stage26_point_estimate_days"
        ],
        "event_patients": {
            "total": len(event_frame),
            "early_hormone": int(
                (event_frame["omitted_arm"] == "early_hormone").sum()
            ),
            "control": int(
                (event_frame["omitted_arm"] == "control").sum()
            ),
        },
        "omitted_set_sha256": canonical_sha256(omitted_set),
        "omitted_set_path_LOCAL_ONLY": output["locked_event_set_local"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
        "v10_integrity_hash": verification[
            "v10_integrity_hash"
        ],
    }
    manifest["influence_id"] = (
        "PAPER_A_V10_EVENT_INFLUENCE_"
        + canonical_sha256(manifest)[:16].upper()
    )
    write_json(manifest, manifest_path)

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    omitted_set_frame.to_csv(
        root / output["locked_event_set_local"],
        index=False,
        encoding="utf-8-sig",
    )
    checks.to_csv(
        table_dir / "s29_112_preflight_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Frozen V10 checks")
    print(dataframe_console(checks))
    print("\nStage 29 lock")
    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: all 36 event-patient deletions locked before "
        "influence estimates were computed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
