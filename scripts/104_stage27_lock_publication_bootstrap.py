from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from _stage27_v10_bootstrap_utils import (
    canonical_sha256,
    dataframe_console,
    load_json,
    locked_bootstrap_settings,
    project_root,
    sha256_file,
    verify_hash_manifest,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage27_v10_bootstrap_config.json"
    )
    output = config["output"]
    lock_path = root / output["bootstrap_lock"]

    print("=" * 128)
    print("STAGE 104 - VERIFY V10/STAGE26 AND LOCK PUBLICATION BOOTSTRAP")
    print("=" * 128)

    v10_manifest, v10_integrity = verify_hash_manifest(
        root,
        root / config["source"]["v10_manifest"],
    )
    stage26_manifest = load_json(
        root / config["source"]["stage26_calculation_manifest"]
    )
    if (
        stage26_manifest["calculation_id"]
        != config["expected"]["stage26_calculation_id"]
    ):
        raise RuntimeError("Unexpected Stage 26 calculation ID.")

    locked_bootstrap = locked_bootstrap_settings(
        root,
        config,
    )

    point = pd.read_csv(
        root / config["source"]["stage26_point_estimate"],
        low_memory=False,
    ).iloc[0]
    observed_point = float(
        point["aipw_ato_rmst_difference_days"]
    )
    if abs(
        observed_point
        - float(config["expected"]["point_estimate_days"])
    ) > 1e-12:
        raise RuntimeError("Stage 26 point estimate mismatch.")

    lock_inputs = [
        root / "stage27_v10_bootstrap_config.json",
        root / "scripts/_stage27_v10_bootstrap_utils.py",
        root / "scripts/104_stage27_lock_publication_bootstrap.py",
        root / "scripts/105_stage27_identity_reproduction.py",
        root / "scripts/106_stage27_run_publication_bootstrap.py",
        root / "scripts/107_stage27_summarize_publication_bootstrap.py",
        root / "run_stage27_candidate_v10_publication_bootstrap.ps1",
        root / config["source"]["v10_manifest"],
        root / config["source"]["v10_protocol"],
        root / config["source"]["v10_estimator_spec"],
        root / config["source"]["stage25c_config"],
        root / config["source"]["stage26_calculation_manifest"],
        root / config["source"]["stage26_point_estimate"],
        root / config["source"]["stage26_partition_estimates"],
        root / config["source"]["v10_cohort"],
        root / config["source"]["v10_compact"],
    ]
    locked_files = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in lock_inputs
    ]

    if lock_path.exists():
        existing = load_json(lock_path)
        for item in existing["locked_files"]:
            path = root / item["path"]
            if (
                not path.exists()
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(
                    f"Existing Stage 27 lock failed integrity: {item['path']}"
                )
        print("Existing Stage 27 bootstrap lock verified.")
        print(json.dumps(existing, indent=2))
        return 0

    created = datetime.now(timezone.utc).isoformat()
    lock = {
        "status": "CANDIDATE_V10_PUBLICATION_BOOTSTRAP_LOCKED",
        "created_utc": created,
        "v10_protocol_id": config["expected"]["protocol_id"],
        "stage26_calculation_id": config["expected"][
            "stage26_calculation_id"
        ],
        "point_estimate_days": observed_point,
        "bootstrap": {
            **config["bootstrap"],
            **locked_bootstrap,
        },
        "estimator": config["estimator"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
        "v10_integrity_hash": canonical_sha256(
            v10_integrity[
                ["path", "expected_sha256"]
            ].to_dict("records")
        ),
    }
    lock["bootstrap_id"] = (
        "PAPER_A_V10_BOOTSTRAP_"
        + canonical_sha256(lock)[:16].upper()
    )
    write_json(lock, lock_path)

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    v10_integrity.to_csv(
        table_dir / "s27_104_v10_lock_integrity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("V10 lock integrity")
    print(dataframe_console(v10_integrity, max_rows=100))
    print("\nStage 27 bootstrap lock")
    print(json.dumps(lock, indent=2))
    print("\nPASS: publication bootstrap locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
