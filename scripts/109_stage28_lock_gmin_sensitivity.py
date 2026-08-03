from __future__ import annotations

from datetime import datetime, timezone
import json

from _stage28_v10_gmin_utils import (
    canonical_sha256,
    load_json,
    project_root,
    sha256_file,
    verify_stage28_inputs,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage28_v10_gmin_sensitivity_config.json"
    )
    output = config["output"]
    manifest_path = root / output["calculation_manifest"]

    print("=" * 128)
    print("STAGE 109 - VERIFY INPUTS AND LOCK G-MIN SENSITIVITY")
    print("=" * 128)

    verification = verify_stage28_inputs(root, config)

    locked_inputs = [
        root / "stage28_v10_gmin_sensitivity_config.json",
        root / "scripts/_stage28_v10_gmin_utils.py",
        root / "scripts/109_stage28_lock_gmin_sensitivity.py",
        root / "scripts/110_stage28_compute_gmin_sensitivity.py",
        root / "scripts/111_stage28_summarize_gmin_sensitivity.py",
        root / "run_stage28_candidate_v10_gmin_sensitivity.ps1",
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

    if manifest_path.exists():
        existing = load_json(manifest_path)
        for item in existing["locked_files"]:
            path = root / item["path"]
            if (
                not path.exists()
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(
                    "Existing Stage 28 lock failed integrity: "
                    + item["path"]
                )
        print("Existing Stage 28 lock verified.")
        print(json.dumps(existing, indent=2))
        return 0

    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "STAGE28_GMIN_SENSITIVITY_LOCKED",
        "created_utc": created,
        "v10_protocol_id": verification["v10_protocol_id"],
        "stage26_calculation_id": verification[
            "stage26_calculation_id"
        ],
        "stage26_primary_point_estimate_days": verification[
            "stage26_point_estimate_days"
        ],
        "g_min_values": config["sensitivity"]["g_min_values"],
        "primary_g_min": config["sensitivity"]["primary_value"],
        "post_hoc_sensitivity_values": config[
            "sensitivity"
        ]["post_hoc_values"],
        "inference": config["sensitivity"]["inference"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
        "v10_integrity_hash": verification[
            "v10_integrity_hash"
        ],
    }
    manifest["sensitivity_id"] = (
        "PAPER_A_V10_GMIN_"
        + canonical_sha256(manifest)[:16].upper()
    )
    write_json(manifest, manifest_path)

    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: Stage 28 G-min sensitivity locked before "
        "sensitivity estimates were computed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
