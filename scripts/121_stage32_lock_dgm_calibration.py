from __future__ import annotations

from datetime import datetime, timezone
import json

from _stage32_dgm_calibration_utils import (
    canonical_sha256,
    empirical_target_values,
    load_json,
    project_root,
    sha256_file,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage32_sequence_dgm_calibration_config.json"
    )
    output = config["outputs"]
    manifest_path = root / output["manifest"]

    print("=" * 128)
    print("STAGE 121 - LOCK EMPIRICAL DGM CALIBRATION")
    print("=" * 128)

    locked_inputs = [
        root / "stage32_sequence_dgm_calibration_config.json",
        root / "scripts/_stage32_dgm_calibration_utils.py",
        root / "scripts/121_stage32_lock_dgm_calibration.py",
        root / "scripts/122_stage32_run_dgm_calibration.py",
        root / "scripts/123_stage32_summarize_dgm_calibration.py",
        root / "run_stage32_sequence_dgm_calibration.ps1",
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
                    "Existing Stage 32 lock failed integrity: "
                    + item["path"]
                )
        print("Existing Stage 32 calibration lock verified.")
        print(json.dumps(existing, indent=2))
        return 0

    manifest = {
        "status": "STAGE32_DGM_CALIBRATION_LOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "empirical_targets": empirical_target_values(config),
        "target_tolerances": config["target_tolerances"],
        "calibration": config["calibration"],
        "decision_rules": config["decision_rules"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
    }
    manifest["calibration_id"] = (
        "PAPER_A_SEQUENCE_DGM_CAL_"
        + canonical_sha256(manifest)[:16].upper()
    )
    write_json(manifest, manifest_path)

    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: empirical targets, tolerances, and parameter "
        "bounds locked before calibration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
