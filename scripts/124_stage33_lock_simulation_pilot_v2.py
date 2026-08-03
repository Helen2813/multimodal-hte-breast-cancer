from __future__ import annotations

from datetime import datetime, timezone
import json

from _stage33_simulation_v2_utils import (
    canonical_sha256,
    load_json,
    project_root,
    sha256_file,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage33_sequence_simulation_pilot_v2_config.json"
    )
    output = config["outputs"]
    manifest_path = root / output["manifest"]

    print("=" * 128)
    print("STAGE 124 - LOCK REVISED SEQUENCING SIMULATION PILOT")
    print("=" * 128)

    stage32_summary = load_json(
        root / config["source"]["stage32_summary"]
    )
    stage32_dgm = load_json(
        root / config["source"]["stage32_recommended_dgm"]
    )
    if (
        stage32_summary["status"]
        != config["expected"]["stage32_status"]
    ):
        raise RuntimeError("Stage 32 calibration is not accepted.")
    if (
        stage32_summary["calibration_id"]
        != config["expected"]["stage32_calibration_id"]
    ):
        raise RuntimeError("Unexpected Stage 32 calibration ID.")
    if (
        stage32_dgm["calibration_id"]
        != config["expected"]["stage32_calibration_id"]
    ):
        raise RuntimeError("Stage 32 recommended DGM ID mismatch.")

    calibrated = stage32_dgm["calibrated_parameters"]
    scenario_parameters = {
        "sequencing_strengths": {
            "none": 0.0,
            "half_empirical": (
                float(calibrated["sequencing_strength"]) / 2.0
            ),
            "empirical": float(
                calibrated["sequencing_strength"]
            ),
        },
        "treatment_effects": {
            "null": 0.0,
            "empirically_calibrated_benefit": float(
                calibrated[
                    "true_treatment_log_hazard_effect"
                ]
            ),
        },
    }

    locked_inputs = [
        root / "stage33_sequence_simulation_pilot_v2_config.json",
        root / "scripts/_stage33_simulation_v2_utils.py",
        root / "scripts/124_stage33_lock_simulation_pilot_v2.py",
        root / "scripts/125_stage33_run_simulation_pilot_v2.py",
        root / "scripts/126_stage33_summarize_simulation_pilot_v2.py",
        root / "run_stage33_sequence_simulation_pilot_v2.ps1",
        root / config["source"]["stage32_manifest"],
        root / config["source"]["stage32_recommended_dgm"],
        root / config["source"]["stage32_summary"],
        root / config["source"]["stage31_utils"],
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
                    "Existing Stage 33 lock failed integrity: "
                    + item["path"]
                )
        print("Existing Stage 33 lock verified.")
        print(json.dumps(existing, indent=2))
        return 0

    manifest = {
        "status": "STAGE33_SIMULATION_PILOT_V2_LOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage32_calibration_id": stage32_summary[
            "calibration_id"
        ],
        "calibrated_parameters": calibrated,
        "fixed_covariate_parameters": stage32_dgm[
            "fixed_covariate_parameters"
        ],
        "censoring_parameters": stage32_dgm[
            "censoring_parameters"
        ],
        "scenario_parameters": scenario_parameters,
        "simulation": config["simulation"],
        "methods": config["methods"],
        "pilot_gates": config["pilot_gates"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
    }
    manifest["simulation_id"] = (
        "PAPER_A_SEQUENCE_SIM_PILOT_V2_"
        + canonical_sha256(manifest)[:16].upper()
    )
    write_json(manifest, manifest_path)

    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: revised scenarios, truths, decomposition, and "
        "diagnostics locked before pilot results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
