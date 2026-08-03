from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path.cwd().resolve()
    config = load_json(root / "stage34_confirmatory_simulation_config.json")
    source = config["source"]
    output = config["outputs"]

    print("=" * 128)
    print("STAGE 130 - LOCK INDEPENDENT CONFIRMATORY SIMULATION")
    print("=" * 128)

    decision = load_json(root / source["stage33c_decision"])
    if decision["status"] != config["expected"]["stage33c_status"]:
        raise RuntimeError("Stage 33C did not authorize confirmatory simulation.")

    dgm = load_json(root / source["stage32_recommended_dgm"])
    if dgm["calibration_id"] != config["expected"]["stage32_calibration_id"]:
        raise RuntimeError("Unexpected Stage 32 calibration ID.")

    calibrated = dgm["calibrated_parameters"]
    scenario_parameters = {
        "sequencing_strengths": {
            "none": 0.0,
            "half_empirical": float(calibrated["sequencing_strength"]) / 2.0,
            "empirical": float(calibrated["sequencing_strength"]),
        },
        "treatment_effects": {
            "true_zero": 0.0,
            "observed_risk_benefit": float(
                calibrated["true_treatment_log_hazard_effect"]
            ),
        },
    }

    locked_relatives = [
        "stage34_confirmatory_simulation_config.json",
        "scripts/130_stage34_lock_confirmatory_simulation.py",
        "scripts/131_stage34_run_confirmatory_simulation.py",
        "scripts/132_stage34_summarize_confirmatory_simulation.py",
        "run_stage34_confirmatory_sequence_simulation.ps1",
        source["stage32_recommended_dgm"],
        source["stage33c_manifest"],
        source["stage33c_decision"],
        source["stage31_utils"],
        source["stage33_utils"],
    ]
    locked_files = []
    for relative in locked_relatives:
        path = root / relative
        locked_files.append({
            "path": relative.replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })

    manifest = {
        "status": "STAGE34_CONFIRMATORY_SIMULATION_LOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage32_calibration_id": dgm["calibration_id"],
        "stage33c_decision_id": decision["decision_id"],
        "calibrated_parameters": calibrated,
        "fixed_covariate_parameters": dgm["fixed_covariate_parameters"],
        "censoring_parameters": dgm["censoring_parameters"],
        "scenario_parameters": scenario_parameters,
        "simulation": config["simulation"],
        "methods": config["methods"],
        "confirmatory_gates": config["confirmatory_gates"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
    }
    raw = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["simulation_id"] = (
        "PAPER_A_SEQUENCE_CONFIRMATORY_"
        + hashlib.sha256(raw).hexdigest()[:16].upper()
    )
    write_json(manifest, root / output["manifest"])

    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: independent confirmatory scenarios, seeds, methods, "
        "truths, and gates locked before execution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
