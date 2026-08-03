from __future__ import annotations

from datetime import datetime, timezone
import json

from _stage26_v10_utils import (
    canonical_sha256,
    dataframe_console,
    load_json,
    project_root,
    sha256_file,
    verify_manifest,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage26_v10_point_estimate_config.json"
    )
    output = config["output"]

    print("=" * 128)
    print("STAGE 101 - VERIFY V10 LOCK AND LOCK POINT-ESTIMATE CALCULATION")
    print("=" * 128)

    calculation_manifest_path = (
        root / output["calculation_manifest"]
    )
    if calculation_manifest_path.exists():
        raise RuntimeError(
            "Stage 26 calculation manifest already exists. "
            "This runner refuses to overwrite a point-estimate lock."
        )
    for key in (
        "partition_table",
        "patient_scores",
        "point_estimate",
        "diagnostics",
        "decision",
    ):
        path = root / output[key]
        if path.exists():
            raise RuntimeError(
                f"Stage 26 output already exists and will not be overwritten: {path}"
            )

    manifest, integrity = verify_manifest(
        root,
        config,
    )
    protocol = load_json(
        root / config["source"]["v10_protocol"]
    )
    estimator_spec = load_json(
        root / config["source"]["v10_estimator_spec"]
    )

    if protocol["protocol_id"] != config["expected"]["protocol_id"]:
        raise RuntimeError("V10 protocol ID mismatch.")
    if bool(protocol["candidate_v10_effect_computed"]):
        raise RuntimeError(
            "V10 protocol indicates that the effect was already computed."
        )
    if bool(estimator_spec["effect_estimated"]):
        raise RuntimeError(
            "V10 estimator spec indicates that the effect was already estimated."
        )

    lock_inputs = [
        root / "stage26_v10_point_estimate_config.json",
        root / "scripts/_stage26_v10_utils.py",
        root / "scripts/101_stage26_lock_point_estimate_calculation.py",
        root / "scripts/102_stage26_compute_candidate_v10_point_estimate.py",
        root / "scripts/103_stage26_review_point_estimate_diagnostics.py",
        root / "run_stage26_candidate_v10_point_estimate.ps1",
        root / config["source"]["v10_manifest"],
        root / config["source"]["v10_protocol"],
        root / config["source"]["v10_estimator_spec"],
        root / config["source"]["stage25c_config"],
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

    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "stage": 26,
        "status": (
            "CANDIDATE_V10_POINT_ESTIMATE_CALCULATION_LOCKED"
        ),
        "created_utc": created,
        "v10_protocol_id": protocol["protocol_id"],
        "effect_computed_at_lock": False,
        "estimator": config["estimator"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
        "v10_lock_integrity_hash": canonical_sha256(
            integrity[
                ["path", "expected_sha256"]
            ].to_dict("records")
        ),
    }
    payload["calculation_id"] = (
        "PAPER_A_V10_POINT_"
        + canonical_sha256(payload)[:16].upper()
    )
    write_json(payload, calculation_manifest_path)

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    integrity.to_csv(
        table_dir / "s26_101_v10_lock_integrity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("V10 locked-file integrity")
    print(dataframe_console(integrity, max_rows=100))
    print("\nStage 26 calculation lock")
    print(json.dumps(payload, indent=2))
    print(
        "\nPASS: Candidate V10 point-estimate calculation "
        "locked before effect estimation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
