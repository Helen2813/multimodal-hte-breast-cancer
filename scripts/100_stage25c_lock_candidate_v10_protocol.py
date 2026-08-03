from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from _stage25c_v10_utils import (
    canonical_sha256,
    load_json,
    project_root,
    sha256_file,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage25c_v10_unclipped_ato_config.json"
    )
    source = config["source"]
    output = config["output"]

    print("=" * 128)
    print("STAGE 100 - LOCK UNCLIPPED STABILIZED CANDIDATE V10")
    print("=" * 128)

    existing = [
        root / output["protocol"],
        root / output["estimator_spec"],
        root / output["manifest"],
        root / output["sentinel"],
    ]
    found = [path for path in existing if path.exists()]
    if found:
        raise RuntimeError(
            "Candidate V10 lock artifacts already exist and will not be overwritten:\n"
            + "\n".join(str(path.relative_to(root)) for path in found)
        )

    table_dir = root / output["table_dir"]
    summary = load_json(
        table_dir / "s25c_99_design_summary.json"
    )
    gates = pd.read_csv(
        table_dir / "s25c_99_design_gates.csv",
        low_memory=False,
    )
    if not bool(summary["all_design_gates_pass"]):
        raise RuntimeError("Stage 25C design summary did not pass.")
    if not bool(gates["pass"].astype(str).str.lower().eq("true").all()):
        raise RuntimeError("Stage 25C gate table contains a failure.")

    v9_estimand = load_json(root / source["v9_estimand"])
    v9_bootstrap = load_json(root / source["v9_bootstrap_registry"])

    estimator_spec = {
        "status": "CANDIDATE_V10_ESTIMATOR_SPEC_LOCKED",
        "effect_estimated": False,
        "population": {
            **config["expected"],
            "eligibility": config["locked_estimator"]["population"],
            "cohort_sha256": sha256_file(root / source["v10_cohort"]),
            "compact_sha256": sha256_file(root / source["v10_compact"]),
        },
        "locked_estimator": config["locked_estimator"],
        "propensity_fit": config["propensity"],
        "ato_score": config["ato_score"],
        "bootstrap": {
            "repetitions": int(v9_bootstrap["n_repetitions"]),
            "base_seed": int(v9_bootstrap["bootstrap_base_seed"]),
            "sampling": v9_bootstrap["sampling"],
            "duplicate_fold_rule": v9_bootstrap["duplicate_fold_rule"],
            "primary_interval": v9_bootstrap["primary_interval"],
            "sensitivity_intervals": v9_bootstrap["sensitivity_intervals"],
            "propensity_refit": (
                "Refit full-sample unpenalized logistic MLE in every patient "
                "bootstrap resample."
            ),
            "censoring_and_outcome_refit": (
                "Refit the exact Candidate V9 cross-fitted censoring and outcome "
                "nuisance models in every bootstrap partition."
            ),
        },
        "pre_effect_design_summary": summary,
        "v9_reference": {
            "protocol_id": v9_estimand["protocol_id"],
            "v9_remains_immutable": True,
        },
    }
    estimator_path = root / output["estimator_spec"]
    write_json(estimator_spec, estimator_path)

    created = datetime.now(timezone.utc).isoformat()
    protocol = {
        "protocol_id": "",
        "status": "PAPER_A_CANDIDATE_V10_LOCKED_BEFORE_EFFECT_ESTIMATION",
        "created_utc": created,
        "disclosure": config["disclosure"],
        "candidate_v10_effect_computed": False,
        "relationship_to_v9": {
            "v9_protocol_id": v9_estimand["protocol_id"],
            "v9_remains_immutable": True,
            "changes": [
                "Sequencing-aware restriction to the frozen strict V10 population.",
                "Full-sample unpenalized logistic propensity on the same 13 baseline variables.",
                "Algebraically stabilized unclipped overlap AIPW score.",
            ],
            "unchanged": [
                "day-180 treatment assignment",
                "730-day post-landmark RMST outcome",
                "ATO target population",
                "13-variable compact adjustment set",
                "cross-fitted censoring nuisance",
                "cross-fitted bounded arm-specific RidgeCV outcome nuisance",
                "20 nuisance partition seeds",
                "G-min=0.10 for censoring only",
                "ordinary patient bootstrap with complete nuisance refitting",
            ],
        },
        "population": estimator_spec["population"],
        "estimator_spec_path": str(
            estimator_path.relative_to(root)
        ).replace("\\", "/"),
        "interpretation_boundary": (
            "The effect targets the overlap population among patients without "
            "documented chemotherapy initiation during the day-180 grace period. "
            "It does not generalize to patients receiving chemotherapy during "
            "that period. The design was locked before the V10 outcome effect "
            "was computed."
        ),
    }

    lock_inputs = [
        root / "stage25c_v10_unclipped_ato_config.json",
        root / "scripts/_stage25c_v10_utils.py",
        root / "scripts/99_stage25c_validate_unclipped_ato_design.py",
        root / "scripts/100_stage25c_lock_candidate_v10_protocol.py",
        root / "run_stage25c_candidate_v10_unclipped_ato_lock.ps1",
        root / source["v10_cohort"],
        root / source["v10_compact"],
        table_dir / "s25c_99_propensity_coefficients.csv",
        table_dir / "s25c_99_unclipped_ato_balance.csv",
        table_dir / "s25c_99_propensity_bootstrap_feasibility.csv",
        table_dir / "s25c_99_design_gates.csv",
        table_dir / "s25c_99_design_summary.json",
        estimator_path,
    ]
    locked_files = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in lock_inputs
    ]

    protocol_hash = canonical_sha256({
        "protocol": protocol,
        "locked_files": locked_files,
    })
    protocol_id = (
        "PAPER_A_CANDIDATE_V10_"
        + protocol_hash[:16].upper()
    )
    protocol["protocol_id"] = protocol_id

    protocol_path = root / output["protocol"]
    write_json(protocol, protocol_path)
    locked_files.append({
        "path": str(protocol_path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(protocol_path),
        "size_bytes": protocol_path.stat().st_size,
    })

    manifest = {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_hash,
        "status": protocol["status"],
        "created_utc": created,
        "candidate_v10_effect_computed": False,
        "candidate_v9_protocol_id": v9_estimand["protocol_id"],
        "locked_counts": config["expected"],
        "locked_files": locked_files,
    }
    manifest_path = root / output["manifest"]
    write_json(manifest, manifest_path)

    sentinel_path = root / output["sentinel"]
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(
        "CANDIDATE V10 LOCKED BEFORE EFFECT ESTIMATION\n"
        f"Protocol ID: {protocol_id}\n"
        f"Created UTC: {created}\n"
        "Candidate V9 remains immutable.\n"
        "The Candidate V10 effect has not been computed.\n",
        encoding="utf-8",
    )

    print("Candidate V10 protocol")
    print(json.dumps(protocol, indent=2))
    print("\nCandidate V10 manifest")
    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: Candidate V10 locked before effect estimation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
