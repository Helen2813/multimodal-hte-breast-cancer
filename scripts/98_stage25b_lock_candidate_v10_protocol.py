from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from _stage25b_v10_balance_utils import (
    canonical_sha256,
    load_json,
    project_root,
    sha256_file,
    verify_stage25_inputs,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage25b_v10_balance_repair_config.json"
    )
    source = config["source"]
    output = config["output"]

    print("=" * 128)
    print("STAGE 98 - LOCK BALANCE-REPAIRED CANDIDATE V10 BEFORE EFFECT ESTIMATION")
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

    verification = verify_stage25_inputs(root, config)
    table_dir = root / output["table_dir"]
    repair_summary = load_json(
        table_dir / "s25b_97_balance_repair_summary.json"
    )
    gates = pd.read_csv(
        table_dir / "s25b_97_balance_repair_gates.csv",
        low_memory=False,
    )
    if not bool(repair_summary["all_balance_repair_gates_pass"]):
        raise RuntimeError("Balance-repair summary did not pass.")
    if not bool(gates["pass"].astype(str).str.lower().eq("true").all()):
        raise RuntimeError("Balance-repair gate table contains a failure.")

    v9_estimand = load_json(root / source["v9_estimand"])
    v9_models = load_json(root / source["v9_model_registry"])
    v9_bootstrap = load_json(root / source["v9_bootstrap_registry"])
    cohort_summary = verification["stage25_summary"]

    estimator_spec = {
        "status": "CANDIDATE_V10_ESTIMATOR_SPEC_LOCKED",
        "effect_estimated": False,
        "population": {
            "n": config["expected"]["n"],
            "treated": config["expected"]["treated"],
            "control": config["expected"]["control"],
            "events": config["expected"]["events"],
            "treated_events": config["expected"]["treated_events"],
            "control_events": config["expected"]["control_events"],
            "cohort_sha256": verification["cohort_sha256"],
            "compact_sha256": verification["compact_sha256"],
            "eligibility": (
                "HR-positive/HER2-negative day-180 landmark survivors "
                "with no documented chemotherapy initiation by day 180 "
                "and ascertainable chemotherapy start timing."
            ),
        },
        "design_repair": config["design_repair"],
        "locked_estimator": config["locked_estimator"],
        "bootstrap": {
            "repetitions": int(v9_bootstrap["n_repetitions"]),
            "base_seed": int(v9_bootstrap["bootstrap_base_seed"]),
            "primary_interval": v9_bootstrap["primary_interval"],
            "sensitivity_intervals": v9_bootstrap[
                "sensitivity_intervals"
            ],
            "propensity_refit_rule": (
                "Refit the full-sample unpenalized logistic MLE in every "
                "patient-bootstrap resample."
            ),
            "censoring_and_outcome_refit_rule": (
                "Refit the exact Candidate V9 cross-fitted nuisance models "
                "within every bootstrap resample and nuisance partition."
            ),
        },
        "v9_reference": {
            "protocol_id": v9_estimand["protocol_id"],
            "model_registry_sha256": sha256_file(
                root / source["v9_model_registry"]
            ),
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
                "Restriction to the frozen strict no-chemotherapy-by-day180 population.",
                "Replacement of the V9 cross-fitted regularized propensity nuisance "
                "with a full-sample unpenalized logistic MLE on the same 13 baseline variables.",
            ],
            "unchanged": [
                "day-180 treatment assignment",
                "730-day post-landmark RMST outcome",
                "ATO target",
                "13-variable compact adjustment set",
                "cross-fitted censoring nuisance",
                "cross-fitted bounded arm-specific RidgeCV outcome nuisance",
                "20 nuisance partition seeds",
                "G-min",
                "patient-bootstrap refitting of all nuisance models",
            ],
        },
        "population": estimator_spec["population"],
        "estimator_spec_path": str(
            estimator_path.relative_to(root)
        ).replace("\\", "/"),
        "pre_effect_balance_repair": repair_summary[
            "propensity_summary"
        ],
        "interpretation_boundary": (
            "This is a post hoc design amendment locked before Candidate V10 "
            "effect estimation. The target is the overlap population among "
            "patients without documented chemotherapy initiation during the "
            "180-day grace period. It does not generalize to patients receiving "
            "chemotherapy during that period."
        ),
    }

    lock_inputs = [
        root / "stage25b_v10_balance_repair_config.json",
        root / "scripts/_stage25b_v10_balance_utils.py",
        root / "scripts/97_stage25b_v10_balance_repair.py",
        root / "scripts/98_stage25b_lock_candidate_v10_protocol.py",
        root / "run_stage25b_candidate_v10_balance_repair.ps1",
        root / source["v10_cohort"],
        root / source["v10_compact"],
        table_dir / "s25b_97_propensity_coefficients.csv",
        table_dir / "s25b_97_repaired_ato_balance.csv",
        table_dir / "s25b_97_propensity_method_comparison.csv",
        table_dir / "s25b_97_balance_repair_gates.csv",
        table_dir / "s25b_97_balance_repair_summary.json",
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
        "locked_counts": {
            key: estimator_spec["population"][key]
            for key in (
                "n",
                "treated",
                "control",
                "events",
                "treated_events",
                "control_events",
            )
        },
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
        "The V10 effect has not been computed.\n",
        encoding="utf-8",
    )

    print("Candidate V10 protocol")
    print(json.dumps(protocol, indent=2))
    print("\nCandidate V10 manifest")
    print(json.dumps(manifest, indent=2))
    print(
        "\nPASS: balance-repaired Candidate V10 locked before effect estimation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
