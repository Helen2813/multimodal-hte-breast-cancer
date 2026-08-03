#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from _stage25_v10_utils import (
    canonical_sha256,
    dataframe_console,
    load_config,
    load_json,
    project_root,
    read_csv,
    refuse_existing_v10_lock,
    sha256_file,
    verify_v9_lock,
    write_json,
    write_text,
)


def main() -> int:
    root = project_root()
    config = load_config(root)
    output = config["output"]

    print("=" * 128)
    print("STAGE 96 - LOCK CANDIDATE V10 BEFORE EFFECT ESTIMATION")
    print("=" * 128)

    refuse_existing_v10_lock(root, config)
    v9_integrity = verify_v9_lock(root, config)

    cohort_summary = load_json(
        root
        / output["table_dir"]
        / "s25_94_v10_cohort_summary.json"
    )
    pre_effect = load_json(
        root
        / output["table_dir"]
        / "s25_95_pre_effect_summary.json"
    )
    gates = read_csv(
        root
        / output["table_dir"]
        / "s25_95_pre_effect_gates.csv"
    )

    if not bool(pre_effect["all_pre_effect_gates_pass"]):
        raise RuntimeError(
            "Candidate V10 pre-effect gates did not pass."
        )
    if not bool(gates["pass"].astype(str).str.lower().eq("true").all()):
        raise RuntimeError(
            "Candidate V10 gate table contains a failure."
        )

    v9_estimand = load_json(
        root / config["source"]["v9_estimand"]
    )
    v9_models = load_json(
        root / config["source"]["v9_model_registry"]
    )
    v9_bootstrap = load_json(
        root / config["source"]["v9_bootstrap_registry"]
    )

    cohort_path = root / output["cohort"]
    compact_path = root / output["compact"]

    protocol = {
        "protocol_id": "",
        "status": config["status_after_lock"],
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "disclosure": config["disclosure"],
        "amendment_reason": (
            "Candidate V9 showed severe chemotherapy-sequencing "
            "imbalance. Chemotherapy was initiated by day 180 in "
            "60.5% of controls versus 26.3% of early hormone "
            "initiators, and known ongoing chemotherapy at day 180 "
            "occurred in 29.3% versus 2.1%, respectively."
        ),
        "relationship_to_v9": {
            "v9_protocol_id": v9_estimand["protocol_id"],
            "v9_remains_immutable": True,
            "v9_role": (
                "Historical locked full-cohort analysis and "
                "secondary sequencing-confounded contrast."
            ),
            "v10_effect_has_not_been_computed": True,
        },
        "population": {
            **config["candidate_v10_population"],
            "n": cohort_summary["candidate_v10_n"],
            "treated": cohort_summary[
                "candidate_v10_treated"
            ],
            "control": cohort_summary[
                "candidate_v10_control"
            ],
            "events": cohort_summary[
                "candidate_v10_events"
            ],
            "treated_events": cohort_summary[
                "candidate_v10_treated_events"
            ],
            "control_events": cohort_summary[
                "candidate_v10_control_events"
            ],
            "features": cohort_summary[
                "candidate_v10_features"
            ],
            "cohort_sha256": sha256_file(cohort_path),
            "compact_sha256": sha256_file(compact_path),
        },
        "estimator": {
            **config["estimator"],
            "horizon_days": config["timing"][
                "horizon_days"
            ],
            "interval_days": config["timing"][
                "interval_days"
            ],
            "changed_from_v9": (
                "Population restriction only. Treatment, outcome, "
                "estimand, nuisance learners, folds, partition seeds, "
                "G-min, outcome bounding, and repeated-score "
                "aggregation remain unchanged."
            ),
            "v9_model_registry_sha256": sha256_file(
                root / config["source"]["v9_model_registry"]
            ),
        },
        "planned_inference": {
            "point_estimate": (
                "Twenty locked nuisance partitions with exact "
                "Candidate V9 estimator."
            ),
            "publication_bootstrap": (
                "Ordinary patient bootstrap with all nuisance "
                "models refitted in every partition."
            ),
            "bootstrap_repetitions": int(
                v9_bootstrap["n_repetitions"]
            ),
            "bootstrap_base_seed": int(
                v9_bootstrap["bootstrap_base_seed"]
            ),
            "primary_interval": (
                v9_bootstrap["primary_interval"]
            ),
            "sensitivity_intervals": (
                v9_bootstrap["sensitivity_intervals"]
            ),
            "point_estimate_code_locked_before_effect": True,
            "bootstrap_not_authorized_until": (
                "Candidate V10 point-estimate runner passes "
                "integrity and numerical diagnostics."
            ),
        },
        "pre_effect_diagnostics": pre_effect,
        "interpretation_boundary": (
            "Candidate V10 estimates an observational overlap-"
            "population contrast among patients without documented "
            "chemotherapy initiation by day 180 and with ascertainable "
            "chemotherapy start timing. It is not a randomized-trial "
            "effect and does not generalize to patients receiving "
            "chemotherapy during the grace period."
        ),
    }

    hash_inputs = [
        root / "stage25_v10_config.json",
        root / "scripts/_stage25_v10_utils.py",
        root / "scripts/94_stage25_build_candidate_v10_cohort.py",
        root / "scripts/95_stage25_v10_pre_effect_gates.py",
        root / "scripts/96_stage25_lock_candidate_v10_protocol.py",
        root / "scripts/_stage26_v10_utils.py",
        root / "scripts/97_stage26_candidate_v10_point_estimate.py",
        root / "run_stage25_candidate_v10_protocol_lock.ps1",
        root / "run_stage26_candidate_v10_point_estimate.ps1",
        cohort_path,
        compact_path,
        root
        / output["table_dir"]
        / "s25_94_v10_cohort_summary.json",
        root
        / output["table_dir"]
        / "s25_95_pre_effect_summary.json",
        root
        / output["table_dir"]
        / "s25_95_pre_effect_gates.csv",
    ]

    hash_rows = [
        {
            "path": str(path.relative_to(root)).replace(
                "\\",
                "/",
            ),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in hash_inputs
    ]

    payload = {
        "protocol": protocol,
        "locked_inputs": hash_rows,
        "v9_integrity_hash": canonical_sha256(
            v9_integrity[
                ["path", "expected_sha256"]
            ].to_dict("records")
        ),
    }
    protocol_hash = canonical_sha256(payload)
    protocol_id = (
        "PAPER_A_CANDIDATE_V10_"
        + protocol_hash[:16].upper()
    )
    protocol["protocol_id"] = protocol_id

    protocol_path = root / output["protocol"]
    write_json(protocol, protocol_path)

    analysis_plan = f"""# Candidate V10 protocol amendment

**Protocol ID:** `{protocol_id}`  
**Status:** `{config['status_after_lock']}`

## Disclosure

{config['disclosure']}

## Reason for amendment

Candidate V9 remains immutable. A diagnostic audit conducted after Candidate V9
identified severe treatment-sequencing imbalance: many patients classified as not
initiating hormone therapy by day 180 were still receiving chemotherapy during the
grace period.

## Candidate V10 population

{config['candidate_v10_population']['eligibility']}

Selection rule:

{config['candidate_v10_population']['selection_rule']}

Future-treatment rule:

{config['candidate_v10_population']['future_treatment_rule']}

Locked counts before effect estimation:

- n = {protocol['population']['n']}
- early hormone initiation = {protocol['population']['treated']}
- no hormone initiation by day 180 = {protocol['population']['control']}
- post-landmark events = {protocol['population']['events']}
- compact baseline features = {protocol['population']['features']}

## Locked estimand

{config['candidate_v10_population']['estimand']}

## Locked estimator

Candidate V10 changes only the eligible population. It retains the exact Candidate
V9 treatment definition, outcome, horizon, ATO target, nuisance learners, five-fold
cross-fitting, G-min, 20 partition seeds, outcome bounding, and patient-level
repeated-score aggregation. The Candidate V10 point-estimate implementation and
runner are included in this lock before the V10 effect is computed.

## Pre-effect diagnostics

- Maximum absolute ATO-weighted SMD:
  {pre_effect['max_abs_ato_weighted_smd']:.6f}
- Treated ATO ESS fraction:
  {pre_effect['ato_ess_fraction_treated']:.6f}
- Control ATO ESS fraction:
  {pre_effect['ato_ess_fraction_control']:.6f}
- Fraction with repeated-mean propensity below 0.05:
  {pre_effect['fraction_propensity_below_0_05']:.6f}
- Fraction with repeated-mean propensity above 0.95:
  {pre_effect['fraction_propensity_above_0_95']:.6f}

All pre-effect gates passed before the Candidate V10 effect was computed.

## Interpretation boundary

{protocol['interpretation_boundary']}
"""
    plan_path = root / output["analysis_plan"]
    write_text(analysis_plan, plan_path)

    locked_files = hash_rows + [
        {
            "path": str(
                protocol_path.relative_to(root)
            ).replace("\\", "/"),
            "sha256": sha256_file(protocol_path),
            "size_bytes": protocol_path.stat().st_size,
        },
        {
            "path": str(
                plan_path.relative_to(root)
            ).replace("\\", "/"),
            "sha256": sha256_file(plan_path),
            "size_bytes": plan_path.stat().st_size,
        },
    ]

    manifest = {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_hash,
        "status": config["status_after_lock"],
        "created_utc": protocol["created_utc"],
        "candidate_v10_effect_computed": False,
        "candidate_v9_protocol_id": (
            v9_estimand["protocol_id"]
        ),
        "locked_counts": {
            key: protocol["population"][key]
            for key in (
                "n",
                "treated",
                "control",
                "events",
                "treated_events",
                "control_events",
                "features",
            )
        },
        "locked_files": locked_files,
    }
    manifest_path = root / output["manifest"]
    write_json(manifest, manifest_path)

    sentinel = (
        "CANDIDATE V10 PROTOCOL LOCKED BEFORE EFFECT ESTIMATION\n"
        f"Protocol ID: {protocol_id}\n"
        f"Created UTC: {protocol['created_utc']}\n"
        "Candidate V9 remains immutable.\n"
        "Do not edit any file listed in the Candidate V10 manifest.\n"
    )
    write_text(sentinel, root / output["sentinel"])

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
