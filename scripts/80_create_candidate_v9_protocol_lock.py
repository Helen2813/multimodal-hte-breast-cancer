#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _stage12_utils import project_paths
from _stage20_utils import (
    canonical_sha256,
    dataframe_console,
    ensure_dirs,
    final_paths,
    load_stage20_config,
    markdown_table,
    project_root,
    refuse_existing_lock,
    sha256_file,
    write_csv,
    write_json,
    write_text,
)


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def required_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    refuse_existing_lock(root)
    cfg = load_stage20_config(root)
    lock_cfg = cfg["lock_requirements"]
    est = cfg["final_estimator"]
    boot = cfg["publication_bootstrap"]
    tables = root / "results/tables"
    outputs = final_paths(root)

    stage19 = pd.read_csv(required_path(root, "results/tables/78_stage19_decision.csv"), low_memory=False)
    checks = pd.read_csv(required_path(root, "results/tables/78_stage19_decision_checks.csv"), low_memory=False)
    point = pd.read_csv(required_path(root, "results/tables/79_candidate_v9_final_point_estimate.csv"), low_memory=False)
    prefixes = pd.read_csv(required_path(root, "results/tables/79_candidate_v9_prefix_convergence.csv"), low_memory=False)
    if len(stage19) != 1 or len(point) != 1:
        raise RuntimeError("Expected exactly one Stage 19 decision row and one Stage 79 point-estimate row.")
    s19 = stage19.iloc[0]
    p = point.iloc[0]

    preflight = pd.DataFrame([
        {
            "check": "stage19_decision",
            "observed": str(s19["stage19_decision"]),
            "expected": lock_cfg["required_stage19_decision"],
            "pass": str(s19["stage19_decision"]) == str(lock_cfg["required_stage19_decision"]),
        },
        {
            "check": "stage19_full_bootstrap_authorized_after_lock",
            "observed": bool_value(s19["full_bootstrap_authorized_after_protocol_lock"]),
            "expected": bool(lock_cfg["required_stage19_authorization"]),
            "pass": bool_value(s19["full_bootstrap_authorized_after_protocol_lock"]) == bool(lock_cfg["required_stage19_authorization"]),
        },
        {
            "check": "stage19_all_gates_pass",
            "observed": float(pd.Series(checks["pass"]).map(bool_value).mean()),
            "expected": float(lock_cfg["required_stage19_gate_pass_fraction"]),
            "pass": float(pd.Series(checks["pass"]).map(bool_value).mean()) == float(lock_cfg["required_stage19_gate_pass_fraction"]),
        },
        {
            "check": "final_point_partitions",
            "observed": int(p["partitions"]),
            "expected": 20,
            "pass": int(p["partitions"]) == 20,
        },
        {
            "check": "final_point_n",
            "observed": int(p["n"]),
            "expected": int(est["expected_n"]),
            "pass": int(p["n"]) == int(est["expected_n"]),
        },
        {
            "check": "stage20_stage21_partition_seeds_identical",
            "observed": est["partition_base_seeds"],
            "expected": boot["partition_base_seeds"],
            "pass": list(est["partition_base_seeds"]) == list(boot["partition_base_seeds"]),
        },
    ])
    if not bool(preflight["pass"].all()):
        raise RuntimeError("Candidate V9 lock preflight failed.\n" + dataframe_console(preflight))

    estimand_json = {
        "protocol_status": cfg["protocol_status_after_lock"],
        "disclosure": cfg["disclosure"],
        **est,
        "locked_point_estimate_days": float(p["estimate_days"]),
        "locked_point_if_se_days": float(p["if_se_days"]),
        "locked_point_diagnostic_if_ci_low_days": float(p["if_ci_low_days"]),
        "locked_point_diagnostic_if_ci_high_days": float(p["if_ci_high_days"]),
    }
    model_json = {
        "protocol_status": cfg["protocol_status_after_lock"],
        "propensity": est["propensity_specification"],
        "censoring": est["censoring_model"],
        "outcome": est["outcome_model"],
        "g_min": est["primary_g_min"],
        "time_interval_days": est["interval_days"],
        "horizon_days": est["horizon_days"],
        "n_folds": est["n_folds"],
        "aggregation": est["aggregation"],
        "partition_base_seeds": est["partition_base_seeds"],
        "maximum_partition_retries": est["maximum_partition_retries"],
        "model_selection_rule": "No learner, tuning, truncation, seed, or outcome-bound choice may be changed after protocol lock.",
    }
    bootstrap_json = {
        "protocol_status": cfg["protocol_status_after_lock"],
        **boot,
    }

    source_paths = project_paths()
    relative_files: list[str] = [
        str(source_paths["landmark_cohort"].relative_to(root)),
        str(source_paths["landmark_compact"].relative_to(root)),
        str(source_paths["landmark_splits"].relative_to(root)),
        str(source_paths["landmark_weights"].relative_to(root)),
        str(source_paths["candidate"].relative_to(root)),
        "stage20_config.json",
        "stage21_config.json",
        "scripts/_common.py",
        "scripts/_stage12_utils.py",
        "scripts/_stage16_utils.py",
        "scripts/_stage17_utils.py",
        "scripts/_stage18_utils.py",
        "scripts/_stage19_utils.py",
        "scripts/_stage20_utils.py",
        "scripts/79_final_20_partition_point_estimate.py",
        "scripts/80_create_candidate_v9_protocol_lock.py",
        "scripts/81_verify_candidate_v9_protocol_lock.py",
        "scripts/_stage21_utils.py",
        "scripts/82_full_publication_bootstrap.py",
        "scripts/83_summarize_publication_bootstrap.py",
        "scripts/84_generate_publication_decision.py",
        "run_stage20_candidate_v9_protocol_lock.ps1",
        "run_stage21_publication_bootstrap.ps1",
        "results/tables/77_stage19_stabilization_summary.csv",
        "results/tables/78_stage19_decision_checks.csv",
        "results/tables/78_stage19_decision.csv",
        "results/tables/79_candidate_v9_preflight_checks.csv",
        "results/tables/79_candidate_v9_partition_estimates.csv",
        "results/tables/79_candidate_v9_prefix_convergence.csv",
        "results/tables/79_candidate_v9_final_point_estimate.csv",
    ]
    if (root / "requirements-lock.txt").exists():
        relative_files.append("requirements-lock.txt")

    core_hashes = []
    for rel in relative_files:
        path = required_path(root, rel)
        core_hashes.append({"path": rel.replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    protocol_payload = {
        "final_estimator": estimand_json,
        "model_registry": model_json,
        "bootstrap_registry": bootstrap_json,
        "core_hashes": core_hashes,
    }
    protocol_hash = canonical_sha256(protocol_payload)
    protocol_id = f"PAPER_A_CANDIDATE_V9_{protocol_hash[:16].upper()}"

    estimand_json["protocol_id"] = protocol_id
    model_json["protocol_id"] = protocol_id
    bootstrap_json["protocol_id"] = protocol_id
    write_json(estimand_json, outputs["primary_estimand"])
    write_json(model_json, outputs["model_registry"])
    write_json(bootstrap_json, outputs["bootstrap_registry"])

    plan = f"""# Paper A final locked analysis plan: Candidate V9

**Protocol ID:** `{protocol_id}`  
**Status:** `{cfg['protocol_status_after_lock']}`  
**Disclosure:** {cfg['disclosure']}

## Scientific question

Among verified HR-positive/HER2-negative patients who are alive and eligible at the day-180 landmark, what is the overlap-population difference in subsequent 730-day restricted mean survival time between hormone-therapy initiation during days 0-180 and no initiation by day 180?

## Primary estimand

- Population: {est['eligibility']}.
- Treatment contrast: {est['treatment_contrast']}.
- Target population: treatment-overlap population (ATO).
- Outcome: {est['horizon_days']:.0f}-day post-landmark RMST.
- Effect scale: treated minus control, in days.
- Locked point estimate before publication bootstrap: {float(p['estimate_days']):.6f} days.
- Influence-function interval at the locked point estimate is diagnostic; the primary final interval is the patient-bootstrap percentile interval.

## Locked estimator

- Five-fold cross-fitting.
- Twenty fixed nuisance partitions with seeds: {', '.join(str(x) for x in est['partition_base_seeds'])}.
- Propensity: {est['propensity_specification']}.
- Censoring: {est['censoring_model']}.
- Outcome nuisance: {est['outcome_model']}.
- Censoring survival floor: G-min = {est['primary_g_min']:.2f}.
- Discrete-time interval: {est['interval_days']:.0f} days.
- Patient-level repeated-score aggregation across all 20 partitions.

## Locked publication bootstrap

- {boot['n_repetitions']} ordinary patient bootstrap repetitions.
- Bootstrap base seed: {boot['bootstrap_base_seed']}.
- Every bootstrap copy of the same patient remains in one nuisance fold.
- All propensity, censoring, and outcome nuisance models are refitted within every bootstrap partition.
- Primary interval: {boot['primary_interval']}.
- Sensitivity intervals: {', '.join(boot['sensitivity_intervals'])}.
- No change to treatment definition, cohort, horizon, G-min, learners, partitions, seeds, bounding, or interval method is permitted after inspection of the publication-bootstrap distribution.

## Identification assumptions

1. Consistency of the recorded treatment strategy.
2. Conditional exchangeability after the locked baseline adjustment set.
3. Positivity in the overlap population.
4. Conditional independent censoring given treatment and locked baseline covariates.
5. Valid source classification of receptor status, treatment, event, and baseline fields.

## Interpretation boundary

The primary result is an observational ATO RMST contrast, not an unconditional efficacy estimate and not a formal randomized-trial result. Treatment initiation timing is reconstructed from recorded clinical fields, and residual confounding remains possible.

## Stage 19 stabilization evidence

{markdown_table(stage19)}

## Candidate V9 prefix convergence on the original cohort

{markdown_table(prefixes)}
"""
    write_text(plan, outputs["analysis_plan"])

    docs = [
        outputs["analysis_plan"], outputs["primary_estimand"],
        outputs["model_registry"], outputs["bootstrap_registry"],
    ]
    locked_files = core_hashes + [
        {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in docs
    ]
    manifest = {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_hash,
        "protocol_status": cfg["protocol_status_after_lock"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": cfg["disclosure"],
        "stage19_decision": str(s19["stage19_decision"]),
        "stage19_all_gates_passed": bool(pd.Series(checks["pass"]).map(bool_value).all()),
        "locked_point_estimate_days": float(p["estimate_days"]),
        "locked_point_if_se_days": float(p["if_se_days"]),
        "publication_bootstrap_repetitions": int(boot["n_repetitions"]),
        "partition_base_seeds": list(est["partition_base_seeds"]),
        "locked_files": locked_files,
    }
    write_json(manifest, outputs["hash_manifest"])
    sentinel = f"""PROTOCOL LOCKED
Protocol ID: {protocol_id}
Status: {cfg['protocol_status_after_lock']}
Created UTC: {manifest['created_utc']}

Do not edit any file listed in:
{outputs['hash_manifest'].relative_to(root)}

Run the integrity checker before and during the publication bootstrap.
"""
    write_text(sentinel, outputs["lock_sentinel"])

    summary = pd.DataFrame([{
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_hash,
        "protocol_status": cfg["protocol_status_after_lock"],
        "locked_point_estimate_days": float(p["estimate_days"]),
        "locked_point_if_se_days": float(p["if_se_days"]),
        "partition_count": len(est["partition_base_seeds"]),
        "publication_bootstrap_repetitions": int(boot["n_repetitions"]),
        "locked_file_count": len(locked_files),
        "manifest_path": str(outputs["hash_manifest"].relative_to(root)),
    }])
    write_csv(preflight, tables / "80_candidate_v9_lock_preflight.csv")
    write_csv(summary, tables / "80_candidate_v9_protocol_lock_summary.csv")

    print("=" * 124)
    print("STAGE 80 - CANDIDATE V9 PROTOCOL LOCK CREATED")
    print("=" * 124)
    print("Lock preflight")
    print(dataframe_console(preflight))
    print("\nProtocol lock summary")
    print(dataframe_console(summary))
    print("\nFinal analysis plan")
    print(plan)
    print("\nLocked file hashes")
    print(dataframe_console(pd.DataFrame(locked_files)))
    print("\nThe final protocol files are write-once. This script will refuse to overwrite them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
