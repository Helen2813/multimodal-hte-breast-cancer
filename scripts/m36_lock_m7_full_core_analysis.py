from __future__ import annotations

import json
from pathlib import Path

from _metabric_m7_utils import (
    load_config,
    out_dir,
    print_table,
    project_root,
    read_rows,
    rel,
    sha256,
    write_csv,
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M7.36 - FULL CORE ANALYSIS LOCK")
    print("=" * 124)

    required = [
        root
        / cfg["metabric_m5_dir"]
        / "m30_metabric_dual_track_protocol.json",
        root / cfg["files"]["m6_engine_decision"],
        root / cfg["files"]["m6_track_a_results"],
        root / cfg["files"]["m6_track_a_paired_bootstrap"],
        root / cfg["files"]["m6_track_b_summary"],
        root / cfg["files"]["m6_decision"],
    ]
    checks = []
    for path in required:
        checks.append({
            "check": rel(root, path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else "",
            "pass": path.exists(),
        })
    if not all(row["pass"] for row in checks):
        raise RuntimeError("M7 preflight failed: required M5/M6 outputs are missing.")

    engine = json.loads(
        (root / cfg["files"]["m6_engine_decision"])
        .read_text(encoding="utf-8")
    )
    m6_summary = json.loads(
        (root / cfg["files"]["m6_track_b_summary"])
        .read_text(encoding="utf-8")
    )
    m6_decision = read_rows(
        root / cfg["files"]["m6_decision"]
    )
    expected_decision = (
        "M6_PILOT_COMPLETE_TRACK_A_READY_"
        "TRACK_B_RECONSTRUCTED_METHOD_REQUIRES_LABEL"
    )
    observed_decision = (
        m6_decision[0]["metabric_m6_decision"]
        if m6_decision else ""
    )
    checks.extend([
        {
            "check": "M6 decision",
            "exists": True,
            "sha256": "",
            "pass": observed_decision == expected_decision,
            "observed": observed_decision,
        },
        {
            "check": "Track B engine label",
            "exists": True,
            "sha256": "",
            "pass": (
                engine["status"]
                == cfg["track_b"]["historical_engine_status"]
            ),
            "observed": engine["status"],
        },
        {
            "check": "M6 Track B mutation universe",
            "exists": True,
            "sha256": "",
            "pass": (
                int(m6_summary["mutation_candidate_universe"])
                == 173
            ),
            "observed": m6_summary[
                "mutation_candidate_universe"
            ],
        },
    ])
    if not all(bool(row["pass"]) for row in checks):
        raise RuntimeError("M7 scientific preflight failed.")

    protocol = {
        "protocol_id": "",
        "status": "METABRIC_M7_FULL_CORE_ANALYSIS_LOCKED",
        "locked_after_pilot_before_full_run": True,
        "pilot_information_used_only_for_feasibility": True,
        "track_a": {
            "purpose": (
                "Full external evaluation of the fixed high-confidence "
                "TCGA-selected RNA/CNA panel in METABRIC."
            ),
            "models": cfg["track_a"]["model_sets"],
            "primary_metric": "Harrell C-index",
            "primary_incremental_contrast": (
                "clinical_rna_cna minus clinical"
            ),
            "secondary_metrics": [
                "5-year binary AUC",
                "10-year binary AUC",
                "Uno C-index truncated at 10 years",
                "IPCW cumulative/dynamic AUC at 5 and 10 years",
                "IPCW Brier score at 5 and 10 years",
                "integrated Brier score from 1 to 10 years",
                "calibration slope",
                "observed-minus-predicted survival at 5 and 10 years",
            ],
            "paired_patient_bootstrap_repetitions": (
                cfg["track_a"]["bootstrap_repetitions"]
            ),
            "paired_bootstrap_seed": (
                cfg["track_a"]["bootstrap_seed"]
            ),
            "no_metabric_refit": True,
            "no_outcome_based_feature_selection": True,
        },
        "track_b": {
            "purpose": (
                "Full repeated nested reconstructed dependency-aware "
                "Paper-1 replication in METABRIC."
            ),
            "label": (
                "reconstructed; not bitwise reproduction of historical IAMB"
            ),
            "outer_repeats": cfg["track_b"]["outer_repeats"],
            "outer_folds": cfg["track_b"]["outer_folds"],
            "repeat_seeds": list(range(
                cfg["track_b"]["repeat_seed_start"],
                cfg["track_b"]["repeat_seed_start"]
                + cfg["track_b"]["outer_repeats"],
            )),
            "engine": cfg["track_b"]["engine"],
            "iamb_alpha": cfg["track_b"]["iamb_alpha"],
            "all_supervised_steps_inside_outer_training_fold": True,
            "clinical_only_fold_matched_comparator": True,
            "algorithmic_variability_not_sampling_ci": True,
            "mutation_rule": (
                "all METABRIC_173 genes; panel-aware zero coding; "
                "nonsynonymous calls only"
            ),
        },
        "boundaries": [
            "No result threshold determines whether an analysis is retained.",
            "Track A tests transport of an assayed subset, not the entire published TCGA panel.",
            "Track B is methodological reconstruction, not exact reproduction.",
            "METABRIC does not validate the TCGA day-180 treatment-initiation estimand.",
            "Methylation and pathway-level analyses remain a later modality-specific stage.",
        ],
    }
    protocol_payload = json.dumps(
        protocol,
        sort_keys=True,
    )
    protocol_hash = __import__("hashlib").sha256(
        protocol_payload.encode("utf-8")
    ).hexdigest()
    protocol["protocol_id"] = (
        f"METABRIC_M7_{protocol_hash[:16].upper()}"
    )
    protocol_path = out / "m36_m7_full_core_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    hash_rows = []
    for path in required + [protocol_path]:
        hash_rows.append({
            "path": rel(root, path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    write_csv(out / "m36_input_hash_manifest.csv", hash_rows)
    write_csv(out / "m36_protocol_checks.csv", checks)

    print("Protocol checks")
    print_table(
        checks,
        ["check", "exists", "observed", "pass", "sha256"],
    )
    print(f"\nProtocol ID: {protocol['protocol_id']}")
    print(json.dumps(protocol, indent=2))

    print("\nPASS: M7 full core protocol locked. No model was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
