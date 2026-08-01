from __future__ import annotations

import json

from _metabric_m8_utils import (
    load_config, out_dir, print_table, project_root, read_rows, rel, sha256, write_csv,
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    print("=" * 124)
    print("METABRIC M8.40 - MODALITY AND CONCORDANCE PROTOCOL LOCK")
    print("=" * 124)

    required = [
        root / cfg["files"]["m7_protocol"],
        root / cfg["files"]["m7_track_a_results"],
        root / cfg["files"]["m7_track_a_deltas"],
        root / cfg["files"]["m7_track_b_repeats"],
        root / cfg["files"]["m7_track_b_summary"],
        root / cfg["files"]["m7_selection_frequency"],
        root / cfg["files"]["m7_decision"],
        root / cfg["files"]["ensembl_mapping"],
    ]
    checks = [{"check": rel(root, path), "observed": sha256(path) if path.exists() else "", "pass": path.exists()} for path in required]
    decisions = read_rows(root / cfg["files"]["m7_decision"])
    observed = decisions[0]["metabric_m7_decision"] if decisions else ""
    checks.append({
        "check": "M7 completion decision",
        "observed": observed,
        "pass": observed == "M7_FULL_CORE_ANALYSIS_COMPLETE_TRACK_B_RECONSTRUCTED",
    })
    if not all(bool(row["pass"]) for row in checks):
        raise RuntimeError("M8 protocol preflight failed")

    settings = cfg["modality_analysis"]
    protocol = {
        "protocol_id": "",
        "status": "METABRIC_M8_MODALITY_CONCORDANCE_PROTOCOL_LOCKED",
        "locked_before_modality_specific_results": True,
        "purpose": "Separate incremental prognostic performance from biological reproducibility.",
        "modality_specific_analysis": {
            "modalities": ["RNA", "CNV", "Methylation", "Mutation"],
            "outer_repeats": settings["outer_repeats"],
            "outer_folds": settings["outer_folds"],
            "repeat_seeds": list(range(settings["repeat_seed_start"], settings["repeat_seed_start"] + settings["outer_repeats"])),
            "continuous_candidate_rule": "Training-fold top 100 positive plus top 100 negative Spearman associations with OS time.",
            "mutation_candidate_rule": "METABRIC_173 nonsynonymous genes with training-fold frequency >=1%, capped at 200.",
            "historical_alpha": settings["historical_alpha"],
            "engine": settings["engine"],
            "label": "reconstructed dependency-aware analysis; not bitwise historical IAMB reproduction",
            "models": ["clinical-only", "selected modality-only", "clinical plus selected modality"],
            "all_supervised_steps_inside_training_fold": True,
            "repeat_quantiles_are_algorithmic_variability_not_confidence_intervals": True,
        },
        "gene_concordance": {
            "tcga_reference": "historical Paper-1 selected modality panels",
            "metabric_core_frequency_threshold": settings["core_selection_frequency"],
            "recurrent_rule": f"selected in at least {settings['stable_folds_within_repeat']} of 5 folds in at least {settings['minimum_stable_repeats']} repeats",
            "assayability_denominator_required": True,
            "mutation_denominator": "METABRIC_173",
            "methylation_requires_probe_to_gene_annotation": True,
        },
        "pathway_concordance": {
            "database": "Reactome GMT",
            "download_url": cfg["reactome"]["url"],
            "cache_and_hash": True,
            "background": "modality-specific assayed gene universe",
            "primary_summary": "Spearman concordance of pathway enrichment scores",
            "secondary_summary": f"Jaccard of top {cfg['reactome']['top_pathways_for_concordance']} pathways",
            "pathways_do_not_modify_models": True,
        },
        "boundaries": [
            "M8 is not designed to rescue negative M7 performance results.",
            "Stable selection is not evidence of causality.",
            "Pathway concordance is biological support, not incremental clinical utility.",
            "Performance results are retained regardless of direction.",
        ],
    }
    payload = json.dumps(protocol, sort_keys=True)
    protocol["protocol_id"] = "METABRIC_M8_" + __import__("hashlib").sha256(payload.encode()).hexdigest()[:16].upper()
    protocol_path = out / "m40_m8_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    write_csv(out / "m40_protocol_checks.csv", checks)
    write_csv(out / "m40_input_hash_manifest.csv", [
        {"path": rel(root, path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in required + [protocol_path]
    ])
    print("Protocol checks")
    print_table(checks, ["check", "observed", "pass"])
    print(f"\nProtocol ID: {protocol['protocol_id']}")
    print(json.dumps(protocol, indent=2))
    print("\nPASS: M8 protocol locked. No new model was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
