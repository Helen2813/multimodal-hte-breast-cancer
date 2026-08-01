from __future__ import annotations

import json
from pathlib import Path

from _metabric_m5_utils import (
    load_config, out_dir, print_table, project_root, read_rows, sha256,
    truth, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    recipe = json.loads((out / "m26_paper1_recipe_registry.json").read_text(encoding="utf-8"))
    panel = json.loads((out / "m27_metabric_173_summary.json").read_text(encoding="utf-8"))
    transport = read_rows(out / "m28_transportability_summary.csv")
    endpoint = json.loads((out / "m29_endpoint_resolution.json").read_text(encoding="utf-8"))

    transport_by = {row["modality"]: row for row in transport}
    rna_primary = int(float(transport_by.get("rna", {}).get("primary_transportable", 0)))
    cna_primary = int(float(transport_by.get("cna", {}).get("primary_transportable", 0)))

    checks = [
        {
            "check": "Paper-1 shared-modality recipe recovered",
            "observed": recipe["status"],
            "pass": recipe["status"] == "PAPER1_SHARED_MODALITY_RECIPE_RECOVERED",
        },
        {
            "check": "Official METABRIC_173 panel recovered",
            "observed": f"{panel['panel_gene_count']} genes; {panel['assigned_samples']} assigned samples",
            "pass": panel["panel_gene_count"] == 173 and panel["negative_coding_allowed"],
        },
        {
            "check": "Primary transportable RNA features",
            "observed": rna_primary,
            "pass": rna_primary > 0,
        },
        {
            "check": "Primary transportable CNA features",
            "observed": cna_primary,
            "pass": cna_primary > 0,
        },
        {
            "check": "Endpoint identity and cohort counts resolved",
            "observed": endpoint["endpoint_lock_ready"],
            "pass": endpoint["endpoint_lock_ready"],
        },
    ]
    all_pass = all(bool(row["pass"]) for row in checks)

    protocol = {
        "protocol_id": "",
        "status": "METABRIC_DUAL_TRACK_PROTOCOL_LOCKED" if all_pass else "METABRIC_DUAL_TRACK_PROTOCOL_NOT_LOCKED",
        "locked_before_model_fitting": True,
        "source_data": {
            "tcga_canonical_table": cfg["tcga_canonical_table"],
            "metabric_clinical_master": f"{cfg['metabric_m2_dir']}/m06_metabric_clinical_master_LOCAL_ONLY.csv",
        },
        "track_a_fixed_tcga_panel_transport": {
            "scientific_question": (
                "Do the transportable members of the fixed TCGA-selected panel retain prognostic utility "
                "in independent METABRIC data after outcome-blind identifier and platform harmonization?"
            ),
            "primary_modalities": cfg["dual_track_protocol"]["track_a_primary_modalities"],
            "primary_rna_features": rna_primary,
            "primary_cna_features": cna_primary,
            "mapping_rule": (
                "Primary accepts Ensembl mappings supported by current/GRCh37 HGNC xrefs; "
                "display-name-only mappings are sensitivity-only; ambiguous/unmapped and unassayed features are excluded."
            ),
            "mutation_rule": (
                "Secondary. Zero means wild-type only for samples assigned to METABRIC_173 "
                "and genes confirmed in the recovered 173-gene panel."
            ),
            "methylation_rule": (
                "Not part of the primary exact-feature transport because TCGA selected CpGs and METABRIC promoter-gene RRBS "
                "are different feature spaces. A later gene/promoter aggregation is secondary and separately locked."
            ),
            "protein_mirna_rule": (
                "No direct external validation because compatible METABRIC modalities are unavailable."
            ),
            "preprocessing": {
                "clinical": "imputation/scaling fitted in TCGA only; locked category handling",
                "rna_primary": (
                    "within-cohort outcome-blind percentile-rank to normal-score transform in TCGA and METABRIC; "
                    "TCGA model coefficients are then applied unchanged"
                ),
                "rna_sensitivity": "within-cohort z score",
                "cna": "retain discrete gene-level values; no outcome-based filtering",
                "feature_selection_in_metabric": False,
            },
            "endpoint": "OS",
            "external_metrics": [
                "Harrell C-index",
                "Uno C-index",
                "time-dependent AUC at prespecified horizons",
                "integrated Brier score",
                "calibration-in-the-large and calibration slope",
            ],
            "uncertainty": {
                "patient_bootstrap_repetitions": cfg["dual_track_protocol"]["bootstrap_repetitions"],
                "primary_interval": "percentile 95% patient bootstrap",
            },
        },
        "track_b_independent_paper1_replication": {
            "scientific_question": (
                "Does the dependency-aware feature-selection framework from Paper 1 recover stable, "
                "biologically concordant features and modality rankings in independent METABRIC data?"
            ),
            "modalities": ["RNA", "CNV", "Methylation", "Mutation"],
            "primary_outcome": cfg["dual_track_protocol"]["track_b_primary_outcome"],
            "sensitivity_outcome": cfg["dual_track_protocol"]["track_b_sensitivity_outcome"],
            "outer_folds": cfg["dual_track_protocol"]["outer_folds"],
            "outer_repeats": cfg["dual_track_protocol"]["outer_repeats"],
            "selection_protocol": {
                "all_supervised_filters_inside_training_fold": True,
                "all_markov_blanket_selection_inside_training_fold": True,
                "historical_paper1_grid_reproduced": True,
                "historical_best_configuration_used_as_prespecified_primary": True,
                "full_grid_inner_training_selection_reported_as_secondary": True,
                "full_cohort_selection_not_used_for_performance_claim": True,
            },
            "mutation_panel_rule": (
                "Candidates restricted to METABRIC_173 genes; panel-aware zeros only."
            ),
            "comparison_outputs": [
                "selection frequency",
                "within-modality stability",
                "Jaccard index",
                "overlap coefficient",
                "gene-level exact overlap with TCGA selections",
                "pathway-level overlap",
                "modality composition",
                "out-of-fold prognostic performance",
            ],
        },
        "joint_comparison_after_both_tracks": [
            "clinical-only",
            "clinical plus fixed TCGA-selected transportable panel",
            "clinical plus independently METABRIC-selected panel",
            "shared pathway-level representation",
        ],
        "nonnegotiable_boundaries": [
            "METABRIC is not an exact validation of the TCGA day-180 treatment-initiation estimand.",
            "Track A uses no METABRIC outcome for mapping, feature selection, or transformation choice.",
            "Track B is independent methodological replication, not fixed-panel external validation.",
            "Exact gene non-replication is interpreted jointly with platform assayability and pathway concordance.",
        ],
    }

    protocol_text = json.dumps(protocol, indent=2, sort_keys=True)
    protocol_hash = __import__("hashlib").sha256(protocol_text.encode("utf-8")).hexdigest()
    protocol["protocol_id"] = f"METABRIC_DUAL_TRACK_{protocol_hash[:16].upper()}"
    protocol_path = out / "m30_metabric_dual_track_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    input_files = [
        out / "m26_paper1_recipe_registry.json",
        out / "m26_paper1_full_grid.csv",
        out / "m27_metabric_173_summary.json",
        out / "m27_metabric_173_gene_list.csv",
        out / "m28_primary_transportable_panel.csv",
        out / "m28_mapping_fallback_sensitivity_panel.csv",
        out / "m29_endpoint_resolution.json",
        protocol_path,
    ]
    hash_rows = []
    for path in input_files:
        hash_rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    write_csv(out / "m30_protocol_hash_manifest.csv", hash_rows)
    write_csv(out / "m30_protocol_checks.csv", checks)

    decision = (
        "M5_DUAL_TRACK_PROTOCOL_LOCKED_READY_FOR_NESTED_PILOT"
        if all_pass else
        "M5_PROTOCOL_NOT_LOCKED_REVIEW_FAILED_GATES"
    )
    decision_row = {
        "metabric_m5_decision": decision,
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "all_checks_pass": all_pass,
        "primary_transportable_rna_features": rna_primary,
        "primary_transportable_cna_features": cna_primary,
        "panel_aware_mutation_ready": panel["negative_coding_allowed"],
        "paper1_recipe_ready": recipe["status"] == "PAPER1_SHARED_MODALITY_RECIPE_RECOVERED",
        "next_step": (
            "Run a small, prespecified nested pilot for Track A and Track B without changing the locked panel or recipe."
            if all_pass else
            "Resolve failed gates before fitting any model."
        ),
    }
    write_csv(out / "m30_protocol_decision.csv", [decision_row])

    print("=" * 124)
    print("METABRIC M5.30 - DUAL-TRACK PROTOCOL DECISION")
    print("=" * 124)
    print(f"Decision: {decision}")
    print(f"Protocol ID: {protocol['protocol_id']}")

    print("\nProtocol checks")
    print_table(checks, ["check", "observed", "pass"])

    print("\nLocked protocol")
    print(json.dumps(protocol, indent=2))

    print("\nHash manifest")
    print_table(hash_rows, ["path", "sha256", "size_bytes"])

    if not all_pass:
        raise RuntimeError("M5 protocol lock failed. Review the checks above.")

    print("\nPASS: dual-track protocol locked before any METABRIC modeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
