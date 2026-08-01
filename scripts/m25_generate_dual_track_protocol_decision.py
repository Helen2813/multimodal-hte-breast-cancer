from __future__ import annotations

import csv
import json
from pathlib import Path

from _metabric_m4_utils import (
    load_config, out_dir, print_table, project_root, sha256, write_csv
)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    mapping_summary = json.loads(
        (out / "m21_mapping_summary.json").read_text(encoding="utf-8")
    )
    assayability = read_rows(out / "m22_fixed_panel_assayability.csv")
    mutation_summary = json.loads(
        (out / "m23_mutation_coverage_summary.json").read_text(encoding="utf-8")
    )
    recipe_status = read_rows(out / "m24_paper1_recipe_status.csv")

    assay_by = {row["modality"]: row for row in assayability}
    rna_available = int(float(assay_by.get("rna", {}).get("available_metabric_symbols", 0)))
    cna_available = int(float(assay_by.get("cna", {}).get("available_metabric_symbols", 0)))

    shared_modalities = set(cfg["paper1_modalities_shared_with_metabric"])
    recipe_complete = (
        {row["modality"] for row in recipe_status} >= shared_modalities
        and all(
            truth(row["summary_found"])
            and truth(row["selected_lists_found"])
            and truth(row["candidate_matrices_found"])
            for row in recipe_status
            if row["modality"] in shared_modalities
        )
    )

    mutation_ready = (
        mutation_summary["mutation_transport_status"]
        == "GENE_LEVEL_WILDTYPE_CODING_ALLOWED_WITH_PANEL_AWARENESS"
    )

    checks = [
        {
            "check": "Selected TCGA Ensembl mapping produced mapped IDs",
            "observed": f"{mapping_summary['mapped_ids']}/{mapping_summary['selected_ensembl_ids']}",
            "pass": int(mapping_summary["mapped_ids"]) > 0,
        },
        {
            "check": "Fixed TCGA RNA panel has METABRIC-assayed features",
            "observed": rna_available,
            "pass": rna_available > 0,
        },
        {
            "check": "Fixed TCGA CNA panel has METABRIC-assayed features",
            "observed": cna_available,
            "pass": cna_available > 0,
        },
        {
            "check": "Mutation negative calls can be panel-aware",
            "observed": mutation_summary["mutation_transport_status"],
            "pass": mutation_ready,
        },
        {
            "check": "Paper-1 shared-modality recipe evidence recovered",
            "observed": recipe_complete,
            "pass": recipe_complete,
        },
    ]

    track_a_core_ready = checks[0]["pass"] and checks[1]["pass"] and checks[2]["pass"]
    track_b_ready = checks[4]["pass"]

    if track_a_core_ready and mutation_ready and track_b_ready:
        decision = "M4_DUAL_TRACK_READY_FOR_PROTOCOL_LOCK_AND_EXECUTION"
    elif track_a_core_ready and track_b_ready:
        decision = "M4_RNA_CNA_DUAL_TRACK_READY_MUTATION_COVERAGE_BLOCKED"
    elif track_a_core_ready:
        decision = "M4_FIXED_PANEL_CORE_READY_PAPER1_REPLICATION_RECIPE_INCOMPLETE"
    else:
        decision = "M4_HARMONIZATION_INCOMPLETE_DO_NOT_MODEL"

    protocol = {
        "status": "M4_DUAL_TRACK_PROTOCOL_DRAFT_NOT_OUTCOME_INSPECTED",
        "decision": decision,
        "scientific_scope": {
            "track_a_fixed_tcga_transport": {
                "purpose": (
                    "Test transport of the TCGA-selected panel without selecting features "
                    "or transformations using METABRIC outcomes."
                ),
                "primary_modalities": ["clinical", "RNA", "CNA"],
                "mutation_rule": (
                    "Include gene-level mutation features only when sample-panel gene coverage is resolved."
                ),
                "methylation_rule": (
                    "Secondary only: TCGA CpG features and METABRIC promoter-gene features "
                    "require an explicitly locked gene/promoter aggregation."
                ),
                "protein_and_mirna": (
                    "Not directly externally validated because compatible METABRIC modalities are absent."
                ),
                "metabric_outcome_use_for_mapping": False,
            },
            "track_b_independent_metabric_paper1_replication": {
                "purpose": (
                    "Repeat the Paper-1 dependency-aware feature-selection framework independently "
                    "inside METABRIC and compare gene, pathway, stability, and modality-level replication."
                ),
                "shared_modalities": cfg["paper1_modalities_shared_with_metabric"],
                "primary_outcome": cfg["independent_replication_protocol"]["primary_outcome"],
                "sensitivity_outcome": cfg["independent_replication_protocol"]["sensitivity_outcome"],
                "outer_folds": cfg["independent_replication_protocol"]["outer_folds"],
                "outer_repeats": cfg["independent_replication_protocol"]["outer_repeats"],
                "all_supervised_screening_and_mb_selection_inside_training_folds": True,
                "full_dataset_selection_used_for_performance_claims": False,
                "reporting": [
                    "selection frequency",
                    "Jaccard and overlap coefficient",
                    "gene-level overlap",
                    "pathway-level overlap",
                    "modality composition",
                    "out-of-fold prognostic performance",
                ],
            },
        },
        "comparison_models_after_protocol_lock": [
            "clinical-only",
            "clinical + fixed TCGA-selected transportable panel",
            "clinical + independently METABRIC-selected panel",
            "shared pathway representation",
        ],
        "nonnegotiable_boundaries": [
            "METABRIC cannot validate the locked TCGA day-180 treatment-initiation estimand.",
            "Track A mapping and transformations are outcome-blind.",
            "Track B is independent methodological replication, not external validation of the fixed TCGA panel.",
            "The positive-bootstrap fraction from Paper A is unrelated to METABRIC feature selection.",
        ],
    }

    protocol_path = out / "m25_dual_track_protocol_DRAFT.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    locked_inputs = [
        out / "m20_selected_tcga_feature_identifiers.csv",
        out / "m21_ensembl_to_hgnc_mapping.csv",
        out / "m22_fixed_panel_feature_map.csv",
        out / "m22_fixed_panel_assayability.csv",
        out / "m23_mutation_coverage_summary.json",
        out / "m24_paper1_recipe_status.csv",
        protocol_path,
    ]
    hash_rows = []
    for path in locked_inputs:
        if path.exists():
            hash_rows.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            })
    write_csv(out / "m25_dual_track_draft_hashes.csv", hash_rows)
    write_csv(out / "m25_decision_checks.csv", checks)
    write_csv(out / "m25_decision.csv", [{
        "metabric_m4_decision": decision,
        "track_a_core_ready": track_a_core_ready,
        "track_b_recipe_ready": track_b_ready,
        "mutation_panel_aware_ready": mutation_ready,
        "protocol_status": protocol["status"],
        "next_step": (
            "Review this log. If RNA/CNA mapping and Paper-1 recipe gates pass, "
            "the next stage will lock the dual-track protocol and run outcome-blind "
            "matrix preprocessing before nested METABRIC feature selection."
        ),
    }])

    print("=" * 124)
    print("METABRIC M4.25 - DUAL-TRACK DECISION")
    print("=" * 124)
    print(f"Decision: {decision}")

    print("\nDecision checks")
    print_table(checks, ["check", "observed", "pass"])

    print("\nDraft protocol")
    print(json.dumps(protocol, indent=2))

    print("\nDraft hash registry")
    print_table(hash_rows, ["path", "sha256", "size_bytes"])

    print("\nNo METABRIC outcome was inspected for Track A.")
    print("Track B selection has not yet been executed.")
    print("\nPASS: M4 dual-track preparation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
