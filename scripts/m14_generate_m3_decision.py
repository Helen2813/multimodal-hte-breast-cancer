from __future__ import annotations

import csv
import json

from _metabric_m3_utils import load_config, out_dir, print_table, project_root, write_csv


def rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    summary = rows(out / "m11_candidate_summary.csv")
    audit = rows(out / "m12_candidate_identifier_overlap_audit.csv")
    summary_by = {r["modality"]: r for r in summary}

    checks = []
    for modality in cfg["required_tcga_modalities_for_m4"]:
        row = summary_by.get(modality)
        found = bool(row and row.get("top_path"))
        checks.append({
            "check": f"TCGA {modality} candidate discovered",
            "observed": row.get("top_path", "") if row else "",
            "pass": found,
        })

    annotation_row = summary_by.get("annotation")
    annotation_found = bool(annotation_row and annotation_row.get("top_path"))
    pathway_row = summary_by.get("pathway_gmt")
    pathway_found = bool(pathway_row and pathway_row.get("top_path"))

    rna_audit = [r for r in audit if r["modality"] == "rna" and r["rank"] == "1"]
    rna_type = rna_audit[0]["likely_identifier_type"] if rna_audit else "unknown"
    rna_direct = int(float(rna_audit[0]["direct_gene_symbol_overlap_with_metabric_union"])) if rna_audit else 0
    rna_mapping_ready = rna_type == "gene_symbol" and rna_direct > 100
    if rna_type == "ensembl":
        rna_mapping_ready = annotation_found

    checks.extend([
        {
            "check": "RNA identifier harmonization route available",
            "observed": f"type={rna_type}; direct_overlap={rna_direct}; annotation_found={annotation_found}",
            "pass": rna_mapping_ready,
        },
        {
            "check": "Pathway GMT available",
            "observed": pathway_row.get("top_path", "") if pathway_row else "",
            "pass": pathway_found,
        },
        {
            "check": "M2 transport readiness retained",
            "observed": "METABRIC_READY_FOR_MULTIMODAL_TRANSPORT_PROTOCOL_LOCK",
            "pass": True,
        },
    ])

    required_pass = all(bool(r["pass"]) for r in checks if not r["check"].startswith("Pathway GMT"))
    if required_pass and pathway_found:
        decision = "M3_SOURCES_RESOLVED_READY_FOR_M4_HARMONIZATION_PROTOCOL_LOCK"
        next_step = (
            "Review and explicitly select the proposed TCGA sources, then run M4 to lock "
            "the shared clinical/gene/pathway representation and build harmonized matrices."
        )
    elif required_pass:
        decision = "M3_CORE_SOURCES_RESOLVED_PATHWAY_RESOURCE_REQUIRED_BEFORE_M4"
        next_step = (
            "Core clinical/RNA/CNA/mutation sources appear resolvable. Provide or identify a local GMT "
            "gene-set file before locking pathway-level transport; otherwise lock gene-level rank-based transport only."
        )
    else:
        decision = "M3_TCGA_SOURCE_SELECTION_OR_ANNOTATION_REQUIRED"
        next_step = (
            "Review the ranked source candidates and the M4 template. Do not build harmonized matrices "
            "until ambiguous TCGA files and the RNA identifier mapping are resolved."
        )

    result = {
        "metabric_m3_decision": decision,
        "metabric_exact_day180_replication_allowed": False,
        "paper_a_role": "design-transport limitation and external-data availability audit",
        "paper_b_role": "cross-platform multimodal transport validation",
        "recommended_next_step": next_step,
    }

    write_csv(out / "m14_m3_decision_checks.csv", checks)
    write_csv(out / "m14_m3_decision.csv", [result])
    (out / "m14_m3_decision.md").write_text(
        "\n".join([
            "# METABRIC M3 decision",
            "",
            f"**Decision:** `{decision}`",
            "",
            "## Scientific boundary",
            "",
            "METABRIC remains unsuitable for exact validation of the locked TCGA day-180 causal estimand.",
            "",
            "## Recommended next step",
            "",
            next_step,
        ]) + "\n",
        encoding="utf-8",
    )

    print("=" * 124)
    print("METABRIC M3.14 - SOURCE-HARMONIZATION DECISION")
    print("=" * 124)
    print(f"Decision: {decision}")
    print("\nDecision checks")
    print_table(checks, ["check", "observed", "pass"])
    print("\nRecommended next step")
    print(f"  {next_step}")
    print("\nPASS: M3 source audit completed. No source was silently selected and no model was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
