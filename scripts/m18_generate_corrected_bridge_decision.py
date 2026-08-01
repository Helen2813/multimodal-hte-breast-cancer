from __future__ import annotations

import json

from _metabric_m3b_utils import load_cfg, output_dir, print_table, read_csv, root, write_csv


def to_int(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def main() -> int:
    project = root()
    cfg = load_cfg(project)
    out = output_dir(project, cfg)

    selected = json.loads((out / "m15_selected_tcga_table.json").read_text(encoding="utf-8"))
    bridge = {r["tcga_modality"]: r for r in read_csv(out / "m16_direct_bridge_summary.csv")}
    mapping_rows = read_csv(out / "m17_local_mapping_candidates.csv")

    rna = bridge.get("rna", {})
    cna = bridge.get("cna", {})
    mutations = bridge.get("mutations", {})
    methylation = bridge.get("methylation", {})

    rna_selected = to_int(rna.get("tcga_selected_identifiers"))
    rna_ensembl = to_int(rna.get("tcga_ensembl_ids"))
    cna_overlap = to_int(cna.get("direct_overlap"))
    mutation_overlap = to_int(mutations.get("direct_overlap"))
    cpg_count = to_int(methylation.get("tcga_cpg_ids"))
    mapping_candidates = [r for r in mapping_rows if to_int(r.get("rna_probe_hits")) >= 2]

    checks = [
        {"check": "Canonical TCGA table resolved", "observed": selected["selected_path"], "pass": True},
        {"check": "TCGA RNA selected feature set resolved", "observed": f"selected={rna_selected}; ensembl={rna_ensembl}", "pass": rna_selected > 0},
        {"check": "Local RNA mapping candidate found", "observed": len(mapping_candidates), "pass": len(mapping_candidates) > 0 or rna_ensembl == 0},
        {"check": "Direct CNA gene overlap", "observed": cna_overlap, "pass": cna_overlap >= int(cfg["minimum_direct_gene_overlap_to_proceed"])},
        {"check": "Direct mutation gene overlap", "observed": mutation_overlap, "pass": mutation_overlap >= int(cfg["minimum_direct_gene_overlap_to_proceed"])},
        {"check": "Methylation marked as gene-aggregation secondary", "observed": f"tcga_cpg_features={cpg_count}", "pass": True},
    ]

    core = checks[0]["pass"] and checks[1]["pass"]
    direct = checks[3]["pass"] and checks[4]["pass"]
    mapping = checks[2]["pass"]

    if core and direct and mapping:
        decision = "M3B_CANONICAL_FEATURES_RESOLVED_READY_FOR_MANUAL_M4_MAPPING_LOCK"
        next_step = "Review and lock the exact RNA mapping candidate path/hash, then build the TCGA-selected RNA/CNA/mutation bridge. Keep methylation secondary until a CpG-to-gene rule is fixed."
    elif core and direct:
        decision = "M3B_CANONICAL_FEATURES_RESOLVED_RNA_MAPPING_REQUIRED"
        next_step = "The TCGA selected feature table and direct CNA/mutation bridge are resolved, but RNA Ensembl IDs still need a reproducible Ensembl-to-HUGO mapping before M4."
    elif core:
        decision = "M3B_CANONICAL_FEATURES_RESOLVED_DIRECT_GENE_BRIDGE_INCOMPLETE"
        next_step = "Review TCGA CNA/mutation naming and preprocessing provenance; do not use statistical-filter result tables as patient-level feature sources."
    else:
        decision = "M3B_CANONICAL_TCGA_SCHEMA_NOT_RESOLVED"
        next_step = "Provide the exact patient-level TCGA multimodal table used by Paper B."

    template = {
        "status": "M4_BRIDGE_TEMPLATE_NOT_LOCKED",
        "canonical_tcga_table": selected,
        "scientific_boundary": "METABRIC outcomes cannot be used for source, mapping, or feature-selection decisions.",
        "paper_a_role": "external design limitation only; no exact day-180 causal replication",
        "paper_b_role": "same-disease cross-platform validation of TCGA-selected modality representations",
        "modalities": {
            "rna": {"selected_count": rna_selected, "mapping_rule": "lock Ensembl-to-HUGO mapping path/hash", "locked": False},
            "cna": {"direct_shared_count": cna_overlap, "mapping_rule": "direct gene-symbol match after prefix stripping", "locked": False},
            "mutations": {"direct_shared_count": mutation_overlap, "mapping_rule": "direct gene-symbol match with panel-awareness", "locked": False},
            "methylation": {"tcga_cpg_count": cpg_count, "mapping_rule": "secondary gene/promoter aggregation; no direct CpG equivalence", "locked": False}
        }
    }
    (out / "m18_m4_bridge_TEMPLATE.json").write_text(json.dumps(template, indent=2), encoding="utf-8")

    result = {"metabric_m3b_decision": decision, "canonical_tcga_table": selected["selected_path"], "recommended_next_step": next_step, "metabric_outcome_inspection_allowed": False}
    write_csv(out / "m18_m3b_decision_checks.csv", checks)
    write_csv(out / "m18_m3b_decision.csv", [result])
    (out / "m18_m3b_decision.md").write_text("\n".join([
        "# METABRIC M3B decision", "", f"**Decision:** `{decision}`", "",
        "## Canonical TCGA table", "", f"`{selected['selected_path']}`", "",
        "## Scientific boundary", "", "No METABRIC outcome may be used for mapping, source selection, or feature selection.", "",
        "## Recommended next step", "", next_step
    ]) + "\n", encoding="utf-8")

    print("=" * 124)
    print("METABRIC M3B.18 - CORRECTED SOURCE-BRIDGE DECISION")
    print("=" * 124)
    print(f"Decision: {decision}")
    print("\nDecision checks")
    print_table(checks, ["check", "observed", "pass"])
    print("\nRecommended next step")
    print(f"  {next_step}")
    print("\nPASS: corrected bridge decision generated. No outcome or model was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
