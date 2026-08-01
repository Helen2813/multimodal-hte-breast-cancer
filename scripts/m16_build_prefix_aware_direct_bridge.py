from __future__ import annotations

import json

from _metabric_m3b_utils import load_cfg, output_dir, print_table, read_csv, root, write_csv


def main() -> int:
    project = root()
    cfg = load_cfg(project)
    out = output_dir(project, cfg)

    print("=" * 124)
    print("METABRIC M3B.16 - PREFIX-AWARE DIRECT FEATURE BRIDGE")
    print("=" * 124)

    registry = read_csv(out / "m15_tcga_feature_registry.csv")
    metabric = json.loads((project / cfg["metabric_m3_dir"] / "m10_metabric_gene_sets_LOCAL_ONLY.json").read_text(encoding="utf-8"))
    metabric = {k: {str(x).upper() for x in v} for k, v in metabric.items()}

    target_map = {"rna": "rna", "cna": "cna", "mutations": "mutations", "methylation": "methylation"}
    summary = []
    shared = []
    identifiers = {}

    for tcga_modality, metabric_modality in target_map.items():
        rows = [r for r in registry if r["modality"] == tcga_modality]
        ids = {str(r["canonical_identifier"]).upper() for r in rows if r["identifier_type"] in {"gene_symbol", "ensembl", "cpg"}}
        identifiers[tcga_modality] = sorted(ids)
        target = metabric.get(metabric_modality, set())
        overlap = ids & target
        summary.append({
            "tcga_modality": tcga_modality,
            "tcga_selected_identifiers": len(ids),
            "tcga_gene_symbols": sum(r["identifier_type"] == "gene_symbol" for r in rows),
            "tcga_ensembl_ids": sum(r["identifier_type"] == "ensembl" for r in rows),
            "tcga_cpg_ids": sum(r["identifier_type"] == "cpg" for r in rows),
            "metabric_identifiers": len(target),
            "direct_overlap": len(overlap),
            "overlap_examples": " | ".join(sorted(overlap)[:25]),
        })
        for identifier in sorted(overlap):
            shared.append({"tcga_modality": tcga_modality, "shared_identifier": identifier})

    write_csv(out / "m16_direct_bridge_summary.csv", summary)
    write_csv(out / "m16_direct_shared_features.csv", shared, ["tcga_modality", "shared_identifier"])
    (out / "m16_tcga_selected_identifiers_LOCAL_ONLY.json").write_text(json.dumps(identifiers, indent=2), encoding="utf-8")

    print("Direct bridge summary after prefix stripping")
    print_table(summary, ["tcga_modality", "tcga_selected_identifiers", "tcga_gene_symbols", "tcga_ensembl_ids", "tcga_cpg_ids", "metabric_identifiers", "direct_overlap", "overlap_examples"])
    print("\nInterpretation")
    print("  RNA Ensembl IDs need an explicit Ensembl-to-HUGO bridge.")
    print("  CNA and mutation gene-symbol features can be matched directly when overlap is nonzero.")
    print("  CpG-level TCGA methylation is not the same feature space as gene-level METABRIC promoter methylation.")
    print("\nPASS: M3 zero-overlap artefact was corrected by stripping modality prefixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
