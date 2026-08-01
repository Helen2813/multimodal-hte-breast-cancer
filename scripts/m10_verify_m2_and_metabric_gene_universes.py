from __future__ import annotations

import json

from _metabric_m3_utils import (
    classify_identifiers, load_config, out_dir, print_table, project_root,
    quick_sha256, read_cbio, rel, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    m2 = (root / cfg["metabric_m2_dir"]).resolve()
    raw = (root / cfg["metabric_raw_dir"]).resolve()

    print("=" * 124)
    print("METABRIC M3.10 - VERIFY M2 AND BUILD METABRIC GENE UNIVERSES")
    print("=" * 124)

    decision_path = m2 / "m09_transport_readiness_decision.csv"
    if not decision_path.exists():
        raise FileNotFoundError(f"Missing M2 decision: {decision_path}")

    import pandas as pd
    decision = pd.read_csv(decision_path)
    observed = str(decision.iloc[0]["metabric_m2_decision"])
    expected = "METABRIC_READY_FOR_MULTIMODAL_TRANSPORT_PROTOCOL_LOCK"
    if observed != expected:
        raise RuntimeError(f"M2 readiness gate failed: {observed}")

    rows = []
    gene_sets = {}

    for role in ("rna", "cna", "methylation"):
        path = raw / cfg["metabric_files"][role]
        table = read_cbio(path)
        first_col = table.columns[0]
        genes = sorted({
            str(x).strip().upper()
            for x in table[first_col].dropna().astype(str)
            if str(x).strip()
        })
        gene_sets[role] = genes
        classification = classify_identifiers(genes[:5000])
        rows.append({
            "modality": role,
            "file": rel(root, path),
            "rows": len(table),
            "first_column": first_col,
            "unique_identifiers": len(genes),
            **classification,
            "quick_sha256": quick_sha256(path),
        })

    mutation_path = raw / cfg["metabric_files"]["mutations"]
    mut = read_cbio(mutation_path)
    gene_col = next((c for c in mut.columns if c.upper() == "HUGO_SYMBOL"), mut.columns[0])
    genes = sorted({
        str(x).strip().upper()
        for x in mut[gene_col].dropna().astype(str)
        if str(x).strip()
    })
    gene_sets["mutations"] = genes
    classification = classify_identifiers(genes[:5000])
    rows.append({
        "modality": "mutations",
        "file": rel(root, mutation_path),
        "rows": len(mut),
        "first_column": gene_col,
        "unique_identifiers": len(genes),
        **classification,
        "quick_sha256": quick_sha256(mutation_path),
    })

    overlaps = []
    mods = list(gene_sets)
    for i, a in enumerate(mods):
        for b in mods[i + 1:]:
            sa, sb = set(gene_sets[a]), set(gene_sets[b])
            overlaps.append({
                "modality_a": a,
                "modality_b": b,
                "overlap_genes": len(sa & sb),
                "a_coverage": len(sa & sb) / len(sa) if sa else 0.0,
                "b_coverage": len(sa & sb) / len(sb) if sb else 0.0,
            })

    write_csv(out / "m10_metabric_gene_universes.csv", rows)
    write_csv(out / "m10_metabric_gene_universe_overlaps.csv", overlaps)
    (out / "m10_metabric_gene_sets_LOCAL_ONLY.json").write_text(
        json.dumps(gene_sets), encoding="utf-8"
    )

    print(f"M2 decision verified: {observed}")
    print("\nMETABRIC gene universes")
    print_table(
        rows,
        ["modality", "rows", "first_column", "unique_identifiers",
         "identifier_type", "gene_like_count", "ensembl_count", "quick_sha256"]
    )

    print("\nCross-modality gene overlap")
    print_table(overlaps, ["modality_a", "modality_b", "overlap_genes", "a_coverage", "b_coverage"])

    print("\nPASS: METABRIC gene universes are ready. No modeling was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
