from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats

from _metabric_m8_utils import (
    benjamini_hochberg,
    download_with_retries,
    jaccard,
    load_config,
    load_reactome_gmt,
    out_dir,
    overlap_coefficient,
    print_table,
    project_root,
    read_rows,
    sha256,
    write_csv,
)


def enrichment_rows(modality, cohort, selected, universe, pathways, settings):
    selected = set(selected) & set(universe)
    universe = set(universe)
    if len(selected) < int(settings["minimum_selected_genes_for_enrichment"]):
        return []
    rows = []
    for pathway, genes in pathways.items():
        background = set(genes) & universe
        size = len(background)
        if size < int(settings["minimum_pathway_size"]) or size > int(settings["maximum_pathway_size"]):
            continue
        overlap = selected & background
        pvalue = stats.hypergeom.sf(len(overlap) - 1, len(universe), size, len(selected))
        rows.append({
            "modality": modality,
            "cohort": cohort,
            "pathway": pathway,
            "selected_genes": len(selected),
            "universe_genes": len(universe),
            "pathway_genes_in_universe": size,
            "overlap_genes": len(overlap),
            "overlap_gene_symbols": " | ".join(sorted(overlap)),
            "p_value": float(pvalue),
            "enrichment_ratio": (
                (len(overlap) / len(selected)) / (size / len(universe))
                if size and universe else float("nan")
            ),
        })
    adjusted = benjamini_hochberg([row["p_value"] for row in rows])
    for row, fdr in zip(rows, adjusted):
        row["fdr_bh"] = float(fdr)
        row["score_minus_log10_p"] = -math.log10(max(row["p_value"], 1e-300))
    rows.sort(key=lambda row: (row["p_value"], -row["overlap_genes"], row["pathway"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["reactome"]
    print("=" * 124)
    print("METABRIC M8.43 - REACTOME PATHWAY CONCORDANCE")
    print("=" * 124)

    cache = out / "cache" / "reactome"
    zip_path = cache / "ReactomePathways.gmt.zip"
    if not zip_path.exists():
        print(f"Downloading Reactome GMT from {settings['url']}")
        download_with_retries(
            settings["url"], zip_path,
            int(settings["timeout_seconds"]), int(settings["max_retries"]),
        )
        source = "downloaded"
    else:
        source = "cache"
    pathways = load_reactome_gmt(zip_path, cache / "extracted")
    cache_summary = {
        "url": settings["url"], "source": source,
        "zip_path": zip_path.relative_to(root).as_posix(),
        "zip_sha256": sha256(zip_path), "pathways_loaded": len(pathways),
    }
    (out / "m43_reactome_cache_summary.json").write_text(
        json.dumps(cache_summary, indent=2), encoding="utf-8"
    )

    gene_rows = read_rows(out / "m42_gene_sets.csv")
    sets = {}
    for row in gene_rows:
        sets.setdefault((row["modality"], row["set_name"]), set()).add(row["gene"].upper())

    modalities = ["RNA", "CNV", "Methylation", "Mutation"]
    enrichment = []
    for modality in modalities:
        universe = sets.get((modality, "Assayed_universe"), set())
        enrichment.extend(enrichment_rows(
            modality, "TCGA_assayable_selected",
            sets.get((modality, "TCGA_assayable_in_METABRIC"), set()),
            universe, pathways, settings,
        ))
        enrichment.extend(enrichment_rows(
            modality, "METABRIC_core_selected",
            sets.get((modality, "METABRIC_core_frequency_ge_0_5"), set()),
            universe, pathways, settings,
        ))

    pooled_universe = set().union(*(sets.get((m, "Assayed_universe"), set()) for m in modalities))
    pooled_tcga = set().union(*(sets.get((m, "TCGA_assayable_in_METABRIC"), set()) for m in modalities))
    pooled_metabric = set().union(*(sets.get((m, "METABRIC_core_frequency_ge_0_5"), set()) for m in modalities))
    enrichment.extend(enrichment_rows(
        "Pooled", "TCGA_assayable_selected", pooled_tcga,
        pooled_universe, pathways, settings,
    ))
    enrichment.extend(enrichment_rows(
        "Pooled", "METABRIC_core_selected", pooled_metabric,
        pooled_universe, pathways, settings,
    ))
    write_csv(out / "m43_reactome_enrichment.csv", enrichment)

    concordance = []
    top_n = int(settings["top_pathways_for_concordance"])
    for modality in modalities + ["Pooled"]:
        tcga = [row for row in enrichment if row["modality"] == modality and row["cohort"] == "TCGA_assayable_selected"]
        metabric = [row for row in enrichment if row["modality"] == modality and row["cohort"] == "METABRIC_core_selected"]
        if not tcga or not metabric:
            concordance.append({
                "modality": modality,
                "status": "INSUFFICIENT_SELECTED_GENES_FOR_PATHWAY_ANALYSIS",
                "top_n": top_n,
            })
            continue
        tcga_lookup = {row["pathway"]: float(row["score_minus_log10_p"]) for row in tcga}
        metabric_lookup = {row["pathway"]: float(row["score_minus_log10_p"]) for row in metabric}
        common = sorted(set(tcga_lookup) & set(metabric_lookup))
        correlation = stats.spearmanr(
            [tcga_lookup[pathway] for pathway in common],
            [metabric_lookup[pathway] for pathway in common],
        ).statistic
        top_tcga = {row["pathway"] for row in tcga[:top_n]}
        top_metabric = {row["pathway"] for row in metabric[:top_n]}
        fdr_tcga = {row["pathway"] for row in tcga if float(row["fdr_bh"]) <= 0.10}
        fdr_metabric = {row["pathway"] for row in metabric if float(row["fdr_bh"]) <= 0.10}
        concordance.append({
            "modality": modality,
            "status": "PATHWAY_CONCORDANCE_ESTIMATED",
            "common_tested_pathways": len(common),
            "spearman_enrichment_score_correlation": float(correlation) if np.isfinite(correlation) else float("nan"),
            "top_n": top_n,
            "top_pathway_overlap": len(top_tcga & top_metabric),
            "top_pathway_jaccard": jaccard(top_tcga, top_metabric),
            "top_pathway_overlap_coefficient": overlap_coefficient(top_tcga, top_metabric),
            "fdr_0_10_tcga": len(fdr_tcga),
            "fdr_0_10_metabric": len(fdr_metabric),
            "fdr_0_10_overlap": len(fdr_tcga & fdr_metabric),
            "shared_top_pathways": " | ".join(sorted(top_tcga & top_metabric)),
        })
    write_csv(out / "m43_pathway_concordance.csv", concordance)

    top_rows = []
    for modality in modalities + ["Pooled"]:
        for cohort in ("TCGA_assayable_selected", "METABRIC_core_selected"):
            top_rows.extend([
                row for row in enrichment
                if row["modality"] == modality and row["cohort"] == cohort
            ][:20])
    write_csv(out / "m43_top_reactome_pathways.csv", top_rows)

    print("Reactome cache")
    print(json.dumps(cache_summary, indent=2))
    print("\nPathway concordance")
    print_table(concordance, [
        "modality", "status", "spearman_enrichment_score_correlation",
        "top_pathway_overlap", "top_pathway_jaccard", "fdr_0_10_tcga",
        "fdr_0_10_metabric", "fdr_0_10_overlap", "shared_top_pathways",
    ])
    print("\nPASS: pathway concordance completed without changing predictive models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
