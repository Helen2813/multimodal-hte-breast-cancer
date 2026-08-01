from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _metabric_m8_utils import (
    figure_dir,
    load_config,
    out_dir,
    print_table,
    project_root,
    read_rows,
    sha256,
    write_csv,
)


def find_metric(rows, modality, metric):
    return next(row for row in rows if row["modality"] == modality and row["metric"] == metric)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    figures = figure_dir(root, cfg)
    print("=" * 124)
    print("METABRIC M8.44 - PUBLICATION ASSETS AND DECISION")
    print("=" * 124)

    modality_summary = read_rows(out / "m41_modality_summary.csv")
    repeat_summary = read_rows(out / "m41_repeat_level_summary.csv")
    frequency = read_rows(out / "m41_feature_selection_frequency.csv")
    gene_concordance = read_rows(out / "m42_gene_concordance.csv")
    pathway_concordance = read_rows(out / "m43_pathway_concordance.csv")
    m7_track_b = read_rows(root / cfg["files"]["m7_track_b_repeats"])
    m7_summary = json.loads((root / cfg["files"]["m7_track_b_summary"]).read_text(encoding="utf-8"))
    protocol = json.loads((out / "m40_m8_protocol.json").read_text(encoding="utf-8"))
    modalities = ["RNA", "CNV", "Methylation", "Mutation"]
    positions = np.arange(len(modalities))

    delta_c_rows = [find_metric(repeat_summary, modality, "delta_c_index_vs_clinical") for modality in modalities]
    plt.figure(figsize=(8, 4.8))
    plt.errorbar(
        positions,
        [float(row["mean"]) for row in delta_c_rows],
        yerr=[float(row["sd"]) for row in delta_c_rows],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(positions, modalities)
    plt.ylabel("Mean delta C-index vs clinical-only")
    plt.title("Modality-specific repeated nested analysis")
    save_figure(figures / "m44_modality_delta_c_index.png")

    delta_auc_rows = [find_metric(repeat_summary, modality, "delta_auc_5y_vs_clinical") for modality in modalities]
    plt.figure(figsize=(8, 4.8))
    plt.errorbar(
        positions,
        [float(row["mean"]) for row in delta_auc_rows],
        yerr=[float(row["sd"]) for row in delta_auc_rows],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(positions, modalities)
    plt.ylabel("Mean delta 5-year AUC vs clinical-only")
    plt.title("Modality-specific repeated nested analysis")
    save_figure(figures / "m44_modality_delta_auc_5y.png")

    repeats = [int(float(row["repeat"])) for row in m7_track_b]
    deltas = [float(row["delta_c_index_vs_clinical"]) for row in m7_track_b]
    plt.figure(figsize=(8, 4.8))
    plt.plot(repeats, deltas, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Repeated split")
    plt.ylabel("Delta C-index vs clinical-only")
    plt.title("Full reconstructed multimodal Track B")
    save_figure(figures / "m44_track_b_repeat_delta_c.png")

    for modality in modalities:
        rows = [row for row in frequency if row["modality"] == modality][:15]
        rows = list(reversed(rows))
        if not rows:
            continue
        plt.figure(figsize=(8, max(4.8, 0.3 * len(rows))))
        plt.barh(np.arange(len(rows)), [float(row["selection_frequency"]) for row in rows])
        plt.yticks(np.arange(len(rows)), [row["gene_or_probe"] for row in rows])
        plt.xlim(0.0, 1.0)
        plt.xlabel("Selection frequency across 50 outer folds")
        plt.title(f"{modality}: most stable selected features")
        save_figure(figures / f"m44_{modality.lower()}_feature_stability.png")

    core_gene_rows = [
        row for row in gene_concordance
        if row["metabric_set"] == "METABRIC_core_frequency_ge_0_5"
    ]
    gene_jaccard = [
        float(next(row["jaccard"] for row in core_gene_rows if row["modality"] == modality))
        for modality in modalities
    ]
    pathway_jaccard = []
    for modality in modalities:
        row = next(item for item in pathway_concordance if item["modality"] == modality)
        pathway_jaccard.append(
            float(row.get("top_pathway_jaccard", "nan"))
            if row["status"] == "PATHWAY_CONCORDANCE_ESTIMATED" else np.nan
        )
    width = 0.35
    plt.figure(figsize=(8, 4.8))
    plt.bar(positions - width / 2, gene_jaccard, width, label="Gene")
    plt.bar(positions + width / 2, pathway_jaccard, width, label="Pathway top-20")
    plt.xticks(positions, modalities)
    plt.ylabel("Jaccard")
    plt.title("Cross-cohort gene and pathway concordance")
    plt.legend()
    save_figure(figures / "m44_gene_pathway_concordance.png")

    checks = [
        {"check": "All four modality analyses completed", "observed": len(modality_summary), "pass": len(modality_summary) == 4},
        {"check": "Gene concordance produced", "observed": len(gene_concordance), "pass": len(gene_concordance) >= 8},
        {"check": "Pathway concordance produced", "observed": len(pathway_concordance), "pass": len(pathway_concordance) >= 5},
        {
            "check": "Reconstructed-method label retained",
            "observed": m7_summary["historical_engine_status"],
            "pass": m7_summary["historical_engine_status"] == "RECONSTRUCTED_IAMB_ENGINE_NOT_BITWISE_REPRODUCED",
        },
    ]
    complete = all(bool(row["pass"]) for row in checks)
    decision = "M8_MODALITY_GENE_PATHWAY_ANALYSIS_COMPLETE" if complete else "M8_ANALYSIS_INCOMPLETE"

    publication_rows = []
    for modality in modalities:
        summary = next(row for row in modality_summary if row["modality"] == modality)
        gene = next(row for row in core_gene_rows if row["modality"] == modality)
        pathway = next(row for row in pathway_concordance if row["modality"] == modality)
        publication_rows.append({
            "modality": modality,
            "mean_delta_c_index": summary["mean_delta_c_index"],
            "sd_delta_c_index": summary["sd_delta_c_index"],
            "mean_delta_auc_5y": summary["mean_delta_auc_5y"],
            "sd_delta_auc_5y": summary["sd_delta_auc_5y"],
            "mean_within_repeat_jaccard": summary["mean_within_repeat_jaccard"],
            "core_features": summary["core_features_frequency_ge_0_5"],
            "tcga_assayable_genes": gene["tcga_assayable_genes"],
            "metabric_core_genes": gene["metabric_selected_genes"],
            "exact_gene_overlap": gene["exact_gene_overlap"],
            "gene_jaccard": gene["jaccard"],
            "gene_overlap_p": gene["hypergeometric_p"],
            "pathway_status": pathway["status"],
            "pathway_score_spearman": pathway.get("spearman_enrichment_score_correlation", ""),
            "top20_pathway_jaccard": pathway.get("top_pathway_jaccard", ""),
        })
    write_csv(out / "m44_publication_summary_table.csv", publication_rows)
    write_csv(out / "m44_decision_checks.csv", checks)

    report = {
        "metabric_m8_decision": decision,
        "protocol_id": protocol["protocol_id"],
        "core_interpretation": (
            "Predictive incremental value and biological reproducibility are reported separately. "
            "Gene or pathway concordance cannot override null or negative performance results."
        ),
        "m7_core_result": {
            "track_a": "The fixed transportable TCGA RNA/CNA panel did not provide reliable incremental discrimination beyond clinical variables.",
            "track_b": "The full reconstructed multimodal procedure was inferior to clinical-only for global survival ranking across all 20 repeated splits.",
        },
        "manuscript_positioning": (
            "A rigorous cross-cohort stress test of dependency-aware multimodal representations, "
            "emphasizing platform assayability, leakage control, feature stability, and the distinction "
            "between biological recurrence and incremental prognostic utility."
        ),
        "next_step": "Generate final manuscript tables, captions, Results, and Discussion text from the locked M7-M8 outputs.",
        "boundaries": [
            "Do not call stable selected genes causal biomarkers.",
            "Do not describe repeated-split quantiles as confidence intervals.",
            "Do not claim exact replication of the historical IAMB implementation.",
            "Do not claim incremental utility unless the locked comparison supports it.",
        ],
    }
    (out / "m44_m8_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# METABRIC M8 manuscript-ready results", "", "## Core result", "",
        "The fixed TCGA-selected transportable RNA/CNA panel did not provide a reliable incremental discrimination advantage over clinical-only in METABRIC.",
        "",
        f"In the leakage-controlled reconstructed multimodal analysis, the mean repeated-split C-index difference was {m7_summary['mean_repeat_delta_c_index']:.4f} relative to clinical-only.",
        "", "## Modality-specific results", "",
    ]
    for row in publication_rows:
        lines.append(
            f"- **{row['modality']}**: mean delta C-index {float(row['mean_delta_c_index']):.4f}; "
            f"mean delta 5-year AUC {float(row['mean_delta_auc_5y']):.4f}; "
            f"gene overlap {row['exact_gene_overlap']}/{row['tcga_assayable_genes']} assayable TCGA-selected genes."
        )
    lines.extend([
        "", "## Interpretation", "",
        "Feature recurrence and pathway concordance are biological reproducibility evidence, not proof of incremental clinical utility or causality.",
    ])
    (out / "m44_manuscript_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    inventory_paths = [
        out / "m40_m8_protocol.json",
        out / "m41_modality_summary.csv",
        out / "m41_repeat_level_summary.csv",
        out / "m41_feature_selection_frequency.csv",
        out / "m42_gene_concordance.csv",
        out / "m43_pathway_concordance.csv",
        out / "m43_top_reactome_pathways.csv",
        out / "m44_publication_summary_table.csv",
        out / "m44_m8_report.json",
        out / "m44_manuscript_results.md",
    ]
    write_csv(out / "m44_output_hashes.csv", [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in inventory_paths
    ])

    print(f"Decision: {decision}")
    print("\nDecision checks")
    print_table(checks, ["check", "observed", "pass"])
    print("\nPublication summary")
    print_table(publication_rows, [
        "modality", "mean_delta_c_index", "mean_delta_auc_5y",
        "mean_within_repeat_jaccard", "core_features", "tcga_assayable_genes",
        "metabric_core_genes", "exact_gene_overlap", "gene_jaccard",
        "top20_pathway_jaccard",
    ])
    print("\nFinal report")
    print(json.dumps(report, indent=2))
    if not complete:
        raise RuntimeError("M8 completion checks failed")
    print("\nPASS: M8 modality, gene, pathway, and publication assets completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
