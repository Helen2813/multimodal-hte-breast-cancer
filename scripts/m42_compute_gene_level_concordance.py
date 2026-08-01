from __future__ import annotations

import re

import pandas as pd

from _metabric_m8_utils import (
    canonical_ensembl,
    gene_like_tokens,
    hypergeometric_overlap,
    jaccard,
    load_config,
    out_dir,
    overlap_coefficient,
    print_table,
    project_root,
    read_feature_list,
    read_rows,
    strip_feature_prefix,
    write_csv,
)


def map_methylation_probes(root, cfg, selected_probes):
    wanted = {probe.upper() for probe in selected_probes}
    for candidate in cfg["methylation_annotation_candidates"]:
        path = root / candidate
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        scored = []
        for column in frame.columns:
            score = int(frame[column].fillna("").astype(str).str.upper().isin(wanted).sum())
            scored.append((score, column))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        if not scored or scored[0][0] == 0:
            continue
        probe_column = scored[0][1]
        annotation_columns = [
            column for column in frame.columns
            if column != probe_column
            and any(token in str(column).lower() for token in ("gene", "hugo", "symbol"))
        ]
        if not annotation_columns:
            annotation_columns = [column for column in frame.columns if column != probe_column]
        rows = []
        matched = frame[frame[probe_column].fillna("").astype(str).str.upper().isin(wanted)]
        for _, record in matched.iterrows():
            genes = set()
            for column in annotation_columns:
                genes.update(gene_like_tokens(record[column]))
            rows.append({
                "probe": str(record[probe_column]).upper(),
                "mapped_genes": " | ".join(sorted(genes)),
                "mapped_gene_count": len(genes),
                "source_path": candidate,
                "probe_column": probe_column,
                "annotation_columns": " | ".join(annotation_columns),
            })
        return rows, candidate
    return [], ""


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    print("=" * 124)
    print("METABRIC M8.42 - GENE-LEVEL CROSS-COHORT CONCORDANCE")
    print("=" * 124)

    mapping = pd.read_csv(root / cfg["files"]["ensembl_mapping"], dtype=str, low_memory=False)
    ensembl_to_gene = {
        str(row["ensembl_id"]).upper(): str(row["selected_hgnc_symbol"]).upper()
        for _, row in mapping.iterrows()
        if pd.notna(row["selected_hgnc_symbol"])
        and str(row["selected_hgnc_symbol"]).strip()
        and str(row["selected_hgnc_symbol"]).upper() != "NAN"
        and str(row["mapping_status"]) not in {"AMBIGUOUS_HGNC_SYMBOLS", "AMBIGUOUS_DISPLAY_NAMES", "UNMAPPED"}
    }

    universes = read_rows(out / "m41_modality_feature_universe.csv")
    frequencies = read_rows(out / "m41_feature_selection_frequency.csv")
    universe_by_modality = {
        modality: {
            row["gene_or_probe"].upper()
            for row in universes if row["modality"] == modality
        }
        for modality in ("RNA", "CNV", "Methylation", "Mutation")
    }

    historical = {
        modality: read_feature_list(root / path)
        for modality, path in cfg["historical_selected_files"].items()
    }
    mapping_details = []
    tcga_sets = {}

    for modality in ("RNA", "CNV"):
        genes = set()
        for feature in historical[modality]:
            ensembl = canonical_ensembl(feature)
            gene = ensembl_to_gene.get(ensembl, "")
            if gene:
                genes.add(gene)
            mapping_details.append({
                "modality": modality,
                "tcga_feature": feature,
                "canonical_identifier": ensembl,
                "mapped_gene": gene,
                "mapping_status": "mapped" if gene else "unmapped",
                "source": "M4 Ensembl-to-HGNC mapping",
            })
        tcga_sets[modality] = genes

    mutation_genes = {
        strip_feature_prefix(feature)
        for feature in historical["Mutation"]
        if strip_feature_prefix(feature) not in {"", "MISSING"}
    }
    tcga_sets["Mutation"] = mutation_genes
    mapping_details.extend({
        "modality": "Mutation",
        "tcga_feature": feature,
        "canonical_identifier": strip_feature_prefix(feature),
        "mapped_gene": strip_feature_prefix(feature) if strip_feature_prefix(feature) != "MISSING" else "",
        "mapping_status": "direct_gene_symbol",
        "source": "direct symbol",
    } for feature in historical["Mutation"])

    probes = [
        strip_feature_prefix(feature)
        for feature in historical["Methylation"]
        if re.match(r"^CG\d+", strip_feature_prefix(feature))
    ]
    methylation_rows, methylation_source = map_methylation_probes(root, cfg, probes)
    probe_lookup = {
        row["probe"]: {gene.strip().upper() for gene in row["mapped_genes"].split("|") if gene.strip()}
        for row in methylation_rows
    }
    methylation_genes = set().union(*probe_lookup.values()) if probe_lookup else set()
    tcga_sets["Methylation"] = methylation_genes
    mapping_details.extend({
        "modality": "Methylation",
        "tcga_feature": probe,
        "canonical_identifier": probe,
        "mapped_gene": " | ".join(sorted(probe_lookup.get(probe, set()))),
        "mapping_status": "mapped" if probe_lookup.get(probe) else "unmapped",
        "source": methylation_source,
    } for probe in probes)

    write_csv(out / "m42_tcga_feature_mapping.csv", mapping_details)
    write_csv(out / "m42_methylation_probe_gene_mapping.csv", methylation_rows)

    gene_set_rows = []
    concordance_rows = []
    mapping_summary = []
    for modality in ("RNA", "CNV", "Methylation", "Mutation"):
        universe = universe_by_modality[modality]
        tcga_all = {gene for gene in tcga_sets[modality] if gene}
        tcga_assayable = tcga_all & universe
        modality_frequency = [row for row in frequencies if row["modality"] == modality]
        metabric_core = {
            row["gene_or_probe"].upper()
            for row in modality_frequency
            if str(row["core_by_frequency"]).lower() == "true"
        }
        metabric_recurrent = {
            row["gene_or_probe"].upper()
            for row in modality_frequency
            if str(row["recurrent_by_repeat"]).lower() == "true"
        }
        for set_name, genes in (
            ("TCGA_all_mapped", tcga_all),
            ("TCGA_assayable_in_METABRIC", tcga_assayable),
            ("METABRIC_core_frequency_ge_0_5", metabric_core),
            ("METABRIC_recurrent_repeat_rule", metabric_recurrent),
            ("Assayed_universe", universe),
        ):
            gene_set_rows.extend({"modality": modality, "set_name": set_name, "gene": gene} for gene in sorted(genes))
        for set_name, metabric_set in (
            ("METABRIC_core_frequency_ge_0_5", metabric_core),
            ("METABRIC_recurrent_repeat_rule", metabric_recurrent),
        ):
            overlap = tcga_assayable & metabric_set
            concordance_rows.append({
                "modality": modality,
                "metabric_set": set_name,
                "tcga_selected_raw_features": len(historical[modality]),
                "tcga_mapped_genes": len(tcga_all),
                "tcga_assayable_genes": len(tcga_assayable),
                "metabric_selected_genes": len(metabric_set),
                "assayed_universe": len(universe),
                "exact_gene_overlap": len(overlap),
                "overlap_genes": " | ".join(sorted(overlap)),
                "jaccard": jaccard(tcga_assayable, metabric_set),
                "overlap_coefficient": overlap_coefficient(tcga_assayable, metabric_set),
                "hypergeometric_p": hypergeometric_overlap(len(universe), len(tcga_assayable), len(metabric_set), len(overlap)),
                "assayability_fraction_of_tcga_mapped": len(tcga_assayable) / len(tcga_all) if tcga_all else 0.0,
                "interpretation_status": "INSUFFICIENT_ASSAYABLE_TCGA_GENES" if len(tcga_assayable) < 5 else "GENE_CONCORDANCE_ESTIMABLE",
            })
        mapping_summary.append({
            "modality": modality,
            "historical_selected_features": len(historical[modality]),
            "mapped_unique_genes": len(tcga_all),
            "assayable_in_metabric": len(tcga_assayable),
            "mapping_fraction": len(tcga_all) / len(historical[modality]) if historical[modality] else 0.0,
            "assayability_fraction": len(tcga_assayable) / len(tcga_all) if tcga_all else 0.0,
            "methylation_annotation_source": methylation_source if modality == "Methylation" else "",
        })

    write_csv(out / "m42_gene_sets.csv", gene_set_rows)
    write_csv(out / "m42_gene_concordance.csv", concordance_rows)
    write_csv(out / "m42_mapping_summary.csv", mapping_summary)

    print("Historical-feature mapping summary")
    print_table(mapping_summary, [
        "modality", "historical_selected_features", "mapped_unique_genes",
        "assayable_in_metabric", "mapping_fraction", "assayability_fraction",
        "methylation_annotation_source",
    ])
    print("\nGene concordance")
    print_table(concordance_rows, [
        "modality", "metabric_set", "tcga_assayable_genes",
        "metabric_selected_genes", "exact_gene_overlap", "jaccard",
        "overlap_coefficient", "hypergeometric_p", "interpretation_status",
        "overlap_genes",
    ])
    print("\nPASS: gene concordance quantified using assayable denominators.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
