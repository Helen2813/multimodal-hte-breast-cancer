from __future__ import annotations

import json
import re

import pandas as pd

from _metabric_m9_utils import (
    load_config,
    out_dir,
    print_table,
    project_root,
    read_feature_list,
    write_csv,
)


GENE_COLUMN_TOKENS = (
    "gene",
    "symbol",
    "hugo",
    "refgene",
    "ucsc_refgene",
    "transcript",
)


def extract_gene_symbols(value: object) -> list[str]:
    if pd.isna(value):
        return []

    symbols = []
    for token in re.split(r"[;,|/\s]+", str(value).upper()):
        token = token.strip()
        if not token:
            continue
        if token.startswith("CG") and token[2:].isdigit():
            continue
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?", token):
            continue
        if token in {"NA", "NAN", "NONE", "UNKNOWN", "CHR", "GENE"}:
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9._-]{1,29}", token):
            continue
        symbols.append(token)
    return sorted(set(symbols))


def main() -> int:
    root = project_root()
    config = load_config(root)
    output = out_dir(root, config)

    print("=" * 124)
    print("METABRIC M9.48R - CORRECTED METHYLATION PROBE-TO-GENE TRANSPORT AUDIT")
    print("=" * 124)

    probes = {
        value.upper()
        for value in read_feature_list(
            root / config["files"]["historical_methylation_selected"]
        )
        if value.upper().startswith("CG")
    }

    universe = pd.read_csv(
        root / config["files"]["m8_modality_universe"],
        dtype=str,
        low_memory=False,
    )
    metabric_gene_universe = {
        str(value).upper()
        for value in universe.loc[
            universe["modality"] == "Methylation",
            "gene_or_probe",
        ].dropna()
    }

    source_rows = []
    audit_rows = []

    for candidate in config["files"]["methylation_annotation_candidates"]:
        path = root / candidate
        if not path.exists():
            source_rows.append({
                "path": candidate,
                "exists": False,
                "rows": 0,
                "columns": "",
                "best_probe_column": "",
                "matched_probes": 0,
                "annotation_columns": "",
                "schema_status": "FILE_MISSING",
            })
            continue

        frame = pd.read_csv(path, dtype=str, low_memory=False)

        probe_scores = []
        for column in frame.columns:
            values = frame[column].fillna("").astype(str).str.upper()
            probe_scores.append((int(values.isin(probes).sum()), column))
        probe_scores.sort(key=lambda item: (-item[0], str(item[1])))
        matched_count, probe_column = probe_scores[0]

        annotation_columns = [
            column
            for column in frame.columns
            if column != probe_column
            and any(
                token in str(column).lower()
                for token in GENE_COLUMN_TOKENS
            )
        ]

        schema_status = (
            "GENE_ANNOTATION_COLUMNS_PRESENT"
            if annotation_columns
            else "NO_GENE_ANNOTATION_COLUMNS"
        )

        source_rows.append({
            "path": candidate,
            "exists": True,
            "rows": len(frame),
            "columns": " | ".join(map(str, frame.columns)),
            "best_probe_column": probe_column,
            "matched_probes": matched_count,
            "annotation_columns": " | ".join(annotation_columns),
            "schema_status": schema_status,
        })

        if matched_count == 0 or not annotation_columns:
            continue

        matched = frame[
            frame[probe_column]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(probes)
        ]

        for _, row in matched.iterrows():
            probe = str(row[probe_column]).upper()
            candidate_genes = set()
            evidence = []

            for column in annotation_columns:
                genes = extract_gene_symbols(row[column])
                candidate_genes.update(genes)
                if genes:
                    evidence.append(f"{column}={'/'.join(genes)}")

            assayable_genes = sorted(
                candidate_genes & metabric_gene_universe
            )

            audit_rows.append({
                "source_path": candidate,
                "probe": probe,
                "candidate_genes": " | ".join(sorted(candidate_genes)),
                "candidate_gene_count": len(candidate_genes),
                "assayable_metabric_genes": " | ".join(assayable_genes),
                "assayable_gene_count": len(assayable_genes),
                "annotation_evidence": " || ".join(evidence),
            })

    write_csv(
        output / "m48_methylation_annotation_sources.csv",
        source_rows,
    )
    write_csv(
        output / "m48_methylation_probe_gene_audit.csv",
        audit_rows,
        fieldnames=[
            "source_path",
            "probe",
            "candidate_genes",
            "candidate_gene_count",
            "assayable_metabric_genes",
            "assayable_gene_count",
            "annotation_evidence",
        ],
    )

    probes_with_mapping = {
        row["probe"]
        for row in audit_rows
        if int(row["candidate_gene_count"]) > 0
    }
    probes_with_assayable_mapping = {
        row["probe"]
        for row in audit_rows
        if int(row["assayable_gene_count"]) > 0
    }
    annotation_sources_present = any(
        row["schema_status"] == "GENE_ANNOTATION_COLUMNS_PRESENT"
        for row in source_rows
    )

    if not annotation_sources_present:
        status = "NO_LOCAL_PROBE_TO_GENE_ANNOTATION"
        interpretation = (
            "The available local methylation result files contain statistical "
            "columns but no probe-to-gene annotation columns. Exact methylation "
            "gene transport cannot be evaluated from these files."
        )
    elif len(probes_with_assayable_mapping) < 5:
        status = "METHYLATION_EXACT_GENE_TRANSPORT_NOT_ESTIMABLE"
        interpretation = (
            "Fewer than five historical selected probes map to genes represented "
            "in the METABRIC promoter-level matrix. Zero exact overlap must not be "
            "interpreted as biological non-replication."
        )
    else:
        status = "METHYLATION_GENE_TRANSPORT_ESTIMABLE"
        interpretation = (
            "Sufficient historical probes map to METABRIC-assayable genes for a "
            "formal exact gene-level comparison."
        )

    summary = {
        "historical_selected_probes": len(probes),
        "local_files_with_gene_annotation_columns": sum(
            row["schema_status"] == "GENE_ANNOTATION_COLUMNS_PRESENT"
            for row in source_rows
        ),
        "probes_with_candidate_gene_mapping": len(probes_with_mapping),
        "probes_with_at_least_one_assayable_metabric_gene": len(
            probes_with_assayable_mapping
        ),
        "metabric_methylation_gene_universe": len(
            metabric_gene_universe
        ),
        "status": status,
        "interpretation": interpretation,
        "correction": (
            "Numeric statistical values are explicitly excluded from gene-symbol "
            "parsing."
        ),
    }
    (
        output / "m48_methylation_transport_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Annotation source audit")
    print_table(
        source_rows,
        [
            "path",
            "exists",
            "rows",
            "best_probe_column",
            "matched_probes",
            "annotation_columns",
            "schema_status",
        ],
    )

    print("\nProbe-to-gene mappings")
    print_table(
        audit_rows,
        [
            "probe",
            "candidate_genes",
            "assayable_metabric_genes",
            "source_path",
        ],
        max_rows=100,
    )

    print("\nCorrected methylation transport summary")
    print(json.dumps(summary, indent=2))

    print("\nPASS: methylation transport audit corrected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
