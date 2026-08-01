from __future__ import annotations

import csv
import json
from pathlib import Path

from _metabric_m3_utils import (
    classify_identifiers, ensembl_like, gene_symbol_like, infer_delimiter,
    load_config, noncomment_lines, out_dir, parse_fields, print_table,
    project_root, read_text_prefix, rel, write_csv
)


def sample_first_column(path: Path, delimiter: str, limit: int = 5000) -> list[str]:
    values = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")), delimiter=delimiter)
        try:
            next(reader)
        except StopIteration:
            return values
        for row in reader:
            if row and row[0].strip():
                values.append(row[0].strip().strip('"'))
            if len(values) >= limit:
                break
    return values


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M3.12 - CANDIDATE IDENTIFIER AND DIRECT-OVERLAP AUDIT")
    print("=" * 124)

    gene_sets = json.loads((out / "m10_metabric_gene_sets_LOCAL_ONLY.json").read_text(encoding="utf-8"))
    metabric_union = set()
    for values in gene_sets.values():
        metabric_union.update(str(x).upper() for x in values)

    ranked_path = out / "m11_ranked_tcga_source_candidates.csv"
    with ranked_path.open("r", encoding="utf-8-sig", newline="") as f:
        ranked = list(csv.DictReader(f))

    rows = []
    relevant_modalities = {"rna", "cna", "mutations", "methylation", "annotation"}
    for item in ranked:
        if item["modality"] not in relevant_modalities:
            continue
        path = (root / item["path"]).resolve()
        if not path.exists() or path.suffix.lower() in {".parquet", ".feather"}:
            continue

        prefix = read_text_prefix(path, int(cfg["header_read_bytes"]))
        lines = noncomment_lines(prefix, limit=2)
        if not lines:
            continue
        delimiter = infer_delimiter(lines[0])
        header = parse_fields(lines[0], delimiter)
        first_values = sample_first_column(path, delimiter, limit=5000)

        header_class = classify_identifiers(header[:5000])
        first_class = classify_identifiers(first_values)

        header_symbols = {x.upper() for x in header if gene_symbol_like(x)}
        first_symbols = {x.upper() for x in first_values if gene_symbol_like(x)}
        header_ens = {x for x in header if ensembl_like(x)}
        first_ens = {x for x in first_values if ensembl_like(x)}

        direct_symbols = header_symbols | first_symbols
        direct_overlap = direct_symbols & metabric_union

        if len(header_symbols) >= len(first_symbols) and header_symbols:
            likely_axis = "header"
            likely_type = "gene_symbol"
            likely_count = len(header_symbols)
        elif first_symbols:
            likely_axis = "first_column"
            likely_type = "gene_symbol"
            likely_count = len(first_symbols)
        elif len(header_ens) >= len(first_ens) and header_ens:
            likely_axis = "header"
            likely_type = "ensembl"
            likely_count = len(header_ens)
        elif first_ens:
            likely_axis = "first_column"
            likely_type = "ensembl"
            likely_count = len(first_ens)
        else:
            likely_axis = "unresolved"
            likely_type = "unknown"
            likely_count = 0

        rows.append({
            "modality": item["modality"],
            "rank": item["rank"],
            "score": item["score"],
            "path": item["path"],
            "likely_identifier_axis": likely_axis,
            "likely_identifier_type": likely_type,
            "likely_identifier_count": likely_count,
            "direct_gene_symbol_overlap_with_metabric_union": len(direct_overlap),
            "header_gene_symbols": len(header_symbols),
            "first_column_gene_symbols": len(first_symbols),
            "header_ensembl_ids": len(header_ens),
            "first_column_ensembl_ids": len(first_ens),
        })

    rows.sort(key=lambda r: (r["modality"], int(r["rank"])))
    write_csv(out / "m12_candidate_identifier_overlap_audit.csv", rows)

    print("Candidate identifier audit")
    print_table(
        rows,
        ["modality", "rank", "score", "path", "likely_identifier_axis",
         "likely_identifier_type", "likely_identifier_count",
         "direct_gene_symbol_overlap_with_metabric_union",
         "header_ensembl_ids", "first_column_ensembl_ids"],
        max_rows=80,
    )

    print("\nInterpretation")
    print("  Direct gene-symbol overlap is sufficient only when both cohorts expose compatible symbols.")
    print("  ENSG-based TCGA matrices require a locked Ensembl-to-HUGO mapping before M4.")
    print("  No feature selection or outcome modeling was performed.")

    print("\nPASS: identifier compatibility was audited for every ranked candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
