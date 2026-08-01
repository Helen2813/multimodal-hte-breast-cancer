from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from _metabric_m4_utils import (
    exact_column, load_config, load_m3b_registry, out_dir, print_table,
    project_root, raw_dir, read_cbio, selected_identifiers, write_csv
)


def parse_panel_file(path: Path) -> dict:
    data = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        data[key.strip().lower()] = value.strip()
    genes = re.split(r"[\t,\s]+", data.get("gene_list", "").strip())
    genes = sorted({gene.upper() for gene in genes if gene})
    return {
        "path": path.as_posix(),
        "stable_id": data.get("stable_id", ""),
        "description": data.get("description", ""),
        "gene_count": len(genes),
        "genes": genes,
    }


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M4.23 - MUTATION PANEL-COVERAGE AUDIT")
    print("=" * 124)

    matrix_path = raw / cfg["metabric_files"]["gene_panel_matrix"]
    matrix = read_cbio(matrix_path)
    sample_col = exact_column(matrix.columns, ["SAMPLE_ID"])
    profile_columns = [column for column in matrix.columns if column != sample_col]
    if sample_col is None or not profile_columns:
        raise RuntimeError("Gene panel matrix has no SAMPLE_ID/profile columns.")

    assignment_rows = []
    panel_ids = set()
    for profile in profile_columns:
        counts = Counter(
            value.strip()
            for value in matrix[profile].dropna().astype(str)
            if value.strip() and value.strip().upper() != "NA"
        )
        for panel_id, count in sorted(counts.items()):
            panel_ids.add(panel_id)
            assignment_rows.append({
                "profile": profile,
                "panel_id": panel_id,
                "assigned_samples": count,
            })

    panel_files = []
    for path in sorted(raw.rglob("data_gene_panel*.txt")):
        if path.name.lower() == cfg["metabric_files"]["gene_panel_matrix"].lower():
            continue
        try:
            panel_files.append(parse_panel_file(path))
        except OSError:
            continue

    panel_by_id = {
        row["stable_id"]: row for row in panel_files if row["stable_id"]
    }

    selected_mutation_rows = selected_identifiers(
        load_m3b_registry(root, cfg), "mutations"
    )
    selected_genes = sorted({
        row["canonical_identifier"].upper()
        for row in selected_mutation_rows
        if row["identifier_type"] == "gene_symbol"
    })

    mutation = read_cbio(raw / cfg["metabric_files"]["mutations"])
    gene_col = exact_column(mutation.columns, ["Hugo_Symbol", "HUGO_SYMBOL"])
    sample_mut_col = exact_column(
        mutation.columns,
        ["Tumor_Sample_Barcode", "TUMOR_SAMPLE_BARCODE", "SAMPLE_ID"],
    )
    observed_genes = sorted({
        value.upper()
        for value in mutation[gene_col].dropna().astype(str)
    }) if gene_col else []
    observed_selected = sorted(set(selected_genes) & set(observed_genes))

    panel_rows = []
    coverage_union = set()
    unresolved_panel_ids = []
    for panel_id in sorted(panel_ids):
        panel = panel_by_id.get(panel_id)
        if panel:
            coverage_union.update(panel["genes"])
            panel_rows.append({
                "panel_id": panel_id,
                "definition_found": True,
                "gene_count": panel["gene_count"],
                "selected_tcga_genes_covered": len(set(selected_genes) & set(panel["genes"])),
                "selected_covered_examples": " | ".join(sorted(set(selected_genes) & set(panel["genes"]))[:25]),
                "definition_path": panel["path"],
            })
        else:
            unresolved_panel_ids.append(panel_id)
            panel_rows.append({
                "panel_id": panel_id,
                "definition_found": False,
                "gene_count": "",
                "selected_tcga_genes_covered": "",
                "selected_covered_examples": "",
                "definition_path": "",
            })

    selected_covered_union = sorted(set(selected_genes) & coverage_union)
    if panel_ids and not unresolved_panel_ids and panel_files:
        status = "GENE_LEVEL_WILDTYPE_CODING_ALLOWED_WITH_PANEL_AWARENESS"
    elif not panel_ids:
        status = "NO_GENE_PANEL_ASSIGNMENTS_FOUND"
    else:
        status = "GENE_PANEL_DEFINITIONS_MISSING_BLOCK_GENE_LEVEL_NEGATIVE_CODING"

    summary = {
        "gene_panel_matrix_rows": len(matrix),
        "profile_columns": profile_columns,
        "unique_panel_ids": sorted(panel_ids),
        "panel_definition_files_found": len(panel_files),
        "unresolved_panel_ids": unresolved_panel_ids,
        "selected_tcga_mutation_genes": len(selected_genes),
        "selected_genes_observed_mutated_in_metabric": len(observed_selected),
        "observed_selected_examples": observed_selected,
        "selected_genes_covered_by_resolved_panels": len(selected_covered_union),
        "covered_selected_examples": selected_covered_union,
        "mutation_transport_status": status,
        "important_rule": (
            "An absent mutation call is wild-type only when that sample's assigned panel covers the gene."
        ),
    }

    write_csv(out / "m23_gene_panel_assignments.csv", assignment_rows)
    write_csv(out / "m23_gene_panel_definitions.csv", panel_rows)
    (out / "m23_mutation_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Gene-panel assignments")
    print_table(assignment_rows, ["profile", "panel_id", "assigned_samples"])

    print("\nGene-panel definitions")
    print_table(
        panel_rows,
        [
            "panel_id", "definition_found", "gene_count",
            "selected_tcga_genes_covered", "selected_covered_examples",
            "definition_path"
        ],
    )

    print("\nMutation coverage summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nPASS: mutation panel coverage audited. No absent call was automatically coded as wild-type.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
