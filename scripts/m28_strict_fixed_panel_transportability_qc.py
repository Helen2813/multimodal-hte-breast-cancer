from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from _metabric_m5_utils import (
    load_config, out_dir, print_table, project_root, read_rows, write_csv
)


def feature_status_from_matrix(columns: list[str], ensembl_id: str, symbol: str) -> bool:
    tokens = [ensembl_id.upper(), f"__{symbol.upper()}"]
    for column in columns:
        upper = str(column).upper()
        if any(token and token in upper for token in tokens):
            return True
    return False


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    m4 = root / cfg["metabric_m4_dir"]

    print("=" * 124)
    print("METABRIC M5.28 - STRICT FIXED-PANEL TRANSPORTABILITY QC")
    print("=" * 124)

    selected = pd.read_csv(
        m4 / "m20_selected_tcga_feature_identifiers.csv",
        dtype=str,
        low_memory=False,
    )
    mapping = pd.read_csv(
        m4 / "m21_ensembl_to_hgnc_mapping.csv",
        dtype=str,
        low_memory=False,
    )
    mapping_lookup = {
        row["ensembl_id"]: row
        for _, row in mapping.iterrows()
    }

    rna_matrix = pd.read_csv(
        m4 / "m22_metabric_fixed_tcga_rna_panel_LOCAL_ONLY.csv",
        nrows=0,
    )
    cna_matrix = pd.read_csv(
        m4 / "m22_metabric_fixed_tcga_cna_panel_LOCAL_ONLY.csv",
        nrows=0,
    )

    primary_allowed = set(cfg["mapping_confidence"]["primary_allowed"])
    sensitivity_only = set(cfg["mapping_confidence"]["sensitivity_only"])

    rows = []
    for _, item in selected.iterrows():
        modality = str(item["modality"])
        identifier = str(item["canonical_identifier"])
        identifier_type = str(item["identifier_type"])
        if modality not in {"rna", "cna"} or identifier_type != "ensembl":
            continue

        mapped = mapping_lookup.get(identifier)
        symbol = ""
        status = "UNMAPPED"
        if mapped is not None:
            symbol = str(mapped.get("selected_hgnc_symbol", "") or "").upper()
            if symbol == "NAN":
                symbol = ""
            status = str(mapped.get("mapping_status", "UNMAPPED"))

        matrix_columns = list(rna_matrix.columns) if modality == "rna" else list(cna_matrix.columns)
        assayed = bool(symbol) and feature_status_from_matrix(matrix_columns, identifier, symbol)

        if status in primary_allowed and assayed:
            role = "PRIMARY_TRANSPORTABLE"
        elif status in sensitivity_only and assayed:
            role = "SENSITIVITY_ONLY_MAPPING_FALLBACK"
        elif status in primary_allowed | sensitivity_only and not assayed:
            role = "MAPPED_BUT_NOT_ASSAYED_IN_METABRIC"
        else:
            role = "EXCLUDED_AMBIGUOUS_OR_UNMAPPED"

        symbol_class = "protein_coding_or_standard_symbol"
        if re.match(r"^(RP11|CTB|CTC|LINC|MIR|SNORD|RNU|RN7SL|.*P\d+$)", symbol):
            symbol_class = "noncoding_pseudogene_or_legacy_locus"

        rows.append({
            "modality": modality,
            "tcga_column": item["tcga_column"],
            "ensembl_id": identifier,
            "mapped_symbol": symbol,
            "mapping_status": status,
            "assayed_in_metabric": assayed,
            "transport_role": role,
            "symbol_class": symbol_class,
        })

    write_csv(out / "m28_feature_level_transportability.csv", rows)

    summary_rows = []
    for modality in ("rna", "cna"):
        subset = [row for row in rows if row["modality"] == modality]
        roles = {}
        statuses = {}
        for row in subset:
            roles[row["transport_role"]] = roles.get(row["transport_role"], 0) + 1
            statuses[row["mapping_status"]] = statuses.get(row["mapping_status"], 0) + 1
        summary_rows.append({
            "modality": modality,
            "selected_ensembl_features": len(subset),
            "primary_transportable": roles.get("PRIMARY_TRANSPORTABLE", 0),
            "sensitivity_only": roles.get("SENSITIVITY_ONLY_MAPPING_FALLBACK", 0),
            "mapped_not_assayed": roles.get("MAPPED_BUT_NOT_ASSAYED_IN_METABRIC", 0),
            "ambiguous_or_unmapped": roles.get("EXCLUDED_AMBIGUOUS_OR_UNMAPPED", 0),
            "primary_transport_fraction": (
                roles.get("PRIMARY_TRANSPORTABLE", 0) / len(subset)
                if subset else 0.0
            ),
            "mapping_status_counts": json.dumps(statuses, sort_keys=True),
        })
    write_csv(out / "m28_transportability_summary.csv", summary_rows)

    primary_rows = [
        row for row in rows if row["transport_role"] == "PRIMARY_TRANSPORTABLE"
    ]
    sensitivity_rows = [
        row for row in rows if row["transport_role"] == "SENSITIVITY_ONLY_MAPPING_FALLBACK"
    ]
    write_csv(out / "m28_primary_transportable_panel.csv", primary_rows)
    write_csv(out / "m28_mapping_fallback_sensitivity_panel.csv", sensitivity_rows)

    print("Feature-level transportability")
    print_table(
        rows,
        [
            "modality", "ensembl_id", "mapped_symbol", "mapping_status",
            "assayed_in_metabric", "transport_role", "symbol_class"
        ],
        max_rows=120,
    )

    print("\nTransportability summary")
    print_table(
        summary_rows,
        [
            "modality", "selected_ensembl_features", "primary_transportable",
            "sensitivity_only", "mapped_not_assayed", "ambiguous_or_unmapped",
            "primary_transport_fraction", "mapping_status_counts"
        ],
    )

    print("\nPrimary panel excludes display-name-only fallbacks, ambiguous mappings, and unassayed features.")
    print("These exclusions are based only on mapping/assayability, never on METABRIC outcomes.")
    print("\nPASS: strict primary and sensitivity transport panels defined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
