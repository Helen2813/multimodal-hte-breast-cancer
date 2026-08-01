from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from _metabric_m4_utils import (
    exact_column, load_config, numeric_diagnostics, out_dir, print_table,
    project_root, raw_dir, read_cbio, selected_identifiers, load_m3b_registry,
    write_csv
)


def aggregate_rows_by_symbol(rows: pd.DataFrame, sample_columns: list[str]) -> pd.DataFrame:
    numeric = rows[sample_columns].apply(pd.to_numeric, errors="coerce")
    numeric.insert(0, "Hugo_Symbol", rows["Hugo_Symbol"].astype(str).str.upper())
    return numeric.groupby("Hugo_Symbol", sort=True).median(numeric_only=True)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    mapping = pd.read_csv(out / "m21_ensembl_to_hgnc_mapping.csv", dtype=str)
    registry = load_m3b_registry(root, cfg)
    clinical_sample = read_cbio(raw / cfg["metabric_files"]["clinical_sample"])
    sample_col = exact_column(clinical_sample.columns, ["SAMPLE_ID"])
    patient_col = exact_column(clinical_sample.columns, ["PATIENT_ID"])
    if sample_col is None or patient_col is None:
        raise RuntimeError("METABRIC sample-to-patient mapping is incomplete.")

    sample_to_patient = dict(zip(
        clinical_sample[sample_col].astype(str).str.strip(),
        clinical_sample[patient_col].astype(str).str.strip(),
    ))

    map_dict = dict(zip(
        mapping["ensembl_id"].astype(str),
        mapping["selected_hgnc_symbol"].fillna("").astype(str).str.upper(),
    ))

    print("=" * 124)
    print("METABRIC M4.22 - OUTCOME-BLIND FIXED TCGA PANEL MATRICES IN METABRIC")
    print("=" * 124)

    feature_map_rows = []
    modality_symbols = {}
    for modality in ("rna", "cna"):
        rows = selected_identifiers(registry, modality)
        symbols = []
        for row in rows:
            if row["identifier_type"] == "ensembl":
                symbol = map_dict.get(row["canonical_identifier"], "")
                source = "ensembl_hgnc_mapping"
            elif row["identifier_type"] == "gene_symbol":
                symbol = row["canonical_identifier"].upper()
                source = "direct_gene_symbol"
            else:
                symbol = ""
                source = "sentinel_or_other"
            if symbol:
                symbols.append(symbol)
            feature_map_rows.append({
                **row,
                "mapped_hgnc_symbol": symbol,
                "mapping_source": source,
            })
        modality_symbols[modality] = sorted(set(symbols))

    write_csv(out / "m22_fixed_panel_feature_map.csv", feature_map_rows)

    # RNA: cleaned matrix is samples x HUGO genes and was verified in M2.
    rna_path = raw / cfg["metabric_files"]["rna_cleaned"]
    rna_header = list(pd.read_csv(rna_path, nrows=0).columns)
    rna_sample_column = rna_header[0]
    rna_available = sorted(set(modality_symbols["rna"]) & set(rna_header[1:]))
    rna_missing = sorted(set(modality_symbols["rna"]) - set(rna_available))
    rna_usecols = [rna_sample_column] + rna_available
    rna = pd.read_csv(rna_path, usecols=rna_usecols, low_memory=False)
    rna = rna.rename(columns={rna_sample_column: "sample_id"})
    rna.insert(
        1,
        "patient_id",
        rna["sample_id"].astype(str).map(sample_to_patient),
    )
    rna_feature_rename = {}
    symbol_to_tcga = defaultdict(list)
    for row in feature_map_rows:
        if row["modality"] == "rna" and row["mapped_hgnc_symbol"]:
            symbol_to_tcga[row["mapped_hgnc_symbol"]].append(row["canonical_identifier"])
    for symbol in rna_available:
        ids = sorted(set(symbol_to_tcga[symbol]))
        rna_feature_rename[symbol] = f"TCGA_RNA_{'+'.join(ids)}__{symbol}"
    rna = rna.rename(columns=rna_feature_rename)
    rna_path_out = out / "m22_metabric_fixed_tcga_rna_panel_LOCAL_ONLY.csv"
    rna.to_csv(rna_path_out, index=False)

    # CNA: gene rows x samples. Stream all rows, retain selected mapped HUGO symbols.
    cna_path = raw / cfg["metabric_files"]["cna"]
    selected_cna_symbols = set(modality_symbols["cna"])
    retained_chunks = []
    header = list(pd.read_csv(cna_path, sep="\t", comment="#", nrows=0).columns)
    hugo_col = exact_column(header, ["Hugo_Symbol", "HUGO_SYMBOL"])
    if hugo_col is None:
        raise RuntimeError("Could not resolve Hugo_Symbol in METABRIC CNA file.")
    sample_columns = [column for column in header if column not in {hugo_col, "Entrez_Gene_Id"}]

    for chunk in pd.read_csv(
        cna_path,
        sep="\t",
        comment="#",
        dtype=str,
        chunksize=int(cfg["matrix_build"]["cna_chunksize"]),
        low_memory=False,
    ):
        symbols = chunk[hugo_col].astype(str).str.upper()
        keep = symbols.isin(selected_cna_symbols)
        if keep.any():
            retained = chunk.loc[keep, [hugo_col] + sample_columns].copy()
            retained = retained.rename(columns={hugo_col: "Hugo_Symbol"})
            retained_chunks.append(retained)

    retained_cna = (
        pd.concat(retained_chunks, ignore_index=True)
        if retained_chunks else
        pd.DataFrame(columns=["Hugo_Symbol"] + sample_columns)
    )
    cna_aggregated = aggregate_rows_by_symbol(retained_cna, sample_columns)
    cna_available = sorted(cna_aggregated.index.astype(str))
    cna_missing = sorted(selected_cna_symbols - set(cna_available))
    cna_patient = cna_aggregated.T.reset_index().rename(columns={"index": "sample_id"})
    cna_patient.insert(
        1,
        "patient_id",
        cna_patient["sample_id"].astype(str).map(sample_to_patient),
    )

    symbol_to_tcga_cna = defaultdict(list)
    for row in feature_map_rows:
        if row["modality"] == "cna" and row["mapped_hgnc_symbol"]:
            symbol_to_tcga_cna[row["mapped_hgnc_symbol"]].append(row["canonical_identifier"])
    cna_feature_rename = {}
    for symbol in cna_available:
        ids = sorted(set(symbol_to_tcga_cna[symbol]))
        cna_feature_rename[symbol] = f"TCGA_CNA_{'+'.join(ids)}__{symbol}"
    cna_patient = cna_patient.rename(columns=cna_feature_rename)
    cna_path_out = out / "m22_metabric_fixed_tcga_cna_panel_LOCAL_ONLY.csv"
    cna_patient.to_csv(cna_path_out, index=False)

    diagnostics = []
    for modality, frame in (("rna", rna), ("cna", cna_patient)):
        for row in numeric_diagnostics(frame, ["sample_id", "patient_id"]):
            diagnostics.append({"modality": modality, **row})
    write_csv(out / "m22_fixed_panel_matrix_diagnostics.csv", diagnostics)

    assayability = [
        {
            "modality": "rna",
            "mapped_unique_symbols": len(modality_symbols["rna"]),
            "available_metabric_symbols": len(rna_available),
            "unavailable_metabric_symbols": len(rna_missing),
            "availability_fraction": (
                len(rna_available) / len(modality_symbols["rna"])
                if modality_symbols["rna"] else 0.0
            ),
            "missing_symbols": " | ".join(rna_missing),
            "output_path": rna_path_out.relative_to(root).as_posix(),
        },
        {
            "modality": "cna",
            "mapped_unique_symbols": len(modality_symbols["cna"]),
            "available_metabric_symbols": len(cna_available),
            "unavailable_metabric_symbols": len(cna_missing),
            "availability_fraction": (
                len(cna_available) / len(modality_symbols["cna"])
                if modality_symbols["cna"] else 0.0
            ),
            "missing_symbols": " | ".join(cna_missing),
            "output_path": cna_path_out.relative_to(root).as_posix(),
        },
    ]
    write_csv(out / "m22_fixed_panel_assayability.csv", assayability)

    print("Fixed-panel assayability")
    print_table(
        assayability,
        [
            "modality", "mapped_unique_symbols", "available_metabric_symbols",
            "unavailable_metabric_symbols", "availability_fraction",
            "missing_symbols", "output_path"
        ],
    )

    print("\nMatrix diagnostics")
    print_table(
        diagnostics,
        ["modality", "feature", "nonmissing", "missing_fraction",
         "unique_nonmissing", "mean", "sd", "minimum", "maximum"],
        max_rows=120,
    )

    print("\nPatient and sample identifiers are stored only in LOCAL_ONLY matrices and are not printed.")
    print("No METABRIC survival or treatment outcome was read.")
    print("\nPASS: outcome-blind RNA and CNA fixed-panel matrices were built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
