from __future__ import annotations

import csv
import json
from collections import OrderedDict

import numpy as np

from _metabric_m2_utils import (
    load_config, out_dir, print_table, project_root, raw_dir,
    read_header, write_csv
)


def float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)
    settings = cfg["rna_provenance"]

    raw_path = raw / cfg["files"]["mrna_raw"]
    clean_path = raw / cfg["files"]["rna_cleaned"]

    print("=" * 124)
    print("METABRIC M2.07 - CLEANED RNA PROVENANCE CHECK")
    print("=" * 124)

    raw_header = read_header(raw_path, "\t")
    clean_header = read_header(clean_path, ",")
    if len(raw_header) < 3 or len(clean_header) < 3:
        raise RuntimeError("RNA headers are unexpectedly short.")

    raw_samples = raw_header[2:]
    clean_genes = clean_header[1:]
    clean_gene_lookup = {gene: idx + 1 for idx, gene in enumerate(clean_genes) if gene}

    selected_genes: list[str] = []
    raw_gene_rows: dict[str, list[str]] = {}
    with raw_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")), delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            if gene and gene in clean_gene_lookup and gene not in raw_gene_rows:
                raw_gene_rows[gene] = row
                selected_genes.append(gene)
                if len(selected_genes) >= int(settings["gene_count"]):
                    break

    if len(selected_genes) < 5:
        raise RuntimeError("Too few common genes were found between raw and cleaned RNA.")

    raw_sample_index = {sample: idx + 2 for idx, sample in enumerate(raw_samples)}
    selected_clean_rows: OrderedDict[str, list[str]] = OrderedDict()
    with clean_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            if not row:
                continue
            sample = row[0].strip().strip('"')
            if sample in raw_sample_index:
                selected_clean_rows[sample] = row
                if len(selected_clean_rows) >= int(settings["sample_count"]):
                    break

    if len(selected_clean_rows) < 5:
        raise RuntimeError("Too few common samples were found between raw and cleaned RNA.")

    pair_rows = []
    raw_values = []
    clean_values = []
    for sample, clean_row in selected_clean_rows.items():
        rsi = raw_sample_index[sample]
        for gene in selected_genes:
            raw_row = raw_gene_rows[gene]
            cgi = clean_gene_lookup[gene]
            rv = float_or_nan(raw_row[rsi]) if rsi < len(raw_row) else np.nan
            cv = float_or_nan(clean_row[cgi]) if cgi < len(clean_row) else np.nan
            if np.isfinite(rv) and np.isfinite(cv):
                raw_values.append(rv)
                clean_values.append(cv)
                pair_rows.append({
                    "sample_id": sample,
                    "gene": gene,
                    "raw_value": rv,
                    "cleaned_value": cv,
                    "absolute_difference": abs(rv - cv),
                })

    if len(raw_values) >= 2 and np.std(raw_values) > 0 and np.std(clean_values) > 0:
        corr = float(np.corrcoef(raw_values, clean_values)[0, 1])
    else:
        corr = np.nan
    median_abs_diff = float(np.median(np.abs(np.asarray(raw_values) - np.asarray(clean_values)))) if raw_values else np.nan
    max_abs_diff = float(np.max(np.abs(np.asarray(raw_values) - np.asarray(clean_values)))) if raw_values else np.nan

    result = {
        "raw_file": raw_path.name,
        "cleaned_file": clean_path.name,
        "selected_samples": len(selected_clean_rows),
        "selected_genes": len(selected_genes),
        "numeric_pairs": len(raw_values),
        "pearson_correlation": corr,
        "median_absolute_difference": median_abs_diff,
        "maximum_absolute_difference": max_abs_diff,
        "correlation_gate": corr >= float(settings["correlation_pass"]) if np.isfinite(corr) else False,
        "absolute_difference_gate": median_abs_diff <= float(settings["median_absolute_difference_pass"]) if np.isfinite(median_abs_diff) else False,
        "minimum_pair_gate": len(raw_values) >= int(settings["minimum_pairs"]),
    }
    result["provenance_status"] = (
        "CLEANED_RNA_NUMERICALLY_MATCHES_RAW_TRANSPOSED_VALUES"
        if result["correlation_gate"] and result["absolute_difference_gate"] and result["minimum_pair_gate"]
        else "CLEANED_RNA_REQUIRES_PROVENANCE_REVIEW"
    )

    public_pairs = [
        {
            "sample_index": i // len(selected_genes) + 1,
            "gene": row["gene"],
            "raw_value": row["raw_value"],
            "cleaned_value": row["cleaned_value"],
            "absolute_difference": row["absolute_difference"],
        }
        for i, row in enumerate(pair_rows)
    ]
    write_csv(out / "m07_rna_provenance_pairs_deidentified.csv", public_pairs)
    write_csv(out / "m07_rna_provenance_summary.csv", [result])
    (out / "m07_rna_provenance_selected_samples_LOCAL_ONLY.json").write_text(
        json.dumps(list(selected_clean_rows.keys()), indent=2), encoding="utf-8"
    )

    print("RNA provenance summary")
    print_table(
        [result],
        ["selected_samples", "selected_genes", "numeric_pairs", "pearson_correlation",
         "median_absolute_difference", "maximum_absolute_difference", "provenance_status"]
    )

    print("\nSelected genes")
    print("  " + ", ".join(selected_genes))

    print("\nPatient/sample identifiers are not printed.")
    print("\nPASS: cleaned RNA provenance was checked against raw numeric values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
