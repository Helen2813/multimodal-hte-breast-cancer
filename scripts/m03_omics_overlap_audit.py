from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from _metabric_m1_utils import (
    choose_column, count_lines, load_config, out_dir, print_table, project_root,
    raw_dir, read_cbio_table, sample_id_columns_from_matrix, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    sample = read_cbio_table(raw / cfg["expected_files"]["clinical_sample"])
    sample_id_col = choose_column(sample.columns, ["SAMPLE_ID"], ["SAMPLE", "ID"])
    patient_id_col = choose_column(sample.columns, ["PATIENT_ID"], ["PATIENT", "ID"])
    if sample_id_col is None:
        raise RuntimeError("Could not resolve SAMPLE_ID in data_clinical_sample.txt")

    known_samples = set(sample[sample_id_col].dropna().astype(str).str.strip())
    known_patients = set(sample[patient_id_col].dropna().astype(str).str.strip()) if patient_id_col else set()

    print("=" * 124)
    print("METABRIC M1.03 - OMICS SCHEMA AND PATIENT/SAMPLE OVERLAP AUDIT")
    print("=" * 124)
    print(f"Clinical sample IDs: {len(known_samples)}")
    print(f"Clinical patient IDs: {len(known_patients)}")

    matrix_roles = ["mrna_raw", "mrna_zscores", "cna", "methylation", "gene_panel", "rna_cleaned"]
    rows = []
    for role in matrix_roles:
        name = cfg["expected_files"][role]
        path = raw / name
        if not path.exists():
            rows.append({
                "role": role, "file": name, "found": False, "size_mb": "",
                "line_count": "", "orientation": "", "header_fields": "",
                "detected_sample_ids": 0, "overlap_with_clinical_samples": 0,
                "clinical_sample_coverage": 0.0,
                "first_field": "", "second_field": ""
            })
            continue

        details = sample_id_columns_from_matrix(path, known_samples)
        detected = set(details.pop("sample_ids_detected"))
        overlap = detected & known_samples
        rows.append({
            "role": role,
            "file": name,
            "found": True,
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
            "line_count": count_lines(path),
            "orientation": details["orientation"],
            "header_fields": details["header_fields"],
            "detected_sample_ids": len(detected),
            "overlap_with_clinical_samples": len(overlap),
            "clinical_sample_coverage": len(overlap) / len(known_samples) if known_samples else 0.0,
            "first_field": details["first_field"],
            "second_field": details["second_field"],
            "delimiter": details["delimiter"],
        })

    mutation_path = raw / cfg["expected_files"]["mutations"]
    mutation_summary = {
        "role": "mutations",
        "file": mutation_path.name,
        "found": mutation_path.exists(),
        "size_mb": round(mutation_path.stat().st_size / (1024 ** 2), 3) if mutation_path.exists() else "",
        "rows": 0,
        "sample_column": "",
        "unique_mutation_samples": 0,
        "overlap_with_clinical_samples": 0,
        "clinical_sample_coverage": 0.0,
        "unique_genes": 0,
    }
    if mutation_path.exists():
        mut = read_cbio_table(mutation_path)
        mut_sample_col = choose_column(
            mut.columns,
            ["TUMOR_SAMPLE_BARCODE", "SAMPLE_ID", "SAMPLE_IDENTIFIER"],
            ["SAMPLE", "BARCODE"],
        )
        gene_col = choose_column(mut.columns, ["HUGO_SYMBOL", "GENE"], ["HUGO", "GENE"])
        mutation_summary["rows"] = int(len(mut))
        mutation_summary["sample_column"] = mut_sample_col or ""
        if mut_sample_col:
            mut_samples = set(mut[mut_sample_col].dropna().astype(str).str.strip())
            overlap = mut_samples & known_samples
            mutation_summary["unique_mutation_samples"] = len(mut_samples)
            mutation_summary["overlap_with_clinical_samples"] = len(overlap)
            mutation_summary["clinical_sample_coverage"] = len(overlap) / len(known_samples) if known_samples else 0.0
        if gene_col:
            mutation_summary["unique_genes"] = int(mut[gene_col].dropna().astype(str).nunique())

    write_csv(out / "m03_matrix_schema_and_overlap.csv", rows)
    write_csv(out / "m03_mutation_schema_and_overlap.csv", [mutation_summary])

    print("\nLarge matrix schema and overlap")
    print_table(
        rows,
        ["role", "file", "found", "size_mb", "line_count", "orientation",
         "header_fields", "overlap_with_clinical_samples", "clinical_sample_coverage",
         "first_field", "second_field"]
    )

    print("\nMutation schema and overlap")
    print_table(
        [mutation_summary],
        ["file", "rows", "sample_column", "unique_mutation_samples",
         "overlap_with_clinical_samples", "clinical_sample_coverage", "unique_genes"]
    )

    available = [r["role"] for r in rows if r["found"] and int(r["overlap_with_clinical_samples"]) > 0]
    if mutation_summary["found"] and int(mutation_summary["overlap_with_clinical_samples"]) > 0:
        available.append("mutations")

    summary = {
        "clinical_sample_ids": len(known_samples),
        "clinical_patient_ids": len(known_patients),
        "omics_modalities_with_detected_clinical_sample_overlap": available,
        "raw_mrna_primary_for_audit": cfg["expected_files"]["mrna_raw"],
        "cleaned_rna_treated_as_derived_until_provenance_is_verified": cfg["expected_files"]["rna_cleaned"],
    }
    (out / "m03_omics_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nOmics audit summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nPASS: omics overlap audit completed by streaming headers/rows. Large matrices were not loaded into memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
