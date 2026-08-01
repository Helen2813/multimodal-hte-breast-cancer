from __future__ import annotations

import json

from _metabric_m2_utils import (
    exact_col, exact_matrix_sample_set, load_config, out_dir, print_table,
    project_root, raw_dir, read_cbio, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M2.05 - CORRECTED EXACT MATRIX COVERAGE")
    print("=" * 124)

    sample = read_cbio(raw / cfg["files"]["clinical_sample"])
    sample_id_col = exact_col(sample.columns, ["SAMPLE_ID"])
    patient_id_col = exact_col(sample.columns, ["PATIENT_ID"])
    if sample_id_col is None or patient_id_col is None:
        raise RuntimeError("Clinical sample mapping is missing SAMPLE_ID or PATIENT_ID.")

    known_samples = set(sample[sample_id_col].dropna().astype(str).str.strip())
    sample_to_patient = dict(zip(
        sample[sample_id_col].astype(str).str.strip(),
        sample[patient_id_col].astype(str).str.strip(),
    ))

    specs = [
        ("mrna_raw", "\t"),
        ("mrna_zscores", "\t"),
        ("cna", "\t"),
        ("methylation", "\t"),
        ("gene_panel", "\t"),
        ("rna_cleaned", ","),
    ]

    rows = []
    sample_sets = {}
    for role, delimiter in specs:
        path = raw / cfg["files"][role]
        orientation, samples, details = exact_matrix_sample_set(path, known_samples, delimiter)
        patients = {sample_to_patient[s] for s in samples if s in sample_to_patient}
        sample_sets[role] = sorted(samples)
        rows.append({
            "role": role,
            "file": path.name,
            "orientation": orientation,
            "exact_sample_count": len(samples),
            "exact_patient_count": len(patients),
            "clinical_sample_coverage": len(samples) / len(known_samples),
            **details,
        })

    mutation = read_cbio(raw / cfg["files"]["mutations"])
    mutation_sample_col = exact_col(
        mutation.columns,
        ["TUMOR_SAMPLE_BARCODE", "SAMPLE_ID", "SAMPLE_IDENTIFIER"]
    )
    mutation_samples = (
        set(mutation[mutation_sample_col].dropna().astype(str).str.strip()) & known_samples
        if mutation_sample_col else set()
    )
    mutation_patients = {sample_to_patient[s] for s in mutation_samples if s in sample_to_patient}
    sample_sets["mutations"] = sorted(mutation_samples)
    rows.append({
        "role": "mutations",
        "file": cfg["files"]["mutations"],
        "orientation": "long_table",
        "exact_sample_count": len(mutation_samples),
        "exact_patient_count": len(mutation_patients),
        "clinical_sample_coverage": len(mutation_samples) / len(known_samples),
        "header_fields": len(mutation.columns),
        "header_sample_count": "",
        "first_column_sample_count": "",
        "first_field": mutation.columns[0] if len(mutation.columns) else "",
        "second_field": mutation.columns[1] if len(mutation.columns) > 1 else "",
    })

    corrections = [
        {
            "m1_item": "diagnosis_year",
            "m1_observed": "AGE_AT_DIAGNOSIS",
            "corrected_status": "ABSENT",
            "reason": "Age is not diagnosis year; the M1 heuristic matched it incorrectly.",
        },
        {
            "m1_item": "gene_panel exact coverage",
            "m1_observed": "50 samples",
            "corrected_status": str(next(r["exact_sample_count"] for r in rows if r["role"] == "gene_panel")),
            "reason": "M1 inspected only the first 50 data rows for sample-oriented matrices.",
        },
        {
            "m1_item": "rna_cleaned exact coverage",
            "m1_observed": "50 samples",
            "corrected_status": str(next(r["exact_sample_count"] for r in rows if r["role"] == "rna_cleaned")),
            "reason": "M1 inspected only the first 50 data rows for sample-oriented matrices.",
        },
    ]

    write_csv(out / "m05_exact_matrix_coverage.csv", rows)
    write_csv(out / "m05_m1_corrections.csv", corrections)
    (out / "m05_exact_sample_sets_LOCAL_ONLY.json").write_text(
        json.dumps(sample_sets, indent=2), encoding="utf-8"
    )

    print("Exact modality coverage")
    print_table(
        rows,
        ["role", "orientation", "exact_sample_count", "exact_patient_count",
         "clinical_sample_coverage", "header_fields", "first_field", "second_field"]
    )

    print("\nCorrections to M1")
    print_table(corrections, ["m1_item", "m1_observed", "corrected_status", "reason"])

    print("\nPASS: exact coverage was recomputed over complete sample columns/rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
