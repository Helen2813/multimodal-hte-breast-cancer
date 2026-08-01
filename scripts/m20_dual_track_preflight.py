from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _metabric_m4_utils import (
    load_config, load_m3b_registry, out_dir, print_table, project_root,
    quick_sha256, raw_dir, rel, selected_identifiers, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M4.20 - DUAL-TRACK PREFLIGHT AND M3B INTERPRETATION CORRECTION")
    print("=" * 124)

    registry = load_m3b_registry(root, cfg)
    source_table = (root / cfg["tcga_canonical_table"]).resolve()

    checks = [
        {
            "check": "canonical TCGA patient-level table exists",
            "observed": rel(root, source_table),
            "pass": source_table.exists(),
        },
        {
            "check": "M3B feature registry rows",
            "observed": len(registry),
            "pass": len(registry) > 0,
        },
        {
            "check": "METABRIC cleaned RNA exists",
            "observed": cfg["metabric_files"]["rna_cleaned"],
            "pass": (raw / cfg["metabric_files"]["rna_cleaned"]).exists(),
        },
        {
            "check": "METABRIC CNA exists",
            "observed": cfg["metabric_files"]["cna"],
            "pass": (raw / cfg["metabric_files"]["cna"]).exists(),
        },
        {
            "check": "METABRIC mutation and panel matrix exist",
            "observed": f"{cfg['metabric_files']['mutations']} | {cfg['metabric_files']['gene_panel_matrix']}",
            "pass": (
                (raw / cfg["metabric_files"]["mutations"]).exists()
                and (raw / cfg["metabric_files"]["gene_panel_matrix"]).exists()
            ),
        },
    ]
    write_csv(out / "m20_preflight_checks.csv", checks)

    if not all(bool(row["pass"]) for row in checks):
        raise RuntimeError("M4 preflight failed. Review m20_preflight_checks.csv")

    feature_rows = []
    for modality in ("rna", "cna", "mutations", "methylation"):
        feature_rows.extend(selected_identifiers(registry, modality))

    summary = []
    for modality in ("rna", "cna", "mutations", "methylation"):
        subset = [row for row in feature_rows if row["modality"] == modality]
        counts = {}
        for row in subset:
            counts[row["identifier_type"]] = counts.get(row["identifier_type"], 0) + 1
        summary.append({
            "modality": modality,
            "selected_columns": len(subset),
            "identifier_counts": json.dumps(counts, sort_keys=True),
            "examples": " | ".join(row["tcga_column"] for row in subset[:10]),
        })

    write_csv(out / "m20_selected_tcga_feature_identifiers.csv", feature_rows)
    write_csv(out / "m20_selected_feature_summary.csv", summary)

    correction = {
        "m3b_cna_zero_overlap_interpretation": (
            "The 50 selected TCGA CNA variables are Ensembl IDs, not HUGO symbols. "
            "A zero direct overlap with METABRIC HUGO rows is expected before Ensembl-to-HGNC mapping."
        ),
        "m3b_mapping_candidate_interpretation": (
            "Files containing selected Ensembl IDs are provenance evidence, not necessarily Ensembl-to-HUGO mapping tables."
        ),
        "track_a": (
            "Fixed TCGA-panel transport: mapping and transformations are outcome-blind in METABRIC."
        ),
        "track_b": (
            "Independent Paper-1 replication: METABRIC performs its own nested feature selection, reported separately."
        ),
        "canonical_tcga_table_sha256_quick": quick_sha256(source_table),
    }
    (out / "m20_interpretation_correction.json").write_text(
        json.dumps(correction, indent=2), encoding="utf-8"
    )

    print("Preflight checks")
    print_table(checks, ["check", "observed", "pass"])

    print("\nSelected TCGA feature identifiers")
    print_table(summary, ["modality", "selected_columns", "identifier_counts", "examples"])

    print("\nCorrected interpretation")
    for key, value in correction.items():
        print(f"  {key}: {value}")

    print("\nPASS: M4 preflight complete. No METABRIC outcome was read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
