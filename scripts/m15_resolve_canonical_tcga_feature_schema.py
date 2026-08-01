from __future__ import annotations

import json

from _metabric_m3b_utils import (
    classify_columns, load_cfg, output_dir, print_table, quick_hash,
    read_header, rel, root, row_count, write_csv
)


def main() -> int:
    project = root()
    cfg = load_cfg(project)
    out = output_dir(project, cfg)

    print("=" * 124)
    print("METABRIC M3B.15 - CANONICAL TCGA PATIENT-LEVEL FEATURE SCHEMA")
    print("=" * 124)

    candidates = []
    selected = None
    selected_registry = None
    for priority, item in enumerate(cfg["preferred_tcga_tables"], 1):
        path = (project / item).resolve()
        if not path.exists():
            candidates.append({"priority": priority, "path": item, "exists": False})
            continue
        header = read_header(path)
        registry = classify_columns(header, cfg["prefixes"])
        counts = {}
        for row in registry:
            counts[row["modality"]] = counts.get(row["modality"], 0) + 1
        candidates.append({
            "priority": priority,
            "path": rel(project, path),
            "exists": True,
            "rows": row_count(path),
            "columns": len(header),
            "modality_counts": json.dumps(counts, sort_keys=True),
            "quick_sha256": quick_hash(path),
        })
        if selected is None and any(counts.get(m, 0) for m in ("rna", "cna", "mutations", "methylation")):
            selected, selected_registry = path, registry

    write_csv(out / "m15_tcga_table_candidates.csv", candidates)
    if selected is None or selected_registry is None:
        raise RuntimeError("No preferred TCGA table with recognized modality prefixes was found.")

    summary = []
    for modality in sorted({r["modality"] for r in selected_registry}):
        rows = [r for r in selected_registry if r["modality"] == modality]
        type_counts = {}
        for row in rows:
            type_counts[row["identifier_type"]] = type_counts.get(row["identifier_type"], 0) + 1
        summary.append({
            "modality": modality,
            "columns": len(rows),
            "identifier_type_counts": json.dumps(type_counts, sort_keys=True),
            "examples": " | ".join(r["column"] for r in rows[: int(cfg["print_feature_examples"])]),
        })

    write_csv(out / "m15_tcga_feature_registry.csv", selected_registry)
    write_csv(out / "m15_tcga_feature_summary.csv", summary)
    (out / "m15_selected_tcga_table.json").write_text(json.dumps({
        "selected_path": rel(project, selected),
        "quick_sha256": quick_hash(selected),
        "rows": row_count(selected),
        "columns": len(selected_registry),
        "selection_rule": "first prespecified preferred table with recognized modality prefixes; no outcome inspected"
    }, indent=2), encoding="utf-8")

    print("Preferred candidates")
    print_table(candidates, ["priority", "path", "exists", "rows", "columns", "modality_counts", "quick_sha256"])
    print(f"\nSelected canonical table: {rel(project, selected)}")
    print("\nPrefix-aware feature summary")
    print_table(summary, ["modality", "columns", "identifier_type_counts", "examples"])
    print("\nPASS: canonical TCGA feature schema resolved without fitting a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
