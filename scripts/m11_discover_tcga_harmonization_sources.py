from __future__ import annotations

import json
from pathlib import Path

from _metabric_m3_utils import (
    classify_identifiers, infer_delimiter, is_excluded, load_config,
    modality_score, noncomment_lines, out_dir, parse_fields, print_table,
    project_root, quick_sha256, read_text_prefix, rel, write_csv
)


def inspect_candidate(root: Path, path: Path, cfg: dict) -> dict:
    prefix = read_text_prefix(path, int(cfg["header_read_bytes"]))
    lines = noncomment_lines(prefix, limit=5)
    first = lines[0] if lines else ""
    delimiter = infer_delimiter(first) if first else "\t"
    fields = parse_fields(first, delimiter) if first else []
    classification = classify_identifiers(fields[:5000])
    return {
        "path": rel(root, path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "header_fields": len(fields),
        "first_field": fields[0] if fields else "",
        "second_field": fields[1] if len(fields) > 1 else "",
        "identifier_type_in_header": classification["identifier_type"],
        "header_gene_like_count": classification["gene_like_count"],
        "header_ensembl_count": classification["ensembl_count"],
        "fields_preview": " | ".join(fields[:8]),
        "quick_sha256": quick_sha256(path),
        "_fields": fields,
    }


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M3.11 - DISCOVER TCGA HARMONIZATION SOURCES")
    print("=" * 124)

    extensions = set(x.lower() for x in cfg["candidate_extensions"])
    max_bytes = float(cfg["maximum_candidate_file_size_gb"]) * (1024 ** 3)
    candidates = []

    for root_name in cfg["search_roots"]:
        base = (root / root_name).resolve()
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            if path.stat().st_size > max_bytes:
                continue
            if is_excluded(root, path, cfg["exclude_path_tokens"]):
                continue
            try:
                candidates.append(inspect_candidate(root, path, cfg))
            except Exception as exc:
                candidates.append({
                    "path": rel(root, path),
                    "name": path.name,
                    "extension": path.suffix.lower(),
                    "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
                    "delimiter": "",
                    "header_fields": 0,
                    "first_field": "",
                    "second_field": "",
                    "identifier_type_in_header": "unreadable",
                    "header_gene_like_count": 0,
                    "header_ensembl_count": 0,
                    "fields_preview": f"ERROR: {type(exc).__name__}: {exc}",
                    "quick_sha256": quick_sha256(path),
                    "_fields": [],
                })

    modalities = ["clinical", "rna", "cna", "mutations", "methylation", "annotation", "pathway_gmt"]
    ranked = []
    top_n = int(cfg["top_candidates_per_modality"])
    for modality in modalities:
        scored = []
        for item in candidates:
            score = modality_score(root / item["path"], item["_fields"], modality)
            if score > 0:
                row = {k: v for k, v in item.items() if not k.startswith("_")}
                row["modality"] = modality
                row["score"] = score
                scored.append(row)
        scored.sort(key=lambda r: (-int(r["score"]), r["path"].lower()))
        for rank, row in enumerate(scored[:top_n], 1):
            row["rank"] = rank
            ranked.append(row)

    write_csv(out / "m11_all_scanned_candidate_files.csv", [
        {k: v for k, v in item.items() if not k.startswith("_")} for item in candidates
    ])
    write_csv(out / "m11_ranked_tcga_source_candidates.csv", ranked)

    summary = []
    for modality in modalities:
        rows = [r for r in ranked if r["modality"] == modality]
        summary.append({
            "modality": modality,
            "candidate_count_printed": len(rows),
            "top_path": rows[0]["path"] if rows else "",
            "top_score": rows[0]["score"] if rows else "",
            "second_score": rows[1]["score"] if len(rows) > 1 else "",
            "score_margin": (
                int(rows[0]["score"]) - int(rows[1]["score"])
                if len(rows) > 1 else (int(rows[0]["score"]) if rows else 0)
            ),
        })
    write_csv(out / "m11_candidate_summary.csv", summary)

    print(f"Candidate files scanned: {len(candidates)}")
    print("\nCandidate summary")
    print_table(summary, ["modality", "candidate_count_printed", "top_path", "top_score", "second_score", "score_margin"])

    for modality in modalities:
        rows = [r for r in ranked if r["modality"] == modality]
        print(f"\nTop {modality} candidates")
        print_table(
            rows,
            ["rank", "score", "path", "size_mb", "header_fields",
             "identifier_type_in_header", "header_gene_like_count",
             "header_ensembl_count", "fields_preview"],
            max_rows=top_n,
        )

    print("\nPASS: TCGA source candidates were discovered without selecting a favorable file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
