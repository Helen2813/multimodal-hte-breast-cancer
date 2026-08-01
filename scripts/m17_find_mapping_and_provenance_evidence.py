from __future__ import annotations

import json

from _metabric_m3b_utils import (
    candidate_mapping_path, load_cfg, output_dir, print_table,
    quick_hash, rel, root, scan_tokens, write_csv
)


def main() -> int:
    project = root()
    cfg = load_cfg(project)
    out = output_dir(project, cfg)

    print("=" * 124)
    print("METABRIC M3B.17 - LOCAL RNA/CpG MAPPING AND PROVENANCE EVIDENCE")
    print("=" * 124)

    selected = json.loads((out / "m16_tcga_selected_identifiers_LOCAL_ONLY.json").read_text(encoding="utf-8"))
    rna_ids = sorted(x for x in selected.get("rna", []) if x.startswith("ENSG"))
    cpg_ids = sorted(x for x in selected.get("methylation", []) if x.lower().startswith("cg"))
    rna_probe = rna_ids[:40]
    cpg_probe = cpg_ids[:40]

    rows = []
    for root_name in cfg["search_roots"]:
        base = (project / root_name).resolve()
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or not candidate_mapping_path(project, path, cfg):
                continue
            try:
                rna_hits = scan_tokens(path, rna_probe, int(cfg["maximum_bytes_scanned_per_file"]))
                cpg_hits = scan_tokens(path, cpg_probe, int(cfg["maximum_bytes_scanned_per_file"]))
            except OSError:
                continue
            if rna_hits or cpg_hits:
                rows.append({
                    "path": rel(project, path),
                    "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
                    "rna_probe_hits": len(rna_hits),
                    "rna_hit_examples": " | ".join(rna_hits[:10]),
                    "cpg_probe_hits": len(cpg_hits),
                    "cpg_hit_examples": " | ".join(cpg_hits[:10]),
                    "quick_sha256": quick_hash(path),
                })

    rows.sort(key=lambda r: (-int(r["rna_probe_hits"]), -int(r["cpg_probe_hits"]), r["path"]))
    write_csv(out / "m17_local_mapping_candidates.csv", rows, ["path", "size_mb", "rna_probe_hits", "rna_hit_examples", "cpg_probe_hits", "cpg_hit_examples", "quick_sha256"])

    print(f"Selected TCGA RNA Ensembl IDs: {len(rna_ids)}")
    print(f"Selected TCGA methylation CpG IDs: {len(cpg_ids)}")
    print("\nLocal mapping/provenance candidates")
    print_table(rows, ["path", "size_mb", "rna_probe_hits", "rna_hit_examples", "cpg_probe_hits", "cpg_hit_examples", "quick_sha256"], max_rows=60)
    print("\nNo candidate is accepted automatically; a mapping path and hash must be locked in M4.")
    print("\nPASS: mapping evidence collected without outcome inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
