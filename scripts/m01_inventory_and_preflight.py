from __future__ import annotations

from pathlib import Path

from _metabric_m1_utils import (
    count_lines, infer_delimiter, load_config, out_dir, print_table, project_root,
    quick_fingerprint, raw_dir, read_first_noncomment_line, rel, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M1.01 - RAW FILE INVENTORY AND PREFLIGHT")
    print("=" * 124)
    print(f"Raw directory: {raw}")

    rows = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file():
            continue
        first = read_first_noncomment_line(path)
        delim = infer_delimiter(first) if first else ""
        rows.append({
            "path": rel(root, path),
            "name": path.name,
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
            "line_count": count_lines(path),
            "delimiter": "\\t" if delim == "\t" else delim,
            "first_noncomment_fields": len(first.split(delim)) if first and delim else 0,
            "quick_fingerprint": quick_fingerprint(path, int(cfg["quick_fingerprint_bytes"])),
        })

    expected_rows = []
    by_name = {p.name: p for p in raw.iterdir() if p.is_file()}
    for role, name in cfg["expected_files"].items():
        path = by_name.get(name)
        expected_rows.append({
            "role": role,
            "expected_name": name,
            "found": bool(path),
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3) if path else "",
        })

    write_csv(out / "m01_file_inventory.csv", rows)
    write_csv(out / "m01_expected_file_preflight.csv", expected_rows)

    print("\nExpected METABRIC files")
    print_table(expected_rows, ["role", "expected_name", "found", "size_mb"])

    missing_required = [
        r["role"] for r in expected_rows
        if r["role"] in {"clinical_patient", "clinical_sample"} and not r["found"]
    ]
    if missing_required:
        raise RuntimeError(f"Required clinical files are missing: {missing_required}")

    print("\nComplete raw inventory")
    print_table(rows, ["name", "size_mb", "line_count", "delimiter", "first_noncomment_fields", "quick_fingerprint"])

    print("\nPASS: raw METABRIC inventory completed. No raw file was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
