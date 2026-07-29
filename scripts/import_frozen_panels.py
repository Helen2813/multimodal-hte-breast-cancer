from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modality_hte.data.provenance import sha256_file

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".pq"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy frozen Paper 1 panel tables and record their provenance."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--manifest",
        default="data/manifests/paper1_panels.yaml",
        help="Output YAML manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    destination = Path(args.destination)
    manifest_path = Path(args.manifest)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No supported panel tables found in {source_dir}. "
            f"Expected one of: {sorted(SUPPORTED_SUFFIXES)}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for source in files:
        target = destination / source.name
        shutil.copy2(source, target)
        entries.append(
            {
                "file": str(target),
                "source_file": str(source),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )

    manifest = {
        "source_repository": args.source_repository,
        "source_commit": args.source_commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    print(f"Imported {len(entries)} panel files.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
