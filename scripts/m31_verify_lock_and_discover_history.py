from __future__ import annotations

import json
import sys
from pathlib import Path

from _metabric_m6_utils import (
    load_config, out_dir, print_table, project_root, rel, require_lifelines,
    sha256, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M6.31 - LOCK VERIFICATION, DEPENDENCIES, AND HISTORICAL ARTIFACT DISCOVERY")
    print("=" * 124)

    m5 = root / cfg["metabric_m5_dir"]
    manifest_path = m5 / "m30_protocol_hash_manifest.csv"
    protocol_path = m5 / "m30_metabric_dual_track_protocol.json"
    decision_path = m5 / "m30_protocol_decision.csv"

    if not all(path.exists() for path in (manifest_path, protocol_path, decision_path)):
        raise FileNotFoundError("M5 protocol lock files are incomplete.")

    import pandas as pd
    manifest = pd.read_csv(manifest_path, dtype=str)
    verification_rows = []
    for _, row in manifest.iterrows():
        path = root / row["path"]
        observed = sha256(path) if path.exists() else ""
        verification_rows.append({
            "path": row["path"],
            "exists": path.exists(),
            "expected_sha256": row["sha256"],
            "observed_sha256": observed,
            "pass": path.exists() and observed == row["sha256"],
        })
    write_csv(out / "m31_m5_lock_verification.csv", verification_rows)
    if not all(bool(row["pass"]) for row in verification_rows):
        raise RuntimeError("M5 lock verification failed.")

    versions = {}
    for name in ("numpy", "pandas", "scipy", "sklearn"):
        module = __import__(name)
        versions[name] = getattr(module, "__version__", "unknown")
    versions["lifelines"] = require_lifelines()
    versions["python"] = sys.version.split()[0]
    (out / "m31_dependency_versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )

    search_tokens = (
        "metabric", "external_validation", "external-validation",
        "min100", "min200", "0.7197", "0.7200", "0.7639"
    )
    artifact_rows = []
    roots = [root / "data", root / "results", root / "scripts"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = rel(root, path).lower()
            name_hit = [token for token in search_tokens if token in relative]
            content_hits = []
            if path.suffix.lower() in {".py", ".ps1", ".json", ".csv", ".tsv", ".txt", ".md"} and path.stat().st_size < 20 * 1024 * 1024:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace").lower()
                    content_hits = [token for token in search_tokens if token in text]
                except OSError:
                    content_hits = []
            if name_hit or content_hits:
                artifact_rows.append({
                    "path": rel(root, path),
                    "size_mb": round(path.stat().st_size / (1024 ** 2), 4),
                    "name_hits": " | ".join(name_hit),
                    "content_hits": " | ".join(content_hits),
                    "sha256": sha256(path),
                })

    artifact_rows.sort(key=lambda row: row["path"])
    write_csv(out / "m31_historical_metabric_artifacts.csv", artifact_rows)

    print("M5 lock verification")
    print_table(verification_rows, ["path", "exists", "pass", "observed_sha256"])

    print("\nDependency versions")
    for key, value in versions.items():
        print(f"  {key}: {value}")

    print("\nHistorical METABRIC artifacts")
    print_table(
        artifact_rows,
        ["path", "size_mb", "name_hits", "content_hits", "sha256"],
        max_rows=150,
    )

    print("\nSource-grounded historical benchmark to reproduce")
    print(json.dumps(cfg["historical_external_benchmark"], indent=2))

    print("\nPASS: M5 lock intact. Historical artifacts discovered without fitting a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
