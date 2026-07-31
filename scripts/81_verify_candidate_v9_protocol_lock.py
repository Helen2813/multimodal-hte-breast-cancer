#!/usr/bin/env python3
from __future__ import annotations

import json

import pandas as pd

from _stage20_utils import (
    dataframe_console,
    final_paths,
    load_stage20_config,
    project_root,
    sha256_file,
    write_csv,
)


def main() -> int:
    root = project_root()
    cfg = load_stage20_config(root)
    outputs = final_paths(root)
    manifest_path = outputs["hash_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for item in manifest["locked_files"]:
        path = root / item["path"]
        observed = sha256_file(path) if path.exists() else None
        rows.append({
            "path": item["path"],
            "exists": path.exists(),
            "expected_sha256": item["sha256"],
            "observed_sha256": observed,
            "match": observed == item["sha256"],
        })
    integrity = pd.DataFrame(rows)

    estimand = json.loads(outputs["primary_estimand"].read_text(encoding="utf-8"))
    bootstrap = json.loads(outputs["bootstrap_registry"].read_text(encoding="utf-8"))
    stage21 = json.loads((root / "stage21_config.json").read_text(encoding="utf-8"))
    semantic = pd.DataFrame([
        {
            "check": "protocol_id_consistent",
            "observed": estimand.get("protocol_id"),
            "expected": manifest.get("protocol_id"),
            "pass": estimand.get("protocol_id") == manifest.get("protocol_id"),
        },
        {
            "check": "partition_seeds_estimator_vs_bootstrap_registry",
            "observed": estimand.get("partition_base_seeds"),
            "expected": bootstrap.get("partition_base_seeds"),
            "pass": estimand.get("partition_base_seeds") == bootstrap.get("partition_base_seeds"),
        },
        {
            "check": "partition_seeds_registry_vs_stage21",
            "observed": bootstrap.get("partition_base_seeds"),
            "expected": stage21["full_bootstrap"].get("partition_base_seeds"),
            "pass": bootstrap.get("partition_base_seeds") == stage21["full_bootstrap"].get("partition_base_seeds"),
        },
        {
            "check": "bootstrap_repetitions_registry_vs_stage21",
            "observed": int(bootstrap.get("n_repetitions")),
            "expected": int(stage21["full_bootstrap"].get("n_repetitions")),
            "pass": int(bootstrap.get("n_repetitions")) == int(stage21["full_bootstrap"].get("n_repetitions")),
        },
        {
            "check": "g_min_registry_vs_stage21",
            "observed": float(estimand.get("primary_g_min")),
            "expected": float(stage21["full_bootstrap"].get("primary_g_min")),
            "pass": float(estimand.get("primary_g_min")) == float(stage21["full_bootstrap"].get("primary_g_min")),
        },
        {
            "check": "horizon_registry_vs_stage21",
            "observed": float(estimand.get("horizon_days")),
            "expected": float(stage21["full_bootstrap"].get("horizon_days")),
            "pass": float(estimand.get("horizon_days")) == float(stage21["full_bootstrap"].get("horizon_days")),
        },
        {
            "check": "lock_sentinel_exists",
            "observed": outputs["lock_sentinel"].exists(),
            "expected": True,
            "pass": outputs["lock_sentinel"].exists(),
        },
        {
            "check": "protocol_status",
            "observed": manifest.get("protocol_status"),
            "expected": cfg["protocol_status_after_lock"],
            "pass": manifest.get("protocol_status") == cfg["protocol_status_after_lock"],
        },
    ])

    all_pass = bool(integrity["match"].all()) and bool(semantic["pass"].all())
    summary = pd.DataFrame([{
        "protocol_id": manifest.get("protocol_id"),
        "locked_file_count": len(integrity),
        "locked_file_hashes_pass": bool(integrity["match"].all()),
        "semantic_checks_pass": bool(semantic["pass"].all()),
        "all_integrity_checks_pass": all_pass,
        "stage21_publication_bootstrap_authorized": all_pass,
    }])

    tables = root / "results/tables"
    write_csv(integrity, tables / "81_candidate_v9_lock_file_integrity.csv")
    write_csv(semantic, tables / "81_candidate_v9_lock_semantic_integrity.csv")
    write_csv(summary, tables / "81_candidate_v9_lock_integrity_summary.csv")

    print("=" * 124)
    print("STAGE 81 - CANDIDATE V9 PROTOCOL LOCK INTEGRITY")
    print("=" * 124)
    print("File integrity")
    print(dataframe_console(integrity))
    print("\nSemantic integrity")
    print(dataframe_console(semantic))
    print("\nIntegrity summary")
    print(dataframe_console(summary))
    if not all_pass:
        raise RuntimeError("Candidate V9 protocol lock integrity failed. Do not run Stage 21.")
    print("\nPASS: Candidate V9 is locked. Stage 21 may be started without modifying any locked file.")
    print("Suggested Git commands after reviewing this log:")
    print('  git add .')
    print('  git commit -m "Lock Paper A Candidate V9 analysis protocol"')
    print('  git tag protocol-lock-paper-A-v9')
    print('  git push origin main --tags')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
