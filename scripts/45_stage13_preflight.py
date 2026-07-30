#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from _stage13_utils import (
    bootstrap_stats_from_summary_or_checkpoint,
    csv_inventory,
    ensure_output_dirs,
    find_point_rows,
    load_config,
    markdown_table,
    numeric,
    project_root,
    sha256_file,
    write_csv,
    write_json,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_output_dirs(root)
    cfg = load_config(root)
    tables = root / "results" / "tables"

    landmark, ccw, landmark_path, ccw_path = find_point_rows(root)
    landmark_boot = bootstrap_stats_from_summary_or_checkpoint(root, "landmark")
    ccw_boot = bootstrap_stats_from_summary_or_checkpoint(root, "ccw")

    required_paths = {
        "stage13_config": root / "stage13_config.json",
        "stage11_candidate_plan": root / "paper_A_treatment_effects" / "analysis_plan_CANDIDATE_V2.md",
        "stage11_estimand_config": root / "paper_A_treatment_effects" / "primary_estimand_CANDIDATE_V2.json",
        "landmark_point_table": landmark_path,
        "ccw_point_table": ccw_path,
        "landmark_checkpoint": tables / "42_landmark_bootstrap_CHECKPOINT.csv",
        "ccw_checkpoint": tables / "43_ccw_bootstrap_CHECKPOINT.csv",
        "ccw_clone_flow": tables / "39_ccw_clone_flow.csv",
        "ccw_feasibility_decision": tables / "39_ccw_feasibility_decision.csv",
    }

    rows = []
    for item, path in required_paths.items():
        rows.append(
            {
                "item": item,
                "required": item not in {"stage11_candidate_plan", "stage11_estimand_config"},
                "found": path.exists(),
                "path": str(path.relative_to(root)) if path.exists() else str(path),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    manifest = pd.DataFrame(rows)

    expected_lm = cfg["expected_point_estimates"]["landmark_rmst_difference_days"]
    expected_ccw = cfg["expected_point_estimates"]["ccw_rmst_difference_days"]
    tol = cfg["replication_tolerance_days"]
    checks = pd.DataFrame(
        [
            {
                "check": "landmark_exact_replication",
                "observed": numeric(landmark.get("estimate_days")),
                "expected": expected_lm,
                "tolerance": tol,
                "pass": abs(numeric(landmark.get("estimate_days")) - expected_lm) <= tol,
            },
            {
                "check": "ccw_point_estimate_recovered",
                "observed": numeric(ccw.get("estimate_days")),
                "expected": expected_ccw,
                "tolerance": tol,
                "pass": abs(numeric(ccw.get("estimate_days")) - expected_ccw) <= tol,
            },
            {
                "check": "landmark_bootstrap_checkpoint_nonempty",
                "observed": landmark_boot.successful_reps,
                "expected": 1,
                "tolerance": np.nan,
                "pass": landmark_boot.successful_reps >= 1,
            },
            {
                "check": "ccw_bootstrap_checkpoint_nonempty",
                "observed": ccw_boot.successful_reps,
                "expected": 1,
                "tolerance": np.nan,
                "pass": ccw_boot.successful_reps >= 1,
            },
        ]
    )

    required_ok = bool(manifest.loc[manifest["required"], "found"].all())
    checks_ok = bool(checks["pass"].all())
    status = "STAGE13_PREFLIGHT_PASSED" if required_ok and checks_ok else "STAGE13_PREFLIGHT_FAILED"

    write_csv(manifest, tables / "45_stage13_input_manifest.csv")
    write_csv(checks, tables / "45_stage13_preflight_checks.csv")
    write_csv(csv_inventory(root), tables / "45_stage13_csv_inventory.csv")
    write_json(
        {
            "status": status,
            "required_inputs_found": required_ok,
            "replication_checks_passed": checks_ok,
            "landmark_point_path": str(landmark_path.relative_to(root)),
            "ccw_point_path": str(ccw_path.relative_to(root)),
            "landmark_bootstrap_source": landmark_boot.source_path,
            "ccw_bootstrap_source": ccw_boot.source_path,
        },
        tables / "45_stage13_preflight.json",
    )
    report = f"""# Stage 13 preflight

**Status:** `{status}`

## Input manifest

{markdown_table(manifest)}

## Replication and checkpoint checks

{markdown_table(checks)}
"""
    write_text(report, tables / "45_stage13_preflight.md")

    print("=" * 112)
    print("STAGE 45 — STAGE 13 PREFLIGHT")
    print("=" * 112)
    print(manifest.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))
    print(f"\nStatus: {status}")
    return 0 if status == "STAGE13_PREFLIGHT_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
