#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from _stage15_utils import ensure_dirs, markdown_table, project_root, read_csv, sha256_file, write_csv, write_text


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    tables = root / "results/tables"
    required = {
        "stage14_decision": tables / "55_stage14_decision.csv",
        "stage14_curve_decomposition": tables / "54_ccw_rmst_decomposition.csv",
        "stage14_trace_manifest": tables / "53_ccw_trace_candidate_manifest.csv",
        "stage14_long_candidate": root / "data/derived/stage14_trace/53_candidate_01.csv",
        "ccw_original_checkpoint": tables / "43_ccw_bootstrap_CHECKPOINT.csv",
        "stage43_script": root / "scripts/43_ccw_sensitivity_bootstrap.py",
        "stage12_utils": root / "scripts/_stage12_utils.py",
        "stage14_utils": root / "scripts/_stage14_utils.py",
    }
    rows = []
    for item, path in required.items():
        rows.append({
            "item": item,
            "found": path.exists(),
            "path": str(path.relative_to(root)) if path.exists() else str(path),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
        })
    manifest = pd.DataFrame(rows)

    decision = ""
    if required["stage14_decision"].exists():
        df = read_csv(required["stage14_decision"])
        if not df.empty:
            decision = str(df.iloc[0].get("stage14_decision", ""))
    checks = pd.DataFrame([
        {
            "check": "all_required_inputs",
            "pass": bool(manifest["found"].all()),
            "detail": f"{int(manifest['found'].sum())}/{len(manifest)} found",
        },
        {
            "check": "stage14_requires_reestimated_truncation",
            "pass": decision == "RUN_REESTIMATED_CCW_WEIGHT_TRUNCATION_SENSITIVITY_BEFORE_PUBLICATION_BOOTSTRAP",
            "detail": decision,
        },
    ])
    status = "STAGE15_PREFLIGHT_PASSED" if bool(checks["pass"].all()) else "STAGE15_PREFLIGHT_FAILED"
    write_csv(manifest, tables / "56_stage15_input_manifest.csv")
    write_csv(checks, tables / "56_stage15_preflight_checks.csv")
    write_text(
        f"# Stage 15 preflight\n\n**Status:** `{status}`\n\n{markdown_table(manifest)}\n\n{markdown_table(checks)}",
        tables / "56_stage15_preflight.md",
    )
    print("=" * 116)
    print("STAGE 56 — STAGE 15 PREFLIGHT")
    print("=" * 116)
    print(manifest.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))
    print(f"\nStatus: {status}")
    return 0 if status == "STAGE15_PREFLIGHT_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
