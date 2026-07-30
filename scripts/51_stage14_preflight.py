#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from _stage14_utils import (
    ensure_dirs,
    load_config,
    markdown_table,
    point_estimate_table,
    project_root,
    read_csv,
    sha256_file,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    tables = root / "results/tables"
    cfg = load_config(root)

    required = {
        "stage13_decision": tables / "50_stage13_decision.csv",
        "stage13_centering": tables / "47_bootstrap_centering_audit.csv",
        "stage13_ccw_invariants": tables / "49_ccw_invariant_checks.csv",
        "landmark_checkpoint": tables / "42_landmark_bootstrap_CHECKPOINT.csv",
        "ccw_checkpoint": tables / "43_ccw_bootstrap_CHECKPOINT.csv",
        "stage41_script": root / "scripts/41_replicate_estimators.py",
        "stage12_utils": root / "scripts/_stage12_utils.py",
    }
    point_path, point_row = point_estimate_table(root)
    required["ccw_point_estimate"] = point_path

    rows = []
    for item, path in required.items():
        rows.append(
            {
                "item": item,
                "found": path.exists(),
                "path": str(path.relative_to(root)) if path.exists() else str(path),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    manifest = pd.DataFrame(rows)

    decision = ""
    if required["stage13_decision"].exists():
        decision_df = read_csv(required["stage13_decision"])
        if not decision_df.empty and "stage13_decision" in decision_df.columns:
            decision = str(decision_df.iloc[0]["stage13_decision"])

    center = read_csv(required["stage13_centering"]) if required["stage13_centering"].exists() else pd.DataFrame()
    centered = bool(
        not center.empty
        and "centering_status" in center.columns
        and (center["centering_status"] == "BOOTSTRAP_CENTERING_ACCEPTABLE").all()
    )
    invariants = read_csv(required["stage13_ccw_invariants"]) if required["stage13_ccw_invariants"].exists() else pd.DataFrame()
    invariant_pass = bool(not invariants.empty and "pass" in invariants.columns and invariants["pass"].all())

    checks = pd.DataFrame(
        [
            {
                "check": "all_required_files",
                "pass": bool(manifest["found"].all()),
                "detail": f"{int(manifest['found'].sum())}/{len(manifest)} found",
            },
            {
                "check": "stage13_decision_requires_curve_export",
                "pass": decision == "CENTERING_PASSED_EXPORT_CCW_CURVES_BEFORE_FULL_BOOTSTRAP",
                "detail": decision,
            },
            {
                "check": "bootstrap_centering_accepted",
                "pass": centered,
                "detail": "both analyses acceptable" if centered else "review Stage 47",
            },
            {
                "check": "ccw_invariants_passed",
                "pass": invariant_pass,
                "detail": "all Stage 49 checks passed" if invariant_pass else "review Stage 49",
            },
        ]
    )
    status = "STAGE14_PREFLIGHT_PASSED" if bool(checks["pass"].all()) else "STAGE14_PREFLIGHT_FAILED"
    write_csv(manifest, tables / "51_stage14_input_manifest.csv")
    write_csv(checks, tables / "51_stage14_preflight_checks.csv")
    write_text(
        f"""# Stage 14 preflight

**Status:** `{status}`

## Inputs

{markdown_table(manifest)}

## Checks

{markdown_table(checks)}
""",
        tables / "51_stage14_preflight.md",
    )

    print("=" * 116)
    print("STAGE 51 — STAGE 14 PREFLIGHT")
    print("=" * 116)
    print(manifest.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))
    print(f"\nStatus: {status}")
    return 0 if status == "STAGE14_PREFLIGHT_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
