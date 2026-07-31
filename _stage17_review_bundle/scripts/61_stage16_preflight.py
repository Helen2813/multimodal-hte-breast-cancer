#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from _stage16_utils import (
    ensure_dirs,
    exact_landmark_payload,
    load_config,
    markdown_table,
    project_root,
    read_csv,
    sha256_file,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    design = cfg["design"]
    tables = root / "results/tables"

    required = {
        "stage15_decision": tables / "59_stage15_decision.csv",
        "stage15_bridge": tables / "57_common_target_estimator_bridge.csv",
        "stage15_bridge_diagnostics": tables / "57_bridge_diagnostics.csv",
        "stage12_utils": root / "scripts/_stage12_utils.py",
        "stage41_script": root / "scripts/41_replicate_estimators.py",
        "landmark_cohort_sample_source": root / "data/derived/stage14_trace/53_candidate_06.csv",
    }
    manifest = pd.DataFrame(
        [
            {
                "item": item,
                "found": path.exists(),
                "path": str(path.relative_to(root)) if path.exists() else str(path),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
            for item, path in required.items()
        ]
    )

    payload = exact_landmark_payload()
    metadata = payload["metadata"]
    checks = pd.DataFrame(
        [
            {
                "check": "all_required_inputs",
                "observed": int(manifest["found"].sum()),
                "expected": len(manifest),
                "pass": bool(manifest["found"].all()),
            },
            {
                "check": "landmark_n",
                "observed": len(payload["frame"]),
                "expected": design["expected_n"],
                "pass": len(payload["frame"]) == design["expected_n"],
            },
            {
                "check": "treated",
                "observed": int(payload["a"].sum()),
                "expected": design["expected_treated"],
                "pass": int(payload["a"].sum()) == design["expected_treated"],
            },
            {
                "check": "controls",
                "observed": int((1 - payload["a"]).sum()),
                "expected": design["expected_control"],
                "pass": int((1 - payload["a"]).sum()) == design["expected_control"],
            },
            {
                "check": "events",
                "observed": int(payload["event"].sum()),
                "expected": design["expected_events"],
                "pass": int(payload["event"].sum()) == design["expected_events"],
            },
            {
                "check": "compact_features",
                "observed": len(payload["features"]),
                "expected": design["expected_features"],
                "pass": len(payload["features"]) == design["expected_features"],
            },
            {
                "check": "exact_aipw_replication",
                "observed": payload["theta"],
                "expected": design["expected_landmark_aipw_days"],
                "pass": abs(
                    payload["theta"] - design["expected_landmark_aipw_days"]
                )
                <= design["replication_tolerance_days"],
            },
        ]
    )
    status = (
        "STAGE16_PREFLIGHT_PASSED"
        if bool(checks["pass"].all())
        else "STAGE16_PREFLIGHT_FAILED"
    )
    write_csv(manifest, tables / "61_stage16_input_manifest.csv")
    write_csv(checks, tables / "61_stage16_preflight_checks.csv")
    write_text(
        f"""# Stage 16 preflight

**Status:** `{status}`

## Inputs

{markdown_table(manifest)}

## Exact reconstruction checks

{markdown_table(checks)}
""",
        tables / "61_stage16_preflight.md",
    )

    print("=" * 118)
    print("STAGE 61 — STAGE 16 PREFLIGHT")
    print("=" * 118)
    print(manifest.to_string(index=False))
    print("\nExact reconstruction checks")
    print(checks.to_string(index=False))
    print(f"\nStatus: {status}")
    return 0 if status == "STAGE16_PREFLIGHT_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
