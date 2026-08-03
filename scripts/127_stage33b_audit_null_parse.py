from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from _stage33b_summary_repair_utils import (
    canonical_sha256,
    expected_scenario_ids,
    load_json,
    project_root,
    restore_effect_regime_from_scenario_id,
    sha256_file,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    repair_config = load_json(
        root / "stage33b_null_summary_repair_config.json"
    )
    source = repair_config["source"]
    output = repair_config["output"]
    expected = repair_config["expected"]

    print("=" * 128)
    print("STAGE 127 - AUDIT STAGE 33 NULL-TOKEN PARSING")
    print("=" * 128)

    stage33_config = load_json(root / source["stage33_config"])
    stage33_manifest = load_json(root / source["stage33_manifest"])
    if stage33_manifest["simulation_id"] != expected["simulation_id"]:
        raise RuntimeError("Unexpected Stage 33 simulation ID.")

    checkpoint_path = root / source["checkpoint"]
    default_read = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )
    safe_read_raw = pd.read_csv(
        checkpoint_path,
        low_memory=False,
        keep_default_na=False,
    )
    safe_read, regime_repair_audit = (
        restore_effect_regime_from_scenario_id(
            safe_read_raw
        )
    )

    default_missing_effect = int(
        default_read["effect_regime"].isna().sum()
    )
    raw_empty_effect = int(
        safe_read_raw["effect_regime"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    safe_null_rows = int(
        (safe_read["effect_regime"] == "null").sum()
    )
    safe_benefit_rows = int(
        (
            safe_read["effect_regime"]
            == "empirically_calibrated_benefit"
        ).sum()
    )

    key_columns = [
        "scenario_id",
        "repetition",
        "method",
    ]
    duplicate_rows = int(
        safe_read.duplicated(key_columns).sum()
    )
    observed_scenarios = set(
        safe_read["scenario_id"].astype(str)
    )
    expected_scenarios = expected_scenario_ids(repair_config)

    original_summary = pd.read_csv(
        root / source["original_scenario_summary"],
        low_memory=False,
        keep_default_na=False,
    )

    audit = pd.DataFrame([
        {
            "check": "checkpoint rows",
            "observed": len(safe_read),
            "expected": expected["checkpoint_row_count"],
            "pass": len(safe_read)
            == expected["checkpoint_row_count"],
        },
        {
            "check": "default parser missing effect_regime rows",
            "observed": default_missing_effect,
            "expected": expected["null_checkpoint_rows"],
            "pass": default_missing_effect
            == expected["null_checkpoint_rows"],
        },
        {
            "check": "raw empty effect_regime cells",
            "observed": raw_empty_effect,
            "expected": expected["null_checkpoint_rows"],
            "pass": raw_empty_effect
            == expected["null_checkpoint_rows"],
        },
        {
            "check": "reconstructed null rows",
            "observed": safe_null_rows,
            "expected": expected["null_checkpoint_rows"],
            "pass": safe_null_rows
            == expected["null_checkpoint_rows"],
        },
        {
            "check": "safe parser benefit rows",
            "observed": safe_benefit_rows,
            "expected": expected["benefit_checkpoint_rows"],
            "pass": safe_benefit_rows
            == expected["benefit_checkpoint_rows"],
        },
        {
            "check": "duplicate checkpoint keys",
            "observed": duplicate_rows,
            "expected": 0,
            "pass": duplicate_rows == 0,
        },
        {
            "check": "scenario inventory count",
            "observed": len(observed_scenarios),
            "expected": expected["scenario_count"],
            "pass": observed_scenarios == expected_scenarios,
        },
        {
            "check": "original summary rows",
            "observed": len(original_summary),
            "expected": expected[
                "original_summary_rows_expected_from_bug"
            ],
            "pass": len(original_summary)
            == expected[
                "original_summary_rows_expected_from_bug"
            ],
        },
    ])
    write_csv(audit, root / output["parse_audit"])
    repair_detail_path = (
        root
        / output["table_dir"]
        / "s33b_127_effect_regime_reconstruction_LOCAL_ONLY.csv"
    )
    write_csv(regime_repair_audit, repair_detail_path)

    inventory = (
        safe_read.groupby(
            [
                "scenario_id",
                "effect_regime",
                "method",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="checkpoint_rows")
        .sort_values(
            ["effect_regime", "scenario_id", "method"]
        )
        .reset_index(drop=True)
    )
    inventory["expected_rows"] = expected[
        "repetitions_per_scenario"
    ]
    inventory["pass"] = (
        inventory["checkpoint_rows"]
        == inventory["expected_rows"]
    )
    write_csv(inventory, root / output["scenario_inventory"])

    if not bool(audit["pass"].all()):
        raise RuntimeError(
            "Stage 33 parsing audit failed.\n"
            + audit.to_string(index=False)
        )
    if not bool(inventory["pass"].all()):
        raise RuntimeError(
            "Stage 33 scenario inventory is incomplete."
        )

    locked_inputs = [
        root / "stage33b_null_summary_repair_config.json",
        root / "scripts/_stage33b_summary_repair_utils.py",
        root / "scripts/127_stage33b_audit_null_parse.py",
        root / "scripts/128_stage33b_repair_summary.py",
        root / "run_stage33b_null_summary_repair.ps1",
        root / source["stage33_config"],
        root / source["stage33_manifest"],
        root / source["checkpoint"],
        root / source["original_scenario_summary"],
        root / source["original_design_gates"],
        root / source["original_final_json"],
    ]
    locked_files = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in locked_inputs
    ]

    manifest = {
        "status": "STAGE33B_NULL_SUMMARY_REPAIR_AUDITED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage33_simulation_id": stage33_manifest["simulation_id"],
        "repair_reason": repair_config["repair_reason"],
        "boundary": repair_config["boundary"],
        "parse_audit": audit.to_dict("records"),
        "scenario_inventory_sha256": canonical_sha256(
            inventory.to_dict("records")
        ),
        "locked_files": locked_files,
    }
    manifest["repair_id"] = (
        "PAPER_A_STAGE33B_NULL_REPAIR_"
        + canonical_sha256(manifest)[:16].upper()
    )
    write_json(manifest, root / output["audit_manifest"])

    print("Parse audit")
    print(audit.to_string(index=False))
    print("\nScenario inventory")
    print(inventory.to_string(index=False))
    print(
        "\nPASS: the missing null summaries are fully explained "
        "by default CSV NA parsing. No simulation rerun is needed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
