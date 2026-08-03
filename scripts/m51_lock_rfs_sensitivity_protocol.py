from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from _metabric_m7_utils import (
    load_config as load_m7_config,
    project_root,
    sha256,
    write_csv,
)
from _metabric_m8_utils import load_config as load_m8_config


def rel(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def main() -> int:
    root = project_root()
    m7_cfg = load_m7_config(root)
    m8_cfg = load_m8_config(root)
    out = root / "results" / "tables" / "metabric_m11_rfs"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("METABRIC M11.51 - RFS SENSITIVITY PROTOCOL LOCK")
    print("=" * 124)
    print("This stage does not modify the primary overall-survival analyses.")
    print("Settings are loaded from both metabric_m7_config.json and metabric_m8_config.json.")

    clinical_path = root / m7_cfg["files"]["clinical_master"]
    clinical = pd.read_csv(clinical_path, low_memory=False)
    required_columns = {"sample_id", "rfs_months", "rfs_event"}
    missing = sorted(required_columns - set(clinical.columns))
    if missing:
        raise RuntimeError(f"Clinical master is missing RFS columns: {missing}")

    valid = (
        pd.to_numeric(clinical["rfs_months"], errors="coerce").notna()
        & pd.to_numeric(clinical["rfs_event"], errors="coerce").notna()
    )
    rfs_n = int(valid.sum())
    rfs_events = int(
        pd.to_numeric(
            clinical.loc[valid, "rfs_event"],
            errors="coerce",
        ).sum()
    )
    if rfs_n < 1000 or rfs_events < 100:
        raise RuntimeError(
            f"Unexpectedly small RFS endpoint: n={rfs_n}, events={rfs_events}"
        )

    full_settings = m7_cfg["track_b"]
    modality_settings = m8_cfg["modality_analysis"]

    expected = {
        "multimodal_outer_repeats": 20,
        "multimodal_outer_folds": 5,
        "modality_outer_repeats": 10,
        "modality_outer_folds": 5,
    }
    observed = {
        "multimodal_outer_repeats": int(full_settings["outer_repeats"]),
        "multimodal_outer_folds": int(full_settings["outer_folds"]),
        "modality_outer_repeats": int(modality_settings["outer_repeats"]),
        "modality_outer_folds": int(modality_settings["outer_folds"]),
    }
    if observed != expected:
        raise RuntimeError(
            "M11 RFS settings do not match the locked OS designs: "
            f"expected={expected}, observed={observed}"
        )

    source_paths = [
        root / "scripts" / "m52_run_track_b_full_repeated_nested_rfs.py",
        root / "scripts" / "m53_run_modality_specific_repeated_nested_rfs.py",
        clinical_path,
        root / "metabric_m7_config.json",
        root / "metabric_m8_config.json",
    ]
    checks = []
    for path in source_paths:
        checks.append({
            "path": rel(root, path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else "",
            "pass": path.exists(),
        })
    if not all(row["pass"] for row in checks):
        raise RuntimeError("M11 RFS protocol preflight failed.")

    protocol = {
        "protocol_id": "",
        "status": "METABRIC_M11_RFS_SENSITIVITY_LOCKED_BEFORE_RESULTS",
        "primary_endpoint_unchanged": "overall survival",
        "sensitivity_endpoint": {
            "time_column": "rfs_months",
            "event_column": "rfs_event",
            "clinical_master_valid_n": rfs_n,
            "clinical_master_events": rfs_events,
        },
        "multimodal_track_b": {
            "outer_repeats": int(full_settings["outer_repeats"]),
            "outer_folds": int(full_settings["outer_folds"]),
            "repeat_seed_start": int(full_settings["repeat_seed_start"]),
            "engine": full_settings["engine"],
            "iamb_alpha": full_settings["iamb_alpha"],
            "clinical_features": full_settings["clinical_features"],
            "all_supervised_steps_inside_outer_training_fold": True,
        },
        "modality_specific_track_b": {
            "modalities": ["RNA", "CNV", "Methylation", "Mutation"],
            "outer_repeats": int(modality_settings["outer_repeats"]),
            "outer_folds": int(modality_settings["outer_folds"]),
            "repeat_seed_start": int(modality_settings["repeat_seed_start"]),
            "engine": modality_settings["engine"],
            "historical_alpha": modality_settings["historical_alpha"],
            "clinical_features": modality_settings["clinical_features"],
            "all_supervised_steps_inside_outer_training_fold": True,
        },
        "inference": {
            "paired_patient_bootstrap_repetitions": 2000,
            "bootstrap_seed": 81101,
            "conditional_on_locked_repeated_oof_models": True,
            "full_pipeline_bootstrap": False,
            "claim_rule": {
                "positive": "entire 95% interval above zero",
                "negative": "entire 95% interval below zero",
                "otherwise": "no reliable incremental utility",
            },
        },
        "boundaries": [
            "Track A is not rerun for RFS.",
            "The overall-survival analysis remains primary.",
            "No new pathway analysis is performed.",
            "No NPI extension is performed for RFS.",
            "RFS results are retained regardless of direction.",
            "RFS does not replace or redefine the primary OS conclusions.",
        ],
        "input_checks": checks,
    }
    payload = json.dumps(protocol, sort_keys=True)
    protocol["protocol_id"] = (
        "METABRIC_M11_RFS_"
        + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16].upper()
    )

    protocol_path = out / "m51_rfs_sensitivity_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(out / "m51_rfs_input_hash_manifest.csv", checks)

    print(json.dumps(protocol, indent=2))
    print("\nPASS: M11 RFS protocol locked. No model was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
