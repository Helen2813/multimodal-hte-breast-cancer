from __future__ import annotations

import json
from pathlib import Path

from modality_hte.config import load_config
from modality_hte.data.validation import validate_inputs


def run(config_path: str | Path) -> dict[str, Path]:
    """Run the input audit and write reproducible reports."""
    config = load_config(config_path)
    output_dir = config.project.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)

    audit, overlap, missingness = validate_inputs(config)

    audit_path = output_dir / "input_validation.json"
    overlap_path = output_dir / "patient_overlap.csv"
    missingness_path = output_dir / "modality_missingness.csv"

    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    overlap.to_csv(overlap_path, index=False)
    missingness.to_csv(missingness_path, index=False)

    return {
        "audit": audit_path,
        "overlap": overlap_path,
        "missingness": missingness_path,
    }
