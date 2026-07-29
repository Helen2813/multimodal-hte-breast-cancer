from pathlib import Path

import pandas as pd
import yaml

from modality_hte.config import load_config
from modality_hte.data.validation import validate_inputs


def test_validation_reports_overlap(tmp_path: Path) -> None:
    clinical = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "time": [100, 200, 300],
            "event": [1, 0, 1],
            "age": [50, 60, 70],
            "chemo": [1, 0, 1],
        }
    )
    rnaseq = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "gene_a": [0.1, 0.2],
            "gene_b": [1.0, None],
        }
    )
    clinical_path = tmp_path / "clinical.csv"
    rnaseq_path = tmp_path / "rnaseq.csv"
    clinical.to_csv(clinical_path, index=False)
    rnaseq.to_csv(rnaseq_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "cohort": "TEST",
                    "patient_id_column": "patient_id",
                    "output_directory": str(tmp_path / "results"),
                },
                "clinical": {
                    "path": str(clinical_path),
                    "time_column": "time",
                    "event_column": "event",
                    "required_columns": ["age"],
                    "treatment_columns": ["chemo"],
                },
                "modalities": {"rnaseq": {"path": str(rnaseq_path)}},
                "validation": {"allow_missing_modalities": False},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    audit, overlap, missingness = validate_inputs(config)

    assert audit["clinical_patients"] == 3
    assert audit["patients_with_all_available_modalities"] == 2
    assert int(overlap.loc[0, "overlap_patients"]) == 2
    assert missingness.loc[missingness["feature"] == "gene_b", "missing_count"].item() == 1
