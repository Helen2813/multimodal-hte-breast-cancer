from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the analysis configuration is incomplete or malformed."""


@dataclass(frozen=True)
class ProjectConfig:
    cohort: str
    patient_id_column: str
    output_directory: Path


@dataclass(frozen=True)
class ClinicalConfig:
    path: Path
    time_column: str
    event_column: str
    required_columns: tuple[str, ...]
    treatment_columns: tuple[str, ...]


@dataclass(frozen=True)
class ModalityConfig:
    name: str
    path: Path


@dataclass(frozen=True)
class ValidationConfig:
    allow_missing_modalities: bool = True
    fail_on_duplicate_patient_ids: bool = True
    require_binary_treatment: bool = True
    require_binary_event: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    project: ProjectConfig
    clinical: ClinicalConfig
    modalities: tuple[ModalityConfig, ...]
    validation: ValidationConfig


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{section}.{key}'.")
    return mapping[key]


def load_config(path: str | Path) -> AnalysisConfig:
    """Load and minimally validate a YAML analysis configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project_raw = _require(raw, "project", "root")
    clinical_raw = _require(raw, "clinical", "root")
    modalities_raw = _require(raw, "modalities", "root")
    validation_raw = raw.get("validation", {})

    if not isinstance(modalities_raw, dict) or not modalities_raw:
        raise ConfigError("'modalities' must be a non-empty mapping.")

    project = ProjectConfig(
        cohort=str(_require(project_raw, "cohort", "project")),
        patient_id_column=str(_require(project_raw, "patient_id_column", "project")),
        output_directory=Path(_require(project_raw, "output_directory", "project")),
    )
    clinical = ClinicalConfig(
        path=Path(_require(clinical_raw, "path", "clinical")),
        time_column=str(_require(clinical_raw, "time_column", "clinical")),
        event_column=str(_require(clinical_raw, "event_column", "clinical")),
        required_columns=tuple(clinical_raw.get("required_columns", [])),
        treatment_columns=tuple(clinical_raw.get("treatment_columns", [])),
    )
    modalities = tuple(
        ModalityConfig(name=str(name), path=Path(_require(value, "path", f"modalities.{name}")))
        for name, value in modalities_raw.items()
    )
    validation = ValidationConfig(
        allow_missing_modalities=bool(validation_raw.get("allow_missing_modalities", True)),
        fail_on_duplicate_patient_ids=bool(
            validation_raw.get("fail_on_duplicate_patient_ids", True)
        ),
        require_binary_treatment=bool(validation_raw.get("require_binary_treatment", True)),
        require_binary_event=bool(validation_raw.get("require_binary_event", True)),
    )
    return AnalysisConfig(
        project=project,
        clinical=clinical,
        modalities=modalities,
        validation=validation,
    )
