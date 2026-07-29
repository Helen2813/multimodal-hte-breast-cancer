from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modality_hte.config import AnalysisConfig
from modality_hte.data.io import read_table


class DataValidationError(ValueError):
    """Raised when an input table violates a required analysis contract."""


@dataclass
class TableAudit:
    name: str
    path: str
    exists: bool
    rows: int | None = None
    columns: int | None = None
    unique_patients: int | None = None
    duplicate_patient_rows: int | None = None
    missing_patient_ids: int | None = None
    feature_columns: int | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "exists": self.exists,
            "rows": self.rows,
            "columns": self.columns,
            "unique_patients": self.unique_patients,
            "duplicate_patient_rows": self.duplicate_patient_rows,
            "missing_patient_ids": self.missing_patient_ids,
            "feature_columns": self.feature_columns,
            "warnings": self.warnings,
        }


def _binary_values(series: pd.Series) -> set[float | int | bool]:
    non_missing = series.dropna().unique().tolist()
    normalized: set[float | int | bool] = set()
    for value in non_missing:
        if isinstance(value, (np.integer, int, bool)):
            normalized.add(int(value))
        elif isinstance(value, (np.floating, float)) and float(value).is_integer():
            normalized.add(int(value))
        else:
            normalized.add(value)
    return normalized


def _audit_table(
    name: str,
    path: Path,
    patient_id_column: str,
    fail_on_duplicates: bool,
) -> tuple[pd.DataFrame, TableAudit]:
    frame = read_table(path)
    if patient_id_column not in frame.columns:
        raise DataValidationError(
            f"Table '{name}' is missing patient identifier column '{patient_id_column}'."
        )

    missing_ids = int(frame[patient_id_column].isna().sum())
    if missing_ids:
        raise DataValidationError(
            f"Table '{name}' contains {missing_ids} rows with missing patient IDs."
        )

    duplicate_rows = int(frame[patient_id_column].duplicated(keep=False).sum())
    if duplicate_rows and fail_on_duplicates:
        raise DataValidationError(
            f"Table '{name}' contains {duplicate_rows} rows belonging to duplicated patient IDs."
        )

    audit = TableAudit(
        name=name,
        path=str(path),
        exists=True,
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        unique_patients=int(frame[patient_id_column].nunique()),
        duplicate_patient_rows=duplicate_rows,
        missing_patient_ids=missing_ids,
        feature_columns=max(int(frame.shape[1]) - 1, 0),
    )
    return frame, audit


def validate_inputs(config: AnalysisConfig) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Validate configured input tables and return audit, overlap, and missingness reports."""
    patient_id = config.project.patient_id_column
    clinical, clinical_audit = _audit_table(
        name="clinical",
        path=config.clinical.path,
        patient_id_column=patient_id,
        fail_on_duplicates=config.validation.fail_on_duplicate_patient_ids,
    )

    required = {
        patient_id,
        config.clinical.time_column,
        config.clinical.event_column,
        *config.clinical.required_columns,
        *config.clinical.treatment_columns,
    }
    missing_columns = sorted(required.difference(clinical.columns))
    if missing_columns:
        raise DataValidationError(
            "Clinical table is missing required columns: " + ", ".join(missing_columns)
        )

    time_values = pd.to_numeric(clinical[config.clinical.time_column], errors="coerce")
    invalid_time = clinical[config.clinical.time_column].notna() & time_values.isna()
    if invalid_time.any():
        raise DataValidationError(
            f"Survival time column '{config.clinical.time_column}' contains non-numeric values."
        )
    if (time_values.dropna() < 0).any():
        raise DataValidationError("Survival time contains negative values.")

    if config.validation.require_binary_event:
        event_values = _binary_values(clinical[config.clinical.event_column])
        if not event_values.issubset({0, 1}):
            raise DataValidationError(
                f"Event column '{config.clinical.event_column}' must be binary 0/1; "
                f"observed values: {sorted(map(str, event_values))}."
            )

    if config.validation.require_binary_treatment:
        for column in config.clinical.treatment_columns:
            treatment_values = _binary_values(clinical[column])
            if not treatment_values.issubset({0, 1}):
                raise DataValidationError(
                    f"Treatment column '{column}' must be binary 0/1; "
                    f"observed values: {sorted(map(str, treatment_values))}."
                )

    clinical_ids = set(clinical[patient_id].astype(str))
    audits: list[TableAudit] = [clinical_audit]
    overlap_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []

    modality_patient_sets: dict[str, set[str]] = {}
    for modality in config.modalities:
        if not modality.path.exists():
            if config.validation.allow_missing_modalities:
                audits.append(
                    TableAudit(
                        name=modality.name,
                        path=str(modality.path),
                        exists=False,
                        warnings=["Configured modality file is currently absent."],
                    )
                )
                continue
            raise FileNotFoundError(f"Missing modality file: {modality.path}")

        frame, audit = _audit_table(
            name=modality.name,
            path=modality.path,
            patient_id_column=patient_id,
            fail_on_duplicates=config.validation.fail_on_duplicate_patient_ids,
        )
        audits.append(audit)

        ids = set(frame[patient_id].astype(str))
        modality_patient_sets[modality.name] = ids
        overlap = clinical_ids.intersection(ids)
        overlap_rows.append(
            {
                "modality": modality.name,
                "clinical_patients": len(clinical_ids),
                "modality_patients": len(ids),
                "overlap_patients": len(overlap),
                "clinical_coverage": len(overlap) / len(clinical_ids) if clinical_ids else np.nan,
            }
        )

        feature_columns = [column for column in frame.columns if column != patient_id]
        for column in feature_columns:
            missingness_rows.append(
                {
                    "modality": modality.name,
                    "feature": column,
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_fraction": float(frame[column].isna().mean()),
                    "unique_non_missing": int(frame[column].nunique(dropna=True)),
                }
            )

    available_sets = list(modality_patient_sets.values())
    complete_modalities = set.intersection(*available_sets) if available_sets else set()
    complete_all = clinical_ids.intersection(complete_modalities)

    audit_report: dict[str, Any] = {
        "cohort": config.project.cohort,
        "patient_id_column": patient_id,
        "clinical_patients": len(clinical_ids),
        "patients_with_all_available_modalities": len(complete_all),
        "available_modalities": sorted(modality_patient_sets),
        "tables": [audit.as_dict() for audit in audits],
    }
    return (
        audit_report,
        pd.DataFrame(overlap_rows),
        pd.DataFrame(missingness_rows),
    )
