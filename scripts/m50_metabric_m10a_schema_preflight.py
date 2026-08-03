#!/usr/bin/env python3
"""
METABRIC Stage M10A: read-only schema preflight for the NPI/calibration extension.

Inspects locked M2/M7/M8/M9 outputs without fitting a model, changing previous
results, or printing patient identifiers. The output provides the exact column
schemas needed for the final M10 implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ID_NAMES = {
    "patient_id", "patient", "sample_id", "sample", "case_id", "case",
    "metabric_id", "metabric_patient_id", "id", "study_id",
}
TIME_NAMES = {
    "os_months", "os_time", "overall_survival_months", "time", "duration",
    "survival_time", "event_time",
}
EVENT_NAMES = {"os_event", "os_status", "event", "status", "death", "died"}
NPI_NAMES = {"npi", "nottingham_prognostic_index"}
REPEAT_NAMES = {"repeat", "repeat_id", "outer_repeat", "split_repeat"}
FOLD_NAMES = {"fold", "fold_id", "outer_fold", "split_fold"}
MODALITY_NAMES = {"modality", "analysis_modality", "omics_modality"}

PREDICTION_KEYWORDS = (
    "pred", "risk", "score", "linear_predictor", "prognostic", "surv", "hazard",
)
CLINICAL_KEYWORDS = ("clinical", "clin_")
MODEL_KEYWORDS = ("model", "omics", "multimodal", "combined", "rna", "cnv", "methyl", "mutation")
FIVE_YEAR_KEYWORDS = ("5y", "5yr", "60m", "60_month", "five_year")


def find_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (
            (candidate / "results" / "tables" / "metabric_m2").is_dir()
            and (candidate / "results" / "tables" / "metabric_m7").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root. Run from inside multimodal-hte-breast-cancer."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def first_matching(columns: Sequence[str], names: set[str]) -> Optional[str]:
    lookup = {normalized(column): column for column in columns}
    for name in sorted(names):
        if name in lookup:
            return lookup[name]
    return None


def matching_columns(columns: Sequence[str], keywords: Sequence[str]) -> List[str]:
    return [
        column for column in columns
        if any(keyword in normalized(column) for keyword in keywords)
    ]


def prediction_candidates(columns: Sequence[str]) -> Dict[str, List[str]]:
    candidates = matching_columns(columns, PREDICTION_KEYWORDS)
    candidates = [
        column for column in candidates
        if normalized(column) not in (TIME_NAMES | EVENT_NAMES)
    ]
    clinical = [
        column for column in candidates
        if any(token in normalized(column) for token in CLINICAL_KEYWORDS)
    ]
    model = [
        column for column in candidates
        if any(token in normalized(column) for token in MODEL_KEYWORDS)
        and column not in clinical
    ]
    five_year = [
        column for column in candidates
        if any(token in normalized(column) for token in FIVE_YEAR_KEYWORDS)
    ]
    return {
        "all_prediction_candidates": candidates,
        "clinical_prediction_candidates": clinical,
        "model_prediction_candidates": model,
        "five_year_prediction_candidates": five_year,
    }


def inspect_csv(path: Path, root: Path) -> Tuple[dict, List[dict], List[dict]]:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    columns = [str(column) for column in frame.columns]

    roles: Dict[str, object] = {
        "patient_id": first_matching(columns, ID_NAMES),
        "time": first_matching(columns, TIME_NAMES),
        "event": first_matching(columns, EVENT_NAMES),
        "npi": first_matching(columns, NPI_NAMES),
        "repeat": first_matching(columns, REPEAT_NAMES),
        "fold": first_matching(columns, FOLD_NAMES),
        "modality": first_matching(columns, MODALITY_NAMES),
    }
    roles.update(prediction_candidates(columns))
    rel = str(path.resolve().relative_to(root.resolve()))

    summary = {
        "relative_path": rel,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
        "detected_patient_id": roles["patient_id"],
        "detected_time": roles["time"],
        "detected_event": roles["event"],
        "detected_npi": roles["npi"],
        "detected_repeat": roles["repeat"],
        "detected_fold": roles["fold"],
        "detected_modality": roles["modality"],
        "prediction_candidate_count": len(roles["all_prediction_candidates"]),
        "five_year_prediction_candidate_count": len(roles["five_year_prediction_candidates"]),
    }

    column_rows: List[dict] = []
    for column in columns:
        series = frame[column]
        column_rows.append({
            "relative_path": rel,
            "column": column,
            "dtype": str(series.dtype),
            "nonmissing": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique_nonmissing": int(series.nunique(dropna=True)),
            "identifier_values_suppressed": normalized(column) in ID_NAMES,
        })

    role_rows: List[dict] = []
    for role, value in roles.items():
        if isinstance(value, list):
            for column in value:
                role_rows.append({"relative_path": rel, "role": role, "column": column})
        else:
            role_rows.append({"relative_path": rel, "role": role, "column": value or ""})

    print("-" * 124)
    print(rel)
    print(f"rows={frame.shape[0]} columns={frame.shape[1]}")
    print("columns:")
    for column in columns:
        print(f"  {column}")
    print("detected roles:")
    for role, value in roles.items():
        print(f"  {role}: {value}")

    return summary, column_rows, role_rows


def existing(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path.is_file()]


def find_candidate_files(root: Path) -> List[Path]:
    tables = root / "results" / "tables"
    exact_candidates = [
        tables / "metabric_m2" / "m06_metabric_clinical_master_LOCAL_ONLY.csv",
        tables / "metabric_m7" / "m37_track_a_full_results.csv",
        tables / "metabric_m7" / "m37_track_a_paired_delta_summary.csv",
        tables / "metabric_m7" / "m38_repeat_level_oof_results.csv",
        tables / "metabric_m7" / "m38_oof_predictions_LOCAL_ONLY.csv",
        tables / "metabric_m8" / "m41_oof_predictions_LOCAL_ONLY.csv",
        tables / "metabric_m8" / "m41_selected_features_checkpoint.csv",
        tables / "metabric_m8" / "m41_candidate_features_checkpoint.csv",
        tables / "metabric_m8" / "m41_modality_feature_universe.csv",
    ]

    discovered: List[Path] = []
    for folder_name in ("metabric_m7", "metabric_m8", "metabric_m9"):
        folder = tables / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            name = path.name.lower()
            if any(token in name for token in (
                "oof", "prediction", "track_a", "bootstrap", "stability",
                "calibration", "brier", "paired_delta", "summary",
            )):
                discovered.append(path)

    unique: List[Path] = []
    seen = set()
    for path in existing(exact_candidates) + discovered:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def has_patient_level_predictions(file_summary: dict) -> bool:
    return bool(
        file_summary.get("detected_patient_id")
        and file_summary.get("prediction_candidate_count", 0) >= 1
        and file_summary.get("rows", 0) > 0
    )


def main() -> int:
    root = find_root(Path.cwd())
    out_dir = root / "results" / "tables" / "metabric_m10a"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("METABRIC STAGE M10A - READ-ONLY NPI AND CALIBRATION SCHEMA PREFLIGHT")
    print("=" * 124)
    print(f"Project root: {root}")
    print("M1-M9 are NOT rerun.")
    print("No model is fitted.")
    print("No feature selection is repeated.")
    print("No patient identifier values are printed.")
    print("=" * 124)

    files = find_candidate_files(root)
    if not files:
        raise FileNotFoundError("No M2/M7/M8/M9 candidate files were found.")

    inventory_rows: List[dict] = []
    column_rows: List[dict] = []
    role_rows: List[dict] = []
    for path in files:
        summary, columns, roles = inspect_csv(path, root)
        inventory_rows.append(summary)
        column_rows.extend(columns)
        role_rows.extend(roles)

    inventory = pd.DataFrame(inventory_rows)
    columns = pd.DataFrame(column_rows)
    roles = pd.DataFrame(role_rows)

    inventory_path = out_dir / "m50_file_inventory.csv"
    columns_path = out_dir / "m50_column_inventory.csv"
    roles_path = out_dir / "m50_detected_roles.csv"
    inventory.to_csv(inventory_path, index=False)
    columns.to_csv(columns_path, index=False)
    roles.to_csv(roles_path, index=False)

    def row_for(name_fragment: str) -> Optional[dict]:
        matches = [
            row for row in inventory_rows
            if name_fragment.lower() in Path(row["relative_path"]).name.lower()
        ]
        return matches[0] if matches else None

    clinical = row_for("m06_metabric_clinical_master")
    track_b_combined = row_for("m38_oof_predictions")
    modality_oof = row_for("m41_oof_predictions")
    track_a_patient_files = [
        row for row in inventory_rows
        if "track_a" in Path(row["relative_path"]).name.lower()
        and has_patient_level_predictions(row)
    ]

    checks = {
        "clinical_master_found": clinical is not None,
        "clinical_master_has_patient_id": bool(clinical and clinical.get("detected_patient_id")),
        "clinical_master_has_npi": bool(clinical and clinical.get("detected_npi")),
        "clinical_master_has_time": bool(clinical and clinical.get("detected_time")),
        "clinical_master_has_event": bool(clinical and clinical.get("detected_event")),
        "track_b_combined_oof_found": track_b_combined is not None,
        "track_b_combined_oof_has_patient_id": bool(track_b_combined and track_b_combined.get("detected_patient_id")),
        "track_b_combined_oof_has_repeat": bool(track_b_combined and track_b_combined.get("detected_repeat")),
        "track_b_combined_oof_has_predictions": bool(track_b_combined and track_b_combined.get("prediction_candidate_count", 0) >= 2),
        "modality_oof_found": modality_oof is not None,
        "modality_oof_has_patient_id": bool(modality_oof and modality_oof.get("detected_patient_id")),
        "modality_oof_has_repeat": bool(modality_oof and modality_oof.get("detected_repeat")),
        "modality_oof_has_modality": bool(modality_oof and modality_oof.get("detected_modality")),
        "modality_oof_has_predictions": bool(modality_oof and modality_oof.get("prediction_candidate_count", 0) >= 2),
        "track_a_patient_level_predictions_found": bool(track_a_patient_files),
    }

    core_keys = (
        "clinical_master_found", "clinical_master_has_patient_id", "clinical_master_has_npi",
        "clinical_master_has_time", "clinical_master_has_event",
        "track_b_combined_oof_found", "track_b_combined_oof_has_patient_id",
        "track_b_combined_oof_has_repeat", "track_b_combined_oof_has_predictions",
        "modality_oof_found", "modality_oof_has_patient_id", "modality_oof_has_repeat",
        "modality_oof_has_modality", "modality_oof_has_predictions",
    )
    core_ready = all(checks[key] for key in core_keys)

    if core_ready and checks["track_a_patient_level_predictions_found"]:
        status = "M10A_READY_FOR_EXACT_NPI_AND_CALIBRATION_IMPLEMENTATION"
    elif core_ready:
        status = "M10A_READY_WITH_TRACK_A_PATIENT_PREDICTION_LIMITATION"
    else:
        status = "M10A_REQUIRES_SCHEMA_REVIEW"

    readiness_path = out_dir / "m50_m10a_readiness.json"
    readiness = {
        "stage": "METABRIC_M10A",
        "status": status,
        "purpose": "Read-only schema audit before the locked NPI and calibration extension.",
        "checks": checks,
        "track_a_patient_prediction_candidates": [row["relative_path"] for row in track_a_patient_files],
        "boundaries": [
            "M1-M9 are not rerun.",
            "No feature selection is repeated.",
            "No prognostic model is fitted.",
            "No manuscript prose is generated.",
            "Patient identifier values are never printed.",
            "Track A NPI will be marked not estimable if locked patient-level predictions are unavailable.",
        ],
        "generated_files": [
            str(inventory_path.relative_to(root)),
            str(columns_path.relative_to(root)),
            str(roles_path.relative_to(root)),
            str(readiness_path.relative_to(root)),
        ],
    }
    readiness_path.write_text(json.dumps(readiness, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("=" * 124)
    print("M10A READINESS")
    print("=" * 124)
    for key, value in checks.items():
        print(f"{key:<55} {value}")
    print("-" * 124)
    print(f"STATUS: {status}")
    print("Generated:")
    for path in readiness["generated_files"]:
        print(f"  {path}")
    print("=" * 124)
    return 0 if core_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
