from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage26_v10_utils import dataframe_console, verify_manifest
from _stage29_v10_event_influence_utils import (
    assemble_frame,
    count_checks,
    run_point_estimate,
)


def project_root() -> Path:
    return Path.cwd().resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(data: object) -> str:
    raw = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_inputs(
    root: Path,
    config: dict,
) -> dict[str, Any]:
    manifest, integrity = verify_manifest(
        root,
        {
            "source": {
                "v10_manifest": config["source"]["v10_manifest"]
            },
            "expected": {
                "protocol_id": config["expected"]["v10_protocol_id"]
            },
        },
    )

    stage26_manifest = load_json(
        root / config["source"]["stage26_calculation_manifest"]
    )
    if (
        stage26_manifest["calculation_id"]
        != config["expected"]["stage26_calculation_id"]
    ):
        raise RuntimeError("Unexpected Stage 26 calculation ID.")

    point = pd.read_csv(
        root / config["source"]["stage26_point_estimate"],
        low_memory=False,
    ).iloc[0]
    observed = float(point["aipw_ato_rmst_difference_days"])
    expected = float(
        config["expected"]["primary_point_estimate_days"]
    )
    tolerance = float(
        config["expected"]["identity_tolerance_days"]
    )
    if abs(observed - expected) > tolerance:
        raise RuntimeError("Stage 26 point estimate mismatch.")

    return {
        "v10_protocol_id": manifest["protocol_id"],
        "stage26_calculation_id": stage26_manifest["calculation_id"],
        "stage26_point_estimate_days": observed,
        "v10_integrity_hash": canonical_sha256(
            integrity[
                ["path", "expected_sha256"]
            ].to_dict("records")
        ),
    }


def load_primary_influence_table(
    root: Path,
    config: dict,
) -> pd.DataFrame:
    scores = pd.read_csv(
        root / config["source"]["stage26_patient_scores"],
        low_memory=False,
    )
    cohort = pd.read_csv(
        root / config["source"]["v10_cohort"],
        low_memory=False,
    )

    required_score_columns = {
        "row_index",
        "patient_id_normalized",
        "influence",
        "absolute_influence",
    }
    missing = required_score_columns - set(scores.columns)
    if missing:
        raise RuntimeError(
            "Stage 26 patient-score columns missing: "
            + ", ".join(sorted(missing))
        )

    scores["patient_id_normalized"] = (
        scores["patient_id_normalized"].astype(str)
    )
    cohort["patient_id_normalized"] = (
        cohort["patient_id_normalized"].astype(str)
    )

    if len(scores) != int(
        config["expected"]["patient_score_rows"]
    ):
        raise RuntimeError(
            f"Expected 271 patient-score rows, found {len(scores)}."
        )
    if scores["patient_id_normalized"].nunique() != len(scores):
        raise RuntimeError("Stage 26 patient-score IDs are not unique.")

    mean_influence = float(
        pd.to_numeric(
            scores["influence"],
            errors="raise",
        ).mean()
    )
    if abs(mean_influence) > 1e-8:
        raise RuntimeError(
            f"Influence values do not have near-zero mean: {mean_influence}"
        )

    merged = scores.merge(
        cohort[
            [
                "patient_id_normalized",
                "analysis_treatment",
                "analysis_event",
                "analysis_time",
            ]
        ],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(scores):
        raise RuntimeError(
            "Stage 26 patient scores did not merge one-to-one with V10."
        )

    for column in (
        "row_index",
        "influence",
        "absolute_influence",
        "analysis_treatment",
        "analysis_event",
        "analysis_time",
    ):
        merged[column] = pd.to_numeric(
            merged[column],
            errors="raise",
        )

    merged["normalized_contribution_days"] = (
        merged["influence"] / len(merged)
    )
    merged["absolute_normalized_contribution_days"] = (
        merged["absolute_influence"] / len(merged)
    )
    merged["arm"] = (
        merged["analysis_treatment"]
        .astype(int)
        .map({0: "control", 1: "early_hormone"})
    )
    merged["event_status"] = (
        merged["analysis_event"]
        .astype(int)
        .map({0: "non_event", 1: "event"})
    )
    return merged


def append_checkpoint(
    row: dict[str, Any],
    path: Path,
) -> None:
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[
            frame["influence_case_id"].astype(str)
            != str(row["influence_case_id"])
        ]
        frame = pd.concat(
            [frame, pd.DataFrame([row])],
            ignore_index=True,
        )
    else:
        frame = pd.DataFrame([row])

    frame = frame.sort_values("influence_rank").reset_index(drop=True)
    write_csv(frame, path)
