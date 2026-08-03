from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage25c_v10_utils import fit_propensity
from _stage26_v10_utils import (
    aggregate_partition_scores,
    assemble_v10_frame,
    dataframe_console,
    fit_partition,
    verify_manifest,
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


def verify_stage28_inputs(
    root: Path,
    config: dict,
) -> dict[str, Any]:
    v10_manifest, v10_integrity = verify_manifest(
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
    observed = float(
        point["aipw_ato_rmst_difference_days"]
    )
    expected = float(
        config["expected"]["primary_point_estimate_days"]
    )
    tolerance = float(
        config["expected"]["reproduction_tolerance_days"]
    )
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"Stage 26 point estimate mismatch: {observed} != {expected}"
        )

    return {
        "v10_protocol_id": v10_manifest["protocol_id"],
        "stage26_calculation_id": stage26_manifest["calculation_id"],
        "stage26_point_estimate_days": observed,
        "v10_integrity_hash": canonical_sha256(
            v10_integrity[
                ["path", "expected_sha256"]
            ].to_dict("records")
        ),
    }


def expected_count_checks(
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
) -> pd.DataFrame:
    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int)
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int)
    expected = config["expected"]

    rows = [
        ("n", len(frame), expected["n"]),
        ("treated", int(treatment.sum()), expected["treated"]),
        ("control", int((1 - treatment).sum()), expected["control"]),
        ("events", int(event.sum()), expected["events"]),
        (
            "treated events",
            int(((treatment == 1) & (event == 1)).sum()),
            expected["treated_events"],
        ),
        (
            "control events",
            int(((treatment == 0) & (event == 1)).sum()),
            expected["control_events"],
        ),
        ("features", len(features), expected["features"]),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "observed": observed,
                "expected": expected_value,
                "pass": observed == expected_value,
            }
            for name, observed, expected_value in rows
        ]
    )


def stage26_like_config(
    base_config: dict,
    g_min: float,
) -> dict[str, Any]:
    return {
        "estimator": {
            "partition_base_seeds": list(
                base_config["estimator"]["partition_base_seeds"]
            ),
            "n_folds": int(base_config["estimator"]["n_folds"]),
            "maximum_partition_retries": int(
                base_config["estimator"][
                    "maximum_partition_retries"
                ]
            ),
            "horizon_days": float(
                base_config["estimator"]["horizon_days"]
            ),
            "interval_days": float(
                base_config["estimator"]["interval_days"]
            ),
            "censoring_g_min": float(g_min),
            "outcome_bound_low": float(
                base_config["estimator"]["outcome_bound_low"]
            ),
            "outcome_bound_high": float(
                base_config["estimator"]["outcome_bound_high"]
            ),
            "score": base_config["estimator"]["score"],
        }
    }
