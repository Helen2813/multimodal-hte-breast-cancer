from __future__ import annotations

import hashlib
import json
import math
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


def verify_inputs(
    root: Path,
    config: dict,
) -> dict[str, Any]:
    v10_manifest, integrity = verify_manifest(
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
        config["expected"]["identity_tolerance_days"]
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
            integrity[
                ["path", "expected_sha256"]
            ].to_dict("records")
        ),
    }


def assemble_frame(
    root: Path,
    config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    frame, features, _ = assemble_v10_frame(
        root,
        {
            "source": {
                "v10_cohort": config["source"]["v10_cohort"],
                "v10_compact": config["source"]["v10_compact"],
            }
        },
    )
    return frame, features


def count_checks(
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
    return pd.DataFrame([
        {
            "check": name,
            "observed": observed,
            "expected": expected_value,
            "pass": observed == expected_value,
        }
        for name, observed, expected_value in rows
    ])


def run_point_estimate(
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
) -> tuple[dict[str, Any], pd.DataFrame]:
    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()
    observed_time = pd.to_numeric(
        frame["analysis_time"],
        errors="coerce",
    ).to_numpy(dtype=float)

    stage25c_config = load_json(
        project_root() / config["source"]["stage25c_config"]
    )
    propensity, _, propensity_fit = fit_propensity(
        frame,
        treatment,
        features,
        stage25c_config,
    )

    partition_rows = []
    patient_rows = []

    for partition_number, base_seed in enumerate(
        config["estimator"]["partition_base_seeds"],
        start=1,
    ):
        row, patient = fit_partition(
            frame,
            features,
            treatment,
            event,
            observed_time,
            propensity,
            partition_number,
            int(base_seed),
            config,
        )
        partition_rows.append(row)
        patient_rows.append(patient)

    partitions = pd.DataFrame(partition_rows)
    patient_scores = pd.concat(
        patient_rows,
        ignore_index=True,
    )
    aggregate, _ = aggregate_partition_scores(
        patient_scores
    )

    effects = pd.to_numeric(
        partitions["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)

    summary = {
        "n": len(frame),
        "treated": int(treatment.sum()),
        "control": int((1 - treatment).sum()),
        "events": int(event.sum()),
        "treated_events": int(
            ((treatment == 1) & (event == 1)).sum()
        ),
        "control_events": int(
            ((treatment == 0) & (event == 1)).sum()
        ),
        "estimate_days": aggregate["estimate_days"],
        "diagnostic_if_se_days": aggregate["if_se_days"],
        "diagnostic_if_ci_low_days": aggregate[
            "if_ci_low_days"
        ],
        "diagnostic_if_ci_high_days": aggregate[
            "if_ci_high_days"
        ],
        "plugin_component_days": aggregate[
            "plugin_component_days"
        ],
        "treated_residual_component_days": aggregate[
            "treated_residual_component_days"
        ],
        "control_residual_component_days": aggregate[
            "control_residual_component_days"
        ],
        "total_residual_augmentation_days": aggregate[
            "total_residual_augmentation_days"
        ],
        "partition_mean_days": float(np.mean(effects)),
        "partition_median_days": float(np.median(effects)),
        "partition_sd_days": float(
            np.std(effects, ddof=1)
        ),
        "partition_mcse_days": float(
            np.std(effects, ddof=1)
            / math.sqrt(len(effects))
        ),
        "partition_min_days": float(np.min(effects)),
        "partition_max_days": float(np.max(effects)),
        "partition_range_days": float(
            np.max(effects) - np.min(effects)
        ),
        "minimum_raw_G": float(
            pd.to_numeric(
                partitions["G_min_raw"],
                errors="raise",
            ).min()
        ),
        "median_pseudo_p99": float(
            pd.to_numeric(
                partitions["pseudo_p99"],
                errors="raise",
            ).median()
        ),
        "maximum_pseudo_max": float(
            pd.to_numeric(
                partitions["pseudo_max"],
                errors="raise",
            ).max()
        ),
        "maximum_nuisance_retry": int(
            pd.to_numeric(
                partitions["nuisance_retry"],
                errors="raise",
            ).max()
        ),
        "propensity_converged": bool(
            propensity_fit["converged"]
        ),
        "maximum_absolute_propensity_coefficient": float(
            propensity_fit["maximum_absolute_coefficient"]
        ),
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(
            np.quantile(propensity, 0.01)
        ),
        "propensity_p99": float(
            np.quantile(propensity, 0.99)
        ),
        "propensity_max": float(np.max(propensity)),
    }
    return summary, partitions


def append_checkpoint(
    row: dict[str, Any],
    path: Path,
) -> None:
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[
            frame["event_case_id"].astype(str)
            != str(row["event_case_id"])
        ]
        frame = pd.concat(
            [frame, pd.DataFrame([row])],
            ignore_index=True,
        )
    else:
        frame = pd.DataFrame([row])

    frame = frame.sort_values(
        ["omitted_arm", "event_case_id"]
    ).reset_index(drop=True)
    write_csv(frame, path)
