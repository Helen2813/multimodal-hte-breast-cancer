#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from _stage12_utils import crossfit_propensity
from _stage18_utils import make_grouped_bootstrap_folds
from _stage16_utils import project_root


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    include_index: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=include_index,
        encoding="utf-8-sig",
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
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


def dataframe_console(
    frame: pd.DataFrame,
    max_rows: int | None = None,
) -> str:
    if frame.empty:
        return "<empty table>"

    view = frame if max_rows is None else frame.head(max_rows)
    with pd.option_context(
        "display.max_rows",
        None if max_rows is None else max_rows,
        "display.max_columns",
        None,
        "display.width",
        360,
        "display.max_colwidth",
        120,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        return view.to_string(index=False)


def ensure_dirs(root: Path, config: dict) -> None:
    for key in ("table_dir", "derived_dir", "protocol_dir"):
        (root / config["output"][key]).mkdir(
            parents=True,
            exist_ok=True,
        )
    (root / "data/derived/manifests").mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "results/logs").mkdir(
        parents=True,
        exist_ok=True,
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return load_json(root / "stage25_v10_config.json")


def normalize_tcga_id(value: object) -> str:
    text = str(value).strip().upper().replace("_", "-")
    match = re.search(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
        text,
    )
    return match.group(1) if match else text


def compact_features(frame: pd.DataFrame) -> list[str]:
    features = [
        column
        for column in frame.columns
        if str(column).startswith("W_")
    ]
    for column in ("diagnosis_year", "diagnosis_year_missing"):
        if column in frame.columns:
            features.append(column)
    return list(dict.fromkeys(features))


def verify_v9_lock(root: Path, config: dict) -> pd.DataFrame:
    manifest_path = root / config["source"]["v9_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = load_json(manifest_path)
    expected_protocol = config["expected_v9"]["protocol_id"]
    if manifest.get("protocol_id") != expected_protocol:
        raise RuntimeError(
            "Unexpected Candidate V9 protocol ID: "
            f"{manifest.get('protocol_id')} != {expected_protocol}"
        )

    rows = []
    for item in manifest["locked_files"]:
        path = root / item["path"]
        observed = sha256_file(path) if path.exists() else None
        rows.append({
            "path": item["path"],
            "exists": path.exists(),
            "expected_sha256": item["sha256"],
            "observed_sha256": observed,
            "match": observed == item["sha256"],
        })

    result = pd.DataFrame(rows)
    if result.empty or not bool(result["match"].all()):
        raise RuntimeError(
            "Candidate V9 integrity failed.\n"
            + dataframe_console(
                result[result["match"] != True],
                max_rows=200,
            )
        )
    return result


def find_clinical_columns(
    clinical: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    columns = list(clinical.columns)

    patient_candidates = [
        column
        for column in columns
        if str(column).lower() in {
            "cases.submitter_id",
            "case_submitter_id",
            "patient_id",
        }
    ]
    if not patient_candidates:
        patient_candidates = [
            column
            for column in columns
            if "submitter_id" in str(column).lower()
            and "case" in str(column).lower()
        ]
    if not patient_candidates:
        raise RuntimeError(
            "Patient ID column not found in clinical.tsv."
        )

    start_candidates = [
        column
        for column in columns
        if "days_to_treatment_start" in str(column).lower()
    ]
    if not start_candidates:
        raise RuntimeError(
            "Treatment start-day column not found in clinical.tsv."
        )

    end_candidates = [
        column
        for column in columns
        if "days_to_treatment_end" in str(column).lower()
    ]

    text_columns = [
        column
        for column in columns
        if any(
            token in str(column).lower()
            for token in config[
                "treatment_classification"
            ]["text_column_tokens"]
        )
    ]
    if not text_columns:
        raise RuntimeError(
            "No treatment classification text columns found."
        )

    return {
        "patient_id": patient_candidates[0],
        "start_day": start_candidates[0],
        "end_day": end_candidates[0] if end_candidates else None,
        "text_columns": text_columns,
    }


def classify_treatment(
    text: object,
    config: dict,
) -> str:
    value = str(text).strip().lower()
    if not value or value in {"nan", "none"}:
        return "other_or_unknown"

    for pattern in config[
        "treatment_classification"
    ]["hormone_patterns"]:
        if re.search(pattern, value):
            return "hormone"

    for pattern in config[
        "treatment_classification"
    ]["chemotherapy_patterns"]:
        if re.search(pattern, value):
            return "chemo"

    if re.search(
        r"\btarget|trastuz|herceptin|pertuz|lapatin|bevaciz",
        value,
    ):
        return "targeted"

    if re.search(r"radiat|radiotherap", value):
        return "radiation"

    return "other_or_unknown"


def build_treatment_registry(
    clinical: pd.DataFrame,
    v9_patient_ids: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    columns = find_clinical_columns(clinical, config)
    timing = config["timing"]
    landmark = float(
        config["candidate_v10_population"]["landmark_day"]
    )

    work = clinical.copy()
    work["patient_id_normalized"] = work[
        columns["patient_id"]
    ].map(normalize_tcga_id)

    text_frame = (
        work[columns["text_columns"]]
        .fillna("")
        .astype(str)
    )
    work["__treatment_text"] = text_frame.apply(
        lambda row: " | ".join(
            value
            for value in row
            if value and value.lower() != "nan"
        ),
        axis=1,
    )
    work["__family"] = work["__treatment_text"].map(
        lambda text: classify_treatment(text, config)
    )

    raw_start = pd.to_numeric(
        work[columns["start_day"]],
        errors="coerce",
    )
    minimum = float(timing["minimum_valid_day"])
    maximum = float(timing["maximum_valid_day"])
    valid_start = (
        raw_start.notna()
        & raw_start.between(minimum, maximum, inclusive="both")
    )
    work["__raw_start"] = raw_start
    work["__valid_start"] = valid_start
    work["__start_day"] = raw_start.where(valid_start)

    if columns["end_day"] is not None:
        work["__end_day"] = pd.to_numeric(
            work[columns["end_day"]],
            errors="coerce",
        )
    else:
        work["__end_day"] = np.nan

    work = work[
        work["patient_id_normalized"].isin(
            set(v9_patient_ids.astype(str))
        )
    ].copy()

    rows = []
    for patient_id in v9_patient_ids.astype(str):
        group = work[
            work["patient_id_normalized"] == patient_id
        ]
        chemo = group[group["__family"] == "chemo"]
        hormone = group[group["__family"] == "hormone"]

        chemo_records = len(chemo)
        valid_chemo = chemo[chemo["__valid_start"]]
        invalid_chemo = chemo[~chemo["__valid_start"]]
        valid_chemo_starts = pd.to_numeric(
            valid_chemo["__start_day"],
            errors="coerce",
        ).dropna()

        valid_hormone = hormone[hormone["__valid_start"]]
        hormone_starts = pd.to_numeric(
            valid_hormone["__start_day"],
            errors="coerce",
        ).dropna()

        earliest_chemo = (
            float(valid_chemo_starts.min())
            if len(valid_chemo_starts)
            else np.nan
        )
        earliest_hormone = (
            float(hormone_starts.min())
            if len(hormone_starts)
            else np.nan
        )

        any_chemo_by_landmark = bool(
            len(valid_chemo_starts)
            and (valid_chemo_starts <= landmark).any()
        )
        any_chemo_after_landmark = bool(
            len(valid_chemo_starts)
            and (valid_chemo_starts > landmark).any()
        )
        all_chemo_start_timing_ascertainable = bool(
            chemo_records == 0
            or (
                len(invalid_chemo) == 0
                and len(valid_chemo_starts) == chemo_records
            )
        )
        broad_no_chemo_by_landmark = not any_chemo_by_landmark
        v10_eligible = bool(
            broad_no_chemo_by_landmark
            and all_chemo_start_timing_ascertainable
        )

        if chemo_records == 0:
            chemo_status = "no_documented_chemo_records"
        elif not all_chemo_start_timing_ascertainable:
            chemo_status = "chemo_start_timing_unascertainable"
        elif any_chemo_by_landmark:
            chemo_status = "chemo_initiated_by_day180"
        elif any_chemo_after_landmark:
            chemo_status = "chemo_initiated_after_day180_only"
        else:
            chemo_status = "chemo_status_unclassified"

        rows.append({
            "patient_id_normalized": patient_id,
            "chemo_records": chemo_records,
            "chemo_records_valid_start": len(valid_chemo_starts),
            "chemo_records_invalid_start": len(invalid_chemo),
            "any_documented_chemo": int(chemo_records > 0),
            "any_chemo_start_by_day180": int(
                any_chemo_by_landmark
            ),
            "any_chemo_start_after_day180": int(
                any_chemo_after_landmark
            ),
            "all_chemo_start_timing_ascertainable": int(
                all_chemo_start_timing_ascertainable
            ),
            "broad_no_chemo_start_by_day180": int(
                broad_no_chemo_by_landmark
            ),
            "candidate_v10_eligible": int(v10_eligible),
            "earliest_chemo_start_day": earliest_chemo,
            "hormone_records": len(hormone),
            "hormone_records_valid_start": len(hormone_starts),
            "earliest_hormone_start_day": earliest_hormone,
            "chemo_timing_status": chemo_status,
        })

    registry = pd.DataFrame(rows)
    classified_text_audit = (
        work[work["__family"].isin(["chemo", "hormone"])]
        .groupby(["__family", "__treatment_text"], dropna=False)
        .agg(
            records=("patient_id_normalized", "size"),
            patients=("patient_id_normalized", "nunique"),
            valid_start_records=("__valid_start", "sum"),
        )
        .reset_index()
        .rename(columns={
            "__family": "treatment_family",
            "__treatment_text": "raw_combined_treatment_text",
        })
        .sort_values(
            ["treatment_family", "records", "raw_combined_treatment_text"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )

    source_metadata = {
        "patient_id_column": columns["patient_id"],
        "treatment_start_column": columns["start_day"],
        "treatment_end_column": columns["end_day"],
        "treatment_text_columns": columns["text_columns"],
        "clinical_rows": len(clinical),
        "v9_treatment_rows": len(work),
    }
    return registry, source_metadata, classified_text_audit


def weighted_mean_variance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0 or float(weights.sum()) <= 0:
        return float("nan"), float("nan")

    normalized = weights / weights.sum()
    mean = float(np.sum(normalized * values))
    variance = float(
        np.sum(normalized * (values - mean) ** 2)
    )
    return mean, variance


def smd_from_moments(
    mean_1: float,
    variance_1: float,
    mean_0: float,
    variance_0: float,
) -> float:
    pooled = (variance_1 + variance_0) / 2.0
    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0 if mean_1 == mean_0 else float("inf")
    return (mean_1 - mean_0) / math.sqrt(pooled)


def balance_table(
    frame: pd.DataFrame,
    features: list[str],
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> pd.DataFrame:
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)

    rows = []
    for feature in features:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        if not np.isfinite(median):
            median = 0.0
        imputed = np.where(np.isfinite(values), values, median)

        treated = treatment == 1
        control = treatment == 0

        mean_1 = float(np.mean(imputed[treated]))
        mean_0 = float(np.mean(imputed[control]))
        var_1 = float(np.var(imputed[treated]))
        var_0 = float(np.var(imputed[control]))
        unweighted_smd = smd_from_moments(
            mean_1,
            var_1,
            mean_0,
            var_0,
        )

        weight_1 = np.where(treated, 1.0 - propensity, 0.0)
        weight_0 = np.where(control, propensity, 0.0)
        weighted_mean_1, weighted_var_1 = weighted_mean_variance(
            imputed[treated],
            weight_1[treated],
        )
        weighted_mean_0, weighted_var_0 = weighted_mean_variance(
            imputed[control],
            weight_0[control],
        )
        weighted_smd = smd_from_moments(
            weighted_mean_1,
            weighted_var_1,
            weighted_mean_0,
            weighted_var_0,
        )

        rows.append({
            "feature": feature,
            "missing_fraction": float(
                np.mean(~np.isfinite(values))
            ),
            "unweighted_mean_treated": mean_1,
            "unweighted_mean_control": mean_0,
            "unweighted_smd": unweighted_smd,
            "ato_weighted_mean_treated": weighted_mean_1,
            "ato_weighted_mean_control": weighted_mean_0,
            "ato_weighted_smd": weighted_smd,
            "abs_ato_weighted_smd": abs(weighted_smd),
        })

    return pd.DataFrame(rows).sort_values(
        ["abs_ato_weighted_smd", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def propensity_diagnostics(
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    estimator = config["estimator"]
    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()
    groups = np.arange(len(frame), dtype=int)

    partition_rows = []
    predictions = []
    for partition_number, base_seed in enumerate(
        estimator["partition_base_seeds"],
        start=1,
    ):
        fold, stratification, fold_retry = (
            make_grouped_bootstrap_folds(
                treatment,
                event,
                groups,
                int(base_seed),
                int(estimator["n_folds"]),
            )
        )
        propensity = crossfit_propensity(
            frame,
            features,
            fold,
            "analysis_treatment",
            int(base_seed) + 10,
        )
        predictions.append(propensity)
        partition_rows.append({
            "partition": partition_number,
            "base_seed": int(base_seed),
            "fold_stratification": stratification,
            "fold_retry": int(fold_retry),
            "propensity_min": float(np.min(propensity)),
            "propensity_p01": float(
                np.quantile(propensity, 0.01)
            ),
            "propensity_p05": float(
                np.quantile(propensity, 0.05)
            ),
            "propensity_median": float(
                np.median(propensity)
            ),
            "propensity_p95": float(
                np.quantile(propensity, 0.95)
            ),
            "propensity_p99": float(
                np.quantile(propensity, 0.99)
            ),
            "propensity_max": float(np.max(propensity)),
        })

    matrix = np.vstack(predictions)
    aggregated = np.mean(matrix, axis=0)

    patient = pd.DataFrame({
        "patient_id_normalized": frame[
            "patient_id_normalized"
        ].astype(str),
        "analysis_treatment": treatment,
        "analysis_event": event,
        "propensity_repeated_mean": aggregated,
        "propensity_repeated_sd": np.std(
            matrix,
            axis=0,
            ddof=1,
        ),
    })

    treated = treatment == 1
    control = treatment == 0
    weight_1 = np.where(treated, 1.0 - aggregated, 0.0)
    weight_0 = np.where(control, aggregated, 0.0)

    ess_1 = float(
        weight_1[treated].sum() ** 2
        / np.sum(weight_1[treated] ** 2)
    )
    ess_0 = float(
        weight_0[control].sum() ** 2
        / np.sum(weight_0[control] ** 2)
    )

    summary = {
        "propensity_min": float(np.min(aggregated)),
        "propensity_p01": float(np.quantile(aggregated, 0.01)),
        "propensity_p05": float(np.quantile(aggregated, 0.05)),
        "propensity_median": float(np.median(aggregated)),
        "propensity_p95": float(np.quantile(aggregated, 0.95)),
        "propensity_p99": float(np.quantile(aggregated, 0.99)),
        "propensity_max": float(np.max(aggregated)),
        "fraction_propensity_below_0_05": float(
            np.mean(aggregated < 0.05)
        ),
        "fraction_propensity_above_0_95": float(
            np.mean(aggregated > 0.95)
        ),
        "ato_ess_treated": ess_1,
        "ato_ess_control": ess_0,
        "ato_ess_fraction_treated": float(
            ess_1 / int(treated.sum())
        ),
        "ato_ess_fraction_control": float(
            ess_0 / int(control.sum())
        ),
    }

    return (
        pd.DataFrame(partition_rows),
        patient,
        summary,
    )


def refuse_existing_v10_lock(root: Path, config: dict) -> None:
    paths = [
        root / config["output"]["protocol"],
        root / config["output"]["analysis_plan"],
        root / config["output"]["manifest"],
        root / config["output"]["sentinel"],
    ]
    existing = [
        str(path.relative_to(root))
        for path in paths
        if path.exists()
    ]
    if existing:
        raise RuntimeError(
            "Candidate V10 lock artifacts already exist. "
            "This runner refuses to overwrite them:\n"
            + "\n".join(existing)
        )
