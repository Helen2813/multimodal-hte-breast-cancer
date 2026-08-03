#!/usr/bin/env python3
"""
METABRIC Stage M10B
===================

Secondary NPI benchmark and publication-figure extension using only locked
out-of-fold predictions from Stages M7-M9.

This stage:
- does not rerun Track A or Track B model development;
- does not repeat feature selection;
- does not fit an NPI+omics model;
- does not generate manuscript prose;
- performs a paired patient bootstrap of locked repeated OOF predictions;
- reports calibration as not estimable when patient-level survival
  probabilities were not saved;
- builds numerical publication figures from locked M7-M9 summaries.

Primary Track A and Track B claims remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BOOTSTRAP_SEED = 71001
PROTOCOL_ID = "METABRIC_M10B_NPI_71001_REPAIR1"
EXPECTED_HASHES = {
    "results/tables/metabric_m2/m06_metabric_clinical_master_LOCAL_ONLY.csv":
        "b7d9f60bd9f107464fe6dc9cbdd76b6f56641377026f59807b81739768ad232c",
    "results/tables/metabric_m7/m38_oof_predictions_LOCAL_ONLY.csv":
        "b1aa5837878423ea1ca4b525f00aba6bbddeb099c2e619abb942b3fdc255acb0",
    "results/tables/metabric_m8/m41_oof_predictions_LOCAL_ONLY.csv":
        "4ea23cb54021d17382ccf060c956ec19fc471f0a441d5ec8103b9888ab47d055",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser.parse_args()


def find_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if all((candidate / rel).is_file() for rel in EXPECTED_HASHES):
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root containing the locked METABRIC inputs."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(root: Path) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = root / rel
        value = sha256(path)
        observed[rel] = value
        if value != expected:
            raise RuntimeError(
                f"Locked input hash mismatch for {rel}\n"
                f"expected={expected}\nobserved={value}"
            )
    return observed


def load_cindex_backend() -> Tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]]:
    try:
        from sksurv.metrics import concordance_index_censored

        def cindex_sksurv(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
            mask = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
            if mask.sum() < 3 or int(np.nansum(event[mask])) == 0:
                return float("nan")
            return float(
                concordance_index_censored(
                    event[mask].astype(bool),
                    time[mask].astype(float),
                    risk[mask].astype(float),
                )[0]
            )
        return "sksurv.metrics.concordance_index_censored", cindex_sksurv
    except Exception:
        pass

    try:
        from lifelines.utils import concordance_index

        def cindex_lifelines(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
            mask = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
            if mask.sum() < 3 or int(np.nansum(event[mask])) == 0:
                return float("nan")
            # lifelines expects a larger prediction to indicate longer survival.
            return float(
                concordance_index(
                    time[mask].astype(float),
                    -risk[mask].astype(float),
                    event_observed=event[mask].astype(int),
                )
            )
        return "lifelines.utils.concordance_index", cindex_lifelines
    except Exception as exc:
        raise ImportError(
            "Neither scikit-survival nor lifelines is available. "
            "The existing METABRIC environment must provide one Harrell C-index backend."
        ) from exc


@dataclass
class AnalysisSpec:
    analysis: str
    modality: str
    frame: pd.DataFrame
    risk_columns: Tuple[str, ...]


def prepare_inputs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(
        root / "results/tables/metabric_m2/m06_metabric_clinical_master_LOCAL_ONLY.csv",
        low_memory=False,
    )
    combined = pd.read_csv(
        root / "results/tables/metabric_m7/m38_oof_predictions_LOCAL_ONLY.csv",
        low_memory=False,
    )
    modality = pd.read_csv(
        root / "results/tables/metabric_m8/m41_oof_predictions_LOCAL_ONLY.csv",
        low_memory=False,
    )

    required_clinical = {"sample_id", "npi", "os_months", "os_event"}
    required_combined = {
        "repeat", "fold", "sample_id", "time_months", "event",
        "model_risk", "clinical_risk",
    }
    required_modality = {
        "modality", "repeat", "fold", "sample_id", "time_months", "event",
        "clinical_risk", "modality_risk", "clinical_modality_risk",
    }
    for name, frame, required in (
        ("clinical", clinical, required_clinical),
        ("combined OOF", combined, required_combined),
        ("modality OOF", modality, required_modality),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{name} file is missing required columns: {missing}")

    npi = clinical[["sample_id", "npi", "os_months", "os_event"]].copy()
    if npi["sample_id"].duplicated().any():
        raise RuntimeError("Clinical master contains duplicated sample_id values.")

    combined = combined.merge(npi, on="sample_id", how="left", validate="many_to_one")
    modality = modality.merge(npi, on="sample_id", how="left", validate="many_to_one")

    # The locked OOF outcomes control evaluation. Clinical-master outcomes are
    # retained only for consistency checks.
    for name, frame in (("combined", combined), ("modality", modality)):
        check = frame[
            frame["os_months"].notna()
            & frame["os_event"].notna()
        ].copy()
        time_diff = np.nanmax(
            np.abs(check["time_months"].astype(float) - check["os_months"].astype(float))
        ) if len(check) else 0.0
        event_diff = int(
            (
                check["event"].astype(int)
                != check["os_event"].astype(int)
            ).sum()
        ) if len(check) else 0
        print(f"{name}: max OOF/clinical time difference={time_diff:.12g}; event mismatches={event_diff}")
        if time_diff > 1e-8 or event_diff != 0:
            raise RuntimeError(f"{name} OOF outcomes do not match the clinical master.")

    return clinical, combined, modality


def enforce_common_repeat_population(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[
        frame["npi"].notna()
        & frame["time_months"].notna()
        & frame["event"].notna()
    ].copy()

    repeats = sorted(frame["repeat"].unique().tolist())
    if not repeats:
        raise RuntimeError("No repeats remain after applying the NPI-observed restriction.")

    id_sets = [
        set(frame.loc[frame["repeat"] == repeat, "sample_id"].astype(str))
        for repeat in repeats
    ]
    common_ids = set.intersection(*id_sets)
    if not common_ids:
        raise RuntimeError("No patient is present in every repeat.")

    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.loc[frame["sample_id"].isin(common_ids)].copy()

    duplicate_counts = (
        frame.groupby(["repeat", "sample_id"], observed=True)
        .size()
    )
    if int(duplicate_counts.max()) != 1:
        raise RuntimeError("Expected one OOF row per patient per repeat.")

    expected_n = len(common_ids)
    per_repeat_n = frame.groupby("repeat")["sample_id"].nunique()
    if not bool((per_repeat_n == expected_n).all()):
        raise RuntimeError("Repeat populations remain inconsistent after intersection.")

    return frame


def build_analysis_specs(combined: pd.DataFrame, modality: pd.DataFrame) -> List[AnalysisSpec]:
    specs: List[AnalysisSpec] = []

    combined_common = enforce_common_repeat_population(combined)
    specs.append(
        AnalysisSpec(
            analysis="combined_reconstructed",
            modality="Multimodal",
            frame=combined_common,
            risk_columns=("npi", "clinical_risk", "model_risk"),
        )
    )

    for modality_name in sorted(modality["modality"].dropna().astype(str).unique()):
        subset = modality.loc[modality["modality"].astype(str) == modality_name].copy()
        subset = enforce_common_repeat_population(subset)
        specs.append(
            AnalysisSpec(
                analysis="modality_specific",
                modality=modality_name,
                frame=subset,
                risk_columns=(
                    "npi",
                    "clinical_risk",
                    "modality_risk",
                    "clinical_modality_risk",
                ),
            )
        )
    return specs


def point_metrics(
    spec: AnalysisSpec,
    cindex: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    rows: List[dict] = []
    for repeat, repeat_frame in spec.frame.groupby("repeat", sort=True):
        time = repeat_frame["time_months"].to_numpy(float)
        event = repeat_frame["event"].to_numpy(int)
        row = {
            "analysis": spec.analysis,
            "modality": spec.modality,
            "repeat": int(repeat),
            "n": int(len(repeat_frame)),
            "events": int(event.sum()),
        }
        for column in spec.risk_columns:
            row[f"{column}_c_index"] = cindex(
                time,
                event,
                repeat_frame[column].to_numpy(float),
            )
        rows.append(row)
    result = pd.DataFrame(rows)

    if "model_risk" in spec.risk_columns:
        result["delta_model_minus_clinical"] = (
            result["model_risk_c_index"] - result["clinical_risk_c_index"]
        )
        result["delta_model_minus_npi"] = (
            result["model_risk_c_index"] - result["npi_c_index"]
        )
    if "clinical_modality_risk" in spec.risk_columns:
        result["delta_model_minus_clinical"] = (
            result["clinical_modality_risk_c_index"]
            - result["clinical_risk_c_index"]
        )
        result["delta_model_minus_npi"] = (
            result["clinical_modality_risk_c_index"]
            - result["npi_c_index"]
        )
        result["delta_modality_only_minus_npi"] = (
            result["modality_risk_c_index"] - result["npi_c_index"]
        )
    result["delta_clinical_minus_npi"] = (
        result["clinical_risk_c_index"] - result["npi_c_index"]
    )
    return result


def create_repeat_lookup(spec: AnalysisSpec) -> Tuple[List[int], np.ndarray, Dict[int, pd.DataFrame]]:
    repeats = sorted(int(value) for value in spec.frame["repeat"].unique())
    patient_ids = np.array(
        sorted(spec.frame["sample_id"].astype(str).unique()),
        dtype=object,
    )
    lookup: Dict[int, pd.DataFrame] = {}
    for repeat in repeats:
        frame = (
            spec.frame.loc[spec.frame["repeat"] == repeat]
            .set_index("sample_id")
            .loc[patient_ids]
        )
        lookup[repeat] = frame
    return repeats, patient_ids, lookup


def bootstrap_one_analysis(
    spec: AnalysisSpec,
    cindex: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    repetitions: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    seed: int,
) -> pd.DataFrame:
    repeats, patient_ids, lookup = create_repeat_lookup(spec)
    n = len(patient_ids)

    if checkpoint_path.is_file():
        completed = pd.read_csv(checkpoint_path)
        expected_columns = {"bootstrap"}
        if not expected_columns.issubset(completed.columns):
            raise RuntimeError(f"Invalid checkpoint schema: {checkpoint_path}")
        start = int(completed["bootstrap"].max()) + 1 if len(completed) else 0
        rows = completed.to_dict("records")
        print(f"Resuming {spec.modality}: {start}/{repetitions} bootstrap samples complete.")
    else:
        start = 0
        rows: List[dict] = []

    # A deterministic seed per bootstrap index makes resumed and uninterrupted
    # runs bitwise consistent.
    for bootstrap_index in range(start, repetitions):
        rng = np.random.default_rng(seed + bootstrap_index)
        sampled_positions = rng.integers(0, n, size=n)
        sampled_ids = patient_ids[sampled_positions]

        repeat_rows: List[dict] = []
        for repeat in repeats:
            frame = lookup[repeat].loc[sampled_ids]
            time = frame["time_months"].to_numpy(float)
            event = frame["event"].to_numpy(int)
            values: Dict[str, float] = {}
            for column in spec.risk_columns:
                values[column] = cindex(
                    time,
                    event,
                    frame[column].to_numpy(float),
                )
            repeat_rows.append(values)

        repeat_frame = pd.DataFrame(repeat_rows)
        row: Dict[str, object] = {
            "analysis": spec.analysis,
            "modality": spec.modality,
            "bootstrap": bootstrap_index,
            "n": n,
            "events_in_resample": int(
                lookup[repeats[0]].loc[sampled_ids, "event"].sum()
            ),
            "repeats": len(repeats),
        }
        for column in spec.risk_columns:
            row[f"mean_{column}_c_index"] = float(repeat_frame[column].mean())

        if "model_risk" in spec.risk_columns:
            row["delta_model_minus_clinical"] = (
                row["mean_model_risk_c_index"]
                - row["mean_clinical_risk_c_index"]
            )
            row["delta_model_minus_npi"] = (
                row["mean_model_risk_c_index"]
                - row["mean_npi_c_index"]
            )
        if "clinical_modality_risk" in spec.risk_columns:
            row["delta_model_minus_clinical"] = (
                row["mean_clinical_modality_risk_c_index"]
                - row["mean_clinical_risk_c_index"]
            )
            row["delta_model_minus_npi"] = (
                row["mean_clinical_modality_risk_c_index"]
                - row["mean_npi_c_index"]
            )
            row["delta_modality_only_minus_npi"] = (
                row["mean_modality_risk_c_index"]
                - row["mean_npi_c_index"]
            )
        row["delta_clinical_minus_npi"] = (
            row["mean_clinical_risk_c_index"]
            - row["mean_npi_c_index"]
        )
        rows.append(row)

        if (
            (bootstrap_index + 1) % checkpoint_every == 0
            or bootstrap_index + 1 == repetitions
        ):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(
                f"{spec.modality}: bootstrap "
                f"{bootstrap_index + 1}/{repetitions}"
            )

    return pd.DataFrame(rows)


def classify_delta(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "positive_difference"
    if ci_high < 0:
        return "negative_difference"
    return "no_reliable_difference"


def summarize_bootstrap(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    identifiers = {"analysis", "modality", "bootstrap", "n", "events_in_resample", "repeats"}
    metric_columns = [
        column for column in bootstrap.columns
        if column not in identifiers
        and pd.api.types.is_numeric_dtype(bootstrap[column])
    ]

    point_means = {
        column: float(point[column].mean())
        for column in point.columns
        if column not in {"analysis", "modality", "repeat", "n", "events"}
        and pd.api.types.is_numeric_dtype(point[column])
    }

    rows: List[dict] = []
    for metric in metric_columns:
        values = bootstrap[metric].dropna().to_numpy(float)
        point_metric = metric
        if metric.startswith("mean_"):
            point_metric = metric.removeprefix("mean_")
        point_value = point_means.get(point_metric, float(np.nanmean(values)))
        ci_low, ci_high = np.quantile(values, [0.025, 0.975])
        rows.append({
            "analysis": str(bootstrap["analysis"].iloc[0]),
            "modality": str(bootstrap["modality"].iloc[0]),
            "metric": metric,
            "point_estimate": point_value,
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_sd": float(np.std(values, ddof=1)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "fraction_positive": float(np.mean(values > 0)),
            "claim_status": (
                classify_delta(float(ci_low), float(ci_high))
                if metric.startswith("delta_")
                else "descriptive_metric"
            ),
        })
    return pd.DataFrame(rows)


def discover_m46_summary(root: Path) -> Tuple[Path, pd.DataFrame]:
    folder = root / "results/tables/metabric_m9"
    for path in sorted(folder.glob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=1000)
        except Exception:
            continue
        required = {"analysis", "modality", "metric", "mean", "ci_low", "ci_high"}
        if required.issubset(frame.columns) and (frame["metric"].astype(str) == "delta_c_index").any():
            return path, pd.read_csv(path)
    raise FileNotFoundError("Could not discover the M9 bootstrap summary CSV.")


def discover_track_a_summary(root: Path) -> Tuple[Path, pd.DataFrame]:
    """
    Discover the locked M7 Track A paired-delta summary.

    M7 writes the model identifier as ``model_set`` rather than ``model``.
    The original M10B discovery gate incorrectly required ``model`` and
    therefore rejected the correct file after the NPI bootstraps had already
    completed. This repaired discovery accepts either name and normalizes the
    returned frame to ``model`` for downstream figure code.
    """
    folder = root / "results/tables/metabric_m7"
    preferred = folder / "m37_track_a_paired_delta_summary.csv"

    candidates = [preferred]
    candidates.extend(sorted(folder.glob("*track_a*paired*delta*summary*.csv")))
    candidates.extend(sorted(folder.glob("*track_a*delta*summary*.csv")))
    candidates.extend(sorted(folder.glob("*.csv")))

    seen = set()
    inspected = []
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            inspected.append(f"{path.name}: unreadable ({exc})")
            continue

        model_column = None
        if "model" in frame.columns:
            model_column = "model"
        elif "model_set" in frame.columns:
            model_column = "model_set"

        required = {"metric", "mean", "ci_low", "ci_high"}
        if model_column is not None and required.issubset(frame.columns):
            metrics = frame["metric"].astype(str).str.lower()
            if metrics.eq("delta_c_index_vs_clinical").any():
                normalized = frame.copy()
                if model_column != "model":
                    normalized = normalized.rename(columns={model_column: "model"})
                print(f"Track A paired-delta summary: {path.relative_to(root)}")
                return path, normalized

        inspected.append(
            f"{path.name}: columns={list(frame.columns)}"
        )

    detail = "\n".join(inspected[:30])
    raise FileNotFoundError(
        "Could not discover the locked Track A paired-delta summary CSV. "
        "Inspected files:\n" + detail
    )


def build_incremental_utility_data(root: Path) -> Tuple[pd.DataFrame, dict]:
    metadata: Dict[str, str] = {}
    track_a_path, track_a = discover_track_a_summary(root)
    metadata["track_a_summary"] = str(track_a_path.relative_to(root))
    track_a = track_a.loc[
        track_a["metric"].astype(str).str.lower().eq("delta_c_index_vs_clinical")
    ].copy()
    track_a["track"] = "Track A fixed transport"
    track_a["label"] = (
        track_a["model"].astype(str)
        .str.replace("clinical_", "", regex=False)
        .str.replace("_", "+", regex=False)
        .str.upper()
    )
    track_a_data = track_a.rename(
        columns={
            "mean": "delta_c_index",
        }
    )[["track", "label", "delta_c_index", "ci_low", "ci_high"]]

    m46_path, m46 = discover_m46_summary(root)
    metadata["m46_summary"] = str(m46_path.relative_to(root))
    m46 = m46.loc[m46["metric"].astype(str).eq("delta_c_index")].copy()
    m46["track"] = "Track B nested reconstruction"
    m46["label"] = m46["modality"].astype(str)
    track_b_data = m46.rename(
        columns={"mean": "delta_c_index"}
    )[["track", "label", "delta_c_index", "ci_low", "ci_high"]]

    data = pd.concat([track_a_data, track_b_data], ignore_index=True)
    data["claim_status"] = [
        classify_delta(low, high)
        for low, high in zip(data["ci_low"], data["ci_high"])
    ]
    return data, metadata


def plot_incremental_utility(data: pd.DataFrame, out_dir: Path) -> None:
    plot_data = data.copy()
    plot_data["display"] = plot_data["track"] + ": " + plot_data["label"]
    plot_data = plot_data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.52 * len(plot_data) + 1.5)))
    x = plot_data["delta_c_index"].to_numpy(float)
    lower = x - plot_data["ci_low"].to_numpy(float)
    upper = plot_data["ci_high"].to_numpy(float) - x
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        linewidth=1.2,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, plot_data["display"])
    ax.set_xlabel(r"Incremental Harrell C-index, $\Delta C$")
    ax.set_title("Incremental prognostic discrimination beyond clinical variables")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_incremental_utility.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure2_incremental_utility.pdf", bbox_inches="tight")
    plt.close(fig)


def build_stability_utility_data(root: Path, utility: pd.DataFrame) -> pd.DataFrame:
    stability_path = (
        root
        / "results/tables/metabric_m9/m47_chance_adjusted_stability_summary.csv"
    )
    stability = pd.read_csv(stability_path)
    required = {"modality", "mean_chance_adjusted_overlap"}
    if not required.issubset(stability.columns):
        raise KeyError("M47 stability summary has an unexpected schema.")

    track_b = utility.loc[
        utility["track"].eq("Track B nested reconstruction")
    ].copy()
    result = stability.merge(
        track_b[
            ["label", "delta_c_index", "ci_low", "ci_high", "claim_status"]
        ].rename(columns={"label": "modality"}),
        on="modality",
        how="inner",
        validate="one_to_one",
    )
    if len(result) != 5:
        raise RuntimeError(
            f"Expected five modality/stability rows, found {len(result)}."
        )
    return result


def plot_stability_utility(data: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    x = data["mean_chance_adjusted_overlap"].to_numpy(float)
    y = data["delta_c_index"].to_numpy(float)
    lower = y - data["ci_low"].to_numpy(float)
    upper = data["ci_high"].to_numpy(float) - y
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        linewidth=1.2,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    for _, row in data.iterrows():
        ax.annotate(
            str(row["modality"]),
            (float(row["mean_chance_adjusted_overlap"]), float(row["delta_c_index"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Mean chance-adjusted selection overlap")
    ax.set_ylabel(r"Incremental Harrell C-index, $\Delta C$")
    ax.set_title("Selection stability did not imply incremental utility")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_stability_vs_incremental_utility.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure4_stability_vs_incremental_utility.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_npi_benchmark(summary: pd.DataFrame, out_dir: Path) -> None:
    data = summary.loc[
        summary["metric"].eq("delta_model_minus_npi")
    ].copy()
    if data.empty:
        return
    data["display"] = data["analysis"] + ": " + data["modality"]
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    x = data["point_estimate"].to_numpy(float)
    lower = x - data["ci_low"].to_numpy(float)
    upper = data["ci_high"].to_numpy(float) - x

    fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.55 * len(data) + 1.5)))
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        linewidth=1.2,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, data["display"])
    ax.set_xlabel(r"Harrell C-index difference versus NPI")
    ax.set_title("Secondary NPI benchmark on common observed-NPI subsets")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "figureS_npi_benchmark.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figureS_npi_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def calibration_status(root: Path) -> pd.DataFrame:
    rows = [
        {
            "track": "Track A fixed transport",
            "assessment": "NPI common-subset comparison",
            "status": "NOT_ESTIMABLE_FROM_LOCKED_OUTPUTS",
            "reason": (
                "Patient-level Track A predictions were not saved. "
                "Track A is not refitted for a secondary benchmark."
            ),
        },
        {
            "track": "Track A fixed transport",
            "assessment": "Aggregate calibration metrics",
            "status": "AVAILABLE_FROM_M7",
            "reason": (
                "M7 contains aggregate calibration and Brier metrics, "
                "but no locked patient-level prediction file for a new common-subset plot."
            ),
        },
        {
            "track": "Track B nested reconstruction",
            "assessment": "NPI common-subset comparison",
            "status": "ESTIMABLE_FROM_LOCKED_OOF_RISK_SCORES",
            "reason": (
                "NPI merges to locked repeated OOF risk scores using sample_id."
            ),
        },
        {
            "track": "Track B nested reconstruction",
            "assessment": "5-year calibration curve",
            "status": "NOT_ESTIMABLE_FROM_LOCKED_OUTPUTS",
            "reason": (
                "The saved OOF files contain relative risk scores only; "
                "they do not contain 5-year survival probabilities or fold-specific baseline hazards. "
                "No evaluation-sample recalibration is performed."
            ),
        },
        {
            "track": "Track B nested reconstruction",
            "assessment": "NPI+omics model",
            "status": "NOT_FITTED",
            "reason": (
                "The extension compares NPI with locked models and does not introduce a new stacked or refitted model."
            ),
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100.")
    root = find_root(Path.cwd())
    out_dir = root / "results/tables/metabric_m10b"
    fig_dir = root / "results/figures/metabric_m10b"
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("METABRIC STAGE M10B - LOCKED OOF NPI BENCHMARK AND PUBLICATION FIGURES")
    print("=" * 124)
    print(f"Project root: {root}")
    print(f"Bootstrap repetitions: {args.bootstrap}")
    print("M1-M9 are NOT rerun.")
    print("No feature selection is repeated.")
    print("No prognostic model is refitted.")
    print("No NPI+omics model is created.")
    print("Calibration is not fabricated from relative risk scores.")
    print("=" * 124)

    hashes = validate_inputs(root)
    backend_name, cindex = load_cindex_backend()
    print(f"Harrell C-index backend: {backend_name}")

    protocol = {
        "protocol_id": PROTOCOL_ID,
        "status": "METABRIC_M10B_SECONDARY_EXTENSION_LOCKED_BEFORE_RESULTS",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sampling_method": (
            "paired patient bootstrap of locked repeated OOF predictions, "
            "averaging C-index contrasts over repeats"
        ),
        "primary_m10b_question": (
            "How do locked Track B predictions compare with the Nottingham "
            "Prognostic Index on the same NPI-observed patients?"
        ),
        "boundaries": [
            "Primary M7-M9 claims are unchanged.",
            "Track A is not refitted because patient-level predictions were not saved.",
            "No feature selection is repeated.",
            "No NPI+omics model is fitted.",
            "The bootstrap is conditional on the fitted repeated OOF models.",
            "Five-year calibration is not estimated from relative risk scores alone.",
        ],
        "input_hashes": hashes,
        "cindex_backend": backend_name,
    }
    protocol_path = out_dir / "m51_m10b_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    clinical, combined, modality = prepare_inputs(root)
    specs = build_analysis_specs(combined, modality)

    all_points: List[pd.DataFrame] = []
    all_bootstraps: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []

    for analysis_index, spec in enumerate(specs):
        repeats = sorted(spec.frame["repeat"].unique())
        n = spec.frame["sample_id"].nunique()
        events = int(
            spec.frame.loc[spec.frame["repeat"] == repeats[0], "event"].sum()
        )
        print("-" * 124)
        print(
            f"{spec.analysis} | {spec.modality}: "
            f"n={n} events={events} repeats={len(repeats)}"
        )

        point = point_metrics(spec, cindex)
        all_points.append(point)

        safe_name = (
            f"{spec.analysis}_{spec.modality}"
            .replace(" ", "_")
            .replace("/", "_")
        )
        checkpoint_path = checkpoint_dir / f"{safe_name}_bootstrap.csv"
        bootstrap = bootstrap_one_analysis(
            spec=spec,
            cindex=cindex,
            repetitions=args.bootstrap,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=checkpoint_path,
            seed=BOOTSTRAP_SEED + analysis_index * 100000,
        )
        all_bootstraps.append(bootstrap)
        all_summaries.append(summarize_bootstrap(point, bootstrap))

    repeat_metrics = pd.concat(all_points, ignore_index=True)
    bootstrap_draws = pd.concat(all_bootstraps, ignore_index=True)
    bootstrap_summary = pd.concat(all_summaries, ignore_index=True)

    repeat_metrics.to_csv(out_dir / "m51_npi_repeat_metrics.csv", index=False)
    bootstrap_draws.to_csv(out_dir / "m51_npi_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(out_dir / "m51_npi_bootstrap_summary.csv", index=False)

    status = calibration_status(root)
    status.to_csv(out_dir / "m51_calibration_and_extension_status.csv", index=False)

    utility, utility_metadata = build_incremental_utility_data(root)
    utility.to_csv(out_dir / "m51_incremental_utility_figure_data.csv", index=False)
    plot_incremental_utility(utility, fig_dir)

    stability_utility = build_stability_utility_data(root, utility)
    stability_utility.to_csv(
        out_dir / "m51_stability_vs_utility_figure_data.csv",
        index=False,
    )
    plot_stability_utility(stability_utility, fig_dir)
    plot_npi_benchmark(bootstrap_summary, fig_dir)

    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "METABRIC_M10B_COMPLETE",
        "primary_claims_changed": False,
        "track_a_npi_status": "NOT_ESTIMABLE_FROM_LOCKED_OUTPUTS_NO_REFIT",
        "track_b_npi_status": "COMPLETED_ON_COMMON_NPI_OBSERVED_SUBSETS",
        "track_b_calibration_status": (
            "NOT_ESTIMABLE_FROM_LOCKED_RELATIVE_RISK_SCORES"
        ),
        "npi_observed_in_clinical_master": int(clinical["npi"].notna().sum()),
        "clinical_master_n": int(len(clinical)),
        "analyses": [
            {
                "analysis": spec.analysis,
                "modality": spec.modality,
                "n": int(spec.frame["sample_id"].nunique()),
                "events": int(
                    spec.frame.loc[
                        spec.frame["repeat"] == spec.frame["repeat"].min(),
                        "event",
                    ].sum()
                ),
                "repeats": int(spec.frame["repeat"].nunique()),
            }
            for spec in specs
        ],
        "utility_input_files": utility_metadata,
        "outputs": [
            "m51_npi_repeat_metrics.csv",
            "m51_npi_bootstrap_draws.csv",
            "m51_npi_bootstrap_summary.csv",
            "m51_calibration_and_extension_status.csv",
            "m51_incremental_utility_figure_data.csv",
            "m51_stability_vs_utility_figure_data.csv",
            "m51_m10b_report.json",
        ],
        "figures": [
            "figure2_incremental_utility.png/pdf",
            "figure4_stability_vs_incremental_utility.png/pdf",
            "figureS_npi_benchmark.png/pdf",
        ],
    }
    report_path = out_dir / "m51_m10b_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 124)
    print("M10B SUMMARY")
    print("=" * 124)
    display = bootstrap_summary.loc[
        bootstrap_summary["metric"].isin(
            [
                "delta_model_minus_clinical",
                "delta_model_minus_npi",
                "delta_clinical_minus_npi",
            ]
        )
    ].copy()
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 20,
        "display.width", 180,
    ):
        print(
            display[
                [
                    "analysis", "modality", "metric", "point_estimate",
                    "ci_low", "ci_high", "claim_status",
                ]
            ].to_string(index=False)
        )

    print("-" * 124)
    print("CALIBRATION / EXTENSION STATUS")
    print(status.to_string(index=False))
    print("-" * 124)
    print("Figures:")
    for path in sorted(fig_dir.glob("figure*")):
        print(f"  {path.relative_to(root)}")
    print("=" * 124)
    print("STATUS: METABRIC_M10B_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
