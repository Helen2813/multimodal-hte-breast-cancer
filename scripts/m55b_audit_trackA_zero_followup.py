#!/usr/bin/env python3
"""
M55b -- Paper 2B read-only audit:
1) Track A fixed-panel external-validation artifacts (M7.37)
2) zero-follow-up observations in Track B OOF predictions (M7.38/M8.41)

Purpose
-------
This script does NOT refit any prognostic model and does NOT rerun any bootstrap.
It only reads already-saved artifacts, recomputes summaries from saved draws /
OOF predictions, and writes a new timestamped audit report.

Run from repository root, for example:

    .\\.venv\\Scripts\\python.exe .\\scripts\\m55b_audit_trackA_zero_followup.py

Interpretation boundary
-----------------------
- Track A calibration slopes and other point metrics are checked for record
  consistency against the saved M37 result file; they are NOT independently
  recomputed because M37 does not save the patient-level Track A risk/survival
  predictions needed for a no-refit recalculation.
- Track A bootstrap summaries ARE recomputed from the saved bootstrap draws.
- Zero-time sensitivity removes rows only at the evaluation stage. No model is
  retrained, so this is not a replacement analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

MANUSCRIPT_TRACK_A_REFERENCES = {
    "clinical": {
        "calibration_slope": 1.247,
    },
    "clinical_cna": {
        "delta_c_index_vs_clinical": 0.0004,
        "calibration_slope": 1.242,
    },
    "clinical_rna": {
        "delta_c_index_vs_clinical": -0.0136,
        "calibration_slope": 0.814,
    },
    "clinical_rna_cna": {
        "delta_c_index_vs_clinical": -0.0121,
        "calibration_slope": 0.803,
    },
}

MODEL_ORDER = ["clinical", "clinical_cna", "clinical_rna", "clinical_rna_cna"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only Track A + zero-follow-up audit for Paper 2B."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Default: auto-detect.",
    )
    p.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Optional parent directory for reports. Default: results/reports/.",
    )
    return p.parse_args()


def find_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "metabric_m7_config.json").is_file():
            raise FileNotFoundError(f"Not a project root: {root}")
        return root

    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (
                (candidate / "metabric_m7_config.json").is_file()
                and (candidate / "metabric_m8_config.json").is_file()
                and (candidate / "metabric_m9_config.json").is_file()
            ):
                return candidate
    raise FileNotFoundError(
        "Could not auto-detect repository root containing metabric_m7_config.json, "
        "metabric_m8_config.json and metabric_m9_config.json."
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def short_id(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    return a[np.isfinite(a)]


def summarize_vector(values: Iterable[float]) -> dict[str, float | int]:
    v = finite(values)
    if len(v) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "median": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "fraction_positive": math.nan,
        }
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else math.nan,
        "median": float(np.median(v)),
        "ci_low": float(np.quantile(v, 0.025)),
        "ci_high": float(np.quantile(v, 0.975)),
        "fraction_positive": float(np.mean(v > 0)),
    }


def cindex(time_values: np.ndarray, event_values: np.ndarray, risk_values: np.ndarray) -> float:
    time_values = np.asarray(time_values, dtype=float)
    event_values = np.asarray(event_values, dtype=int)
    risk_values = np.asarray(risk_values, dtype=float)
    mask = np.isfinite(time_values) & np.isfinite(event_values) & np.isfinite(risk_values)
    if mask.sum() < 3 or int(event_values[mask].sum()) == 0:
        return float("nan")
    try:
        from lifelines.utils import concordance_index

        return float(
            concordance_index(
                time_values[mask],
                -risk_values[mask],
                event_observed=event_values[mask],
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "lifelines is required for the zero-time OOF sensitivity. "
            "Run this script in the same .venv used by the analysis."
        ) from exc


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        status: str,
        check: str,
        details: str,
        required: bool = True,
    ) -> None:
        self.checks.append(
            {
                "status": status,
                "check": check,
                "details": details,
                "required": bool(required),
            }
        )

    def require_file(self, path: Path, label: str) -> bool:
        if path.is_file():
            self.add("PASS", label, f"Found: {path}", True)
            return True
        self.add("NOT_VERIFIED", label, f"Missing: {path}", True)
        return False

    def counts(self) -> Counter:
        return Counter(row["status"] for row in self.checks)

    def overall(self) -> str:
        c = self.counts()
        if c.get("FAIL", 0) or c.get("NOT_VERIFIED", 0):
            return "NOT_COMPLETE"
        if c.get("REVIEW", 0):
            return "REVIEW_NEEDED"
        return "CHECKED_ARTIFACTS_CONSISTENT"


def track_a_paths(root: Path) -> dict[str, Path]:
    base = root / "results" / "tables" / "metabric_m7"
    return {
        "full_results": base / "m37_track_a_full_results.csv",
        "bootstrap": base / "m37_track_a_bootstrap_1000.csv",
        "bootstrap_summary": base / "m37_track_a_bootstrap_summary.csv",
        "deltas": base / "m37_track_a_paired_deltas_1000.csv",
        "delta_summary": base / "m37_track_a_paired_delta_summary.csv",
        "prefix_check": base / "m37_pilot_prefix_verification.csv",
        "registry": base / "m37_track_a_model_registry.json",
    }


def track_b_paths(root: Path) -> dict[str, Path]:
    return {
        "m38_predictions": root
        / "results"
        / "tables"
        / "metabric_m7"
        / "m38_oof_predictions_LOCAL_ONLY.csv",
        "m41_predictions": root
        / "results"
        / "tables"
        / "metabric_m8"
        / "m41_oof_predictions_LOCAL_ONLY.csv",
    }


def audit_track_a(
    root: Path,
    out: Path,
    audit: Audit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    print("\n[1/3] Auditing Track A saved M37 artifacts")
    paths = track_a_paths(root)
    for key, path in paths.items():
        audit.require_file(path, f"Track A file: {key}")

    if not paths["full_results"].is_file() or not paths["deltas"].is_file():
        return [], [], []

    full = read_csv(paths["full_results"])
    deltas = read_csv(paths["deltas"])

    required_full_cols = {"model_set", "harrell_c_index", "calibration_slope"}
    missing = required_full_cols - set(full.columns)
    if missing:
        audit.add(
            "FAIL",
            "Track A full-results schema",
            f"Missing required columns: {sorted(missing)}",
            True,
        )
        return [], [], []

    model_sets = set(full["model_set"].astype(str))
    expected = set(MODEL_ORDER)
    if model_sets == expected:
        audit.add(
            "PASS",
            "Track A model-set registry",
            f"Observed all expected models: {MODEL_ORDER}",
            True,
        )
    else:
        audit.add(
            "REVIEW",
            "Track A model-set registry",
            f"Observed={sorted(model_sets)} expected={MODEL_ORDER}",
            True,
        )

    clinical_rows = full.loc[full["model_set"].astype(str) == "clinical"]
    if len(clinical_rows) != 1:
        audit.add(
            "FAIL",
            "Track A clinical reference row",
            f"Expected one clinical row, observed {len(clinical_rows)}.",
            True,
        )
        return [], [], []
    clinical_c = float(numeric(clinical_rows["harrell_c_index"]).iloc[0])

    full_rows: list[dict[str, Any]] = []
    for _, row in full.iterrows():
        model = str(row["model_set"])
        harrell = float(pd.to_numeric(row.get("harrell_c_index"), errors="coerce"))
        item: dict[str, Any] = {
            "model_set": model,
            "harrell_c_index": harrell,
            "point_delta_c_vs_clinical": harrell - clinical_c if model != "clinical" else 0.0,
        }
        for col in [
            "tcga_n",
            "metabric_n",
            "features",
            "rna_features",
            "cna_features",
            "tcga_train_c_index",
            "uno_c_10y",
            "binary_auc_5y",
            "binary_auc_5y_n",
            "binary_auc_10y",
            "binary_auc_10y_n",
            "ipcw_auc_5y",
            "ipcw_auc_10y",
            "brier_5y",
            "brier_10y",
            "integrated_brier_1_to_10y",
            "calibration_slope",
            "mean_predicted_survival_5y",
            "observed_km_survival_5y",
            "observed_minus_predicted_survival_5y",
            "mean_predicted_survival_10y",
            "observed_km_survival_10y",
            "observed_minus_predicted_survival_10y",
        ]:
            if col in full.columns:
                item[col] = float(pd.to_numeric(row.get(col), errors="coerce"))
        full_rows.append(item)

    save_csv(out / "trackA_full_results_audit.csv", full_rows)

    required_delta_cols = {"repetition", "model_set", "delta_c_index_vs_clinical"}
    missing_delta = required_delta_cols - set(deltas.columns)
    if missing_delta:
        audit.add(
            "FAIL",
            "Track A paired-delta schema",
            f"Missing columns: {sorted(missing_delta)}",
            True,
        )
        return full_rows, [], []

    bootstrap_rows: list[dict[str, Any]] = []
    for model in sorted(deltas["model_set"].dropna().astype(str).unique()):
        sub = deltas.loc[deltas["model_set"].astype(str) == model].copy()
        for metric in [
            "delta_c_index_vs_clinical",
            "delta_auc_5y_vs_clinical",
            "delta_auc_10y_vs_clinical",
        ]:
            if metric not in sub.columns:
                continue
            s = summarize_vector(numeric(sub[metric]).to_numpy())
            bootstrap_rows.append(
                {
                    "model_set": model,
                    "metric": metric,
                    **s,
                    "evidence": "recomputed from saved M37 draws; no new resampling",
                }
            )
    save_csv(out / "trackA_bootstrap_recomputed.csv", bootstrap_rows)

    cfg7 = load_json(root / "metabric_m7_config.json")
    configured_reps = int(cfg7.get("track_a", {}).get("bootstrap_repetitions", -1))
    unique_reps = int(pd.to_numeric(deltas["repetition"], errors="coerce").dropna().nunique())
    audit.add(
        "PASS" if unique_reps == configured_reps else "FAIL",
        "Track A bootstrap repetition count",
        f"Observed {unique_reps} unique draws; config={configured_reps}.",
        True,
    )

    saved_summary_cmp: list[dict[str, Any]] = []
    if paths["delta_summary"].is_file():
        saved = read_csv(paths["delta_summary"])
        for rec in bootstrap_rows:
            model, metric = rec["model_set"], rec["metric"]
            match = saved.loc[
                (saved["model_set"].astype(str) == str(model))
                & (saved["metric"].astype(str) == str(metric))
            ]
            if len(match) != 1:
                saved_summary_cmp.append(
                    {
                        "model_set": model,
                        "metric": metric,
                        "status": "NOT_VERIFIED",
                        "details": f"saved summary rows={len(match)}",
                    }
                )
                audit.add(
                    "NOT_VERIFIED",
                    f"Track A saved summary: {model}/{metric}",
                    f"Expected one saved row, observed {len(match)}.",
                    True,
                )
                continue
            r = match.iloc[0]
            diffs = {}
            for col in ["mean", "sd", "median", "ci_low", "ci_high", "fraction_positive"]:
                if col in r.index and col in rec:
                    diffs[col] = abs(float(pd.to_numeric(r[col], errors="coerce")) - float(rec[col]))
            max_diff = max(diffs.values()) if diffs else math.nan
            status = "PASS" if (not diffs or max_diff <= 1e-12) else "REVIEW"
            saved_summary_cmp.append(
                {
                    "model_set": model,
                    "metric": metric,
                    "status": status,
                    "max_absolute_difference": max_diff,
                    "details": json.dumps(diffs, sort_keys=True),
                }
            )
            audit.add(
                status,
                f"Track A saved summary: {model}/{metric}",
                f"max absolute difference={max_diff:.3g}" if np.isfinite(max_diff) else "no comparable fields",
                True,
            )
    save_csv(out / "trackA_saved_summary_comparison.csv", saved_summary_cmp)

    manuscript_cmp: list[dict[str, Any]] = []
    row_lookup = {r["model_set"]: r for r in full_rows}
    boot_lookup = {(r["model_set"], r["metric"]): r for r in bootstrap_rows}
    for model, refs in MANUSCRIPT_TRACK_A_REFERENCES.items():
        if model not in row_lookup:
            continue
        for metric, expected_value in refs.items():
            if metric == "delta_c_index_vs_clinical":
                observed = boot_lookup.get((model, "delta_c_index_vs_clinical"), {}).get("mean", math.nan)
                tolerance = 0.00015
                evidence = "bootstrap mean from saved 1000 draws"
            else:
                observed = row_lookup[model].get(metric, math.nan)
                tolerance = 0.0015
                evidence = "saved M37 point metric; not independently refit"
            diff = abs(float(observed) - float(expected_value)) if np.isfinite(observed) else math.nan
            status = "PASS" if np.isfinite(diff) and diff <= tolerance else "REVIEW"
            manuscript_cmp.append(
                {
                    "model_set": model,
                    "metric": metric,
                    "manuscript_reference_rounded": expected_value,
                    "observed": observed,
                    "absolute_difference": diff,
                    "tolerance": tolerance,
                    "status": status,
                    "evidence": evidence,
                }
            )
            audit.add(
                status,
                f"Track A manuscript reference: {model}/{metric}",
                f"reference={expected_value}; observed={observed}; evidence={evidence}",
                False,
            )
    save_csv(out / "trackA_manuscript_reference_check.csv", manuscript_cmp)

    registry_rows: list[dict[str, Any]] = []
    if paths["registry"].is_file():
        registry = load_json(paths["registry"])
        if isinstance(registry, dict):
            for model, payload in registry.items():
                rec: dict[str, Any] = {"model_set": model}
                if isinstance(payload, dict):
                    rec["registry_keys"] = "|".join(sorted(map(str, payload.keys())))
                    if "concordance_train" in payload:
                        rec["registry_concordance_train"] = float(payload["concordance_train"])
                        if model in row_lookup and "tcga_train_c_index" in row_lookup[model]:
                            rec["full_results_tcga_train_c_index"] = row_lookup[model]["tcga_train_c_index"]
                            rec["absolute_difference"] = abs(
                                rec["registry_concordance_train"]
                                - rec["full_results_tcga_train_c_index"]
                            )
                            status = "PASS" if rec["absolute_difference"] <= 1e-12 else "REVIEW"
                            rec["status"] = status
                            audit.add(
                                status,
                                f"Track A registry train C: {model}",
                                f"difference={rec['absolute_difference']:.3g}",
                                False,
                            )
                    for key in ["coefficients", "params", "coef", "coef_"]:
                        if key in payload:
                            obj = payload[key]
                            try:
                                rec["coefficient_count"] = len(obj)
                            except Exception:
                                pass
                            break
                registry_rows.append(rec)
        else:
            audit.add(
                "REVIEW",
                "Track A model registry schema",
                f"Expected dict, observed {type(registry).__name__}.",
                False,
            )
    save_csv(out / "trackA_model_registry_audit.csv", registry_rows)

    if paths["prefix_check"].is_file():
        prefix = read_csv(paths["prefix_check"])
        if "pass" in prefix.columns and len(prefix):
            raw = str(prefix.iloc[0]["pass"]).strip().lower()
            ok = raw in {"true", "1", "1.0", "yes"}
            audit.add(
                "PASS" if ok else "REVIEW",
                "Track A pilot-prefix identity check",
                prefix.iloc[0].to_json(),
                False,
            )

    for row in full_rows:
        model = row["model_set"]
        for metric in [
            "harrell_c_index",
            "uno_c_10y",
            "binary_auc_5y",
            "binary_auc_10y",
            "ipcw_auc_5y",
            "ipcw_auc_10y",
        ]:
            if metric in row and np.isfinite(row[metric]):
                val = float(row[metric])
                audit.add(
                    "PASS" if 0 <= val <= 1 else "FAIL",
                    f"Track A metric range: {model}/{metric}",
                    f"value={val}",
                    True,
                )
        for metric in ["brier_5y", "brier_10y", "integrated_brier_1_to_10y"]:
            if metric in row and np.isfinite(row[metric]):
                val = float(row[metric])
                audit.add(
                    "PASS" if val >= 0 else "FAIL",
                    f"Track A metric range: {model}/{metric}",
                    f"value={val}",
                    True,
                )
        if "calibration_slope" in row:
            val = float(row["calibration_slope"])
            audit.add(
                "PASS" if np.isfinite(val) and val > 0 else "REVIEW",
                f"Track A calibration slope finite/positive: {model}",
                f"value={val}; record-consistency only, not independent recalculation",
                True,
            )

    print("  Track A saved point/secondary metrics read successfully.")
    for rec in bootstrap_rows:
        if rec["metric"] == "delta_c_index_vs_clinical":
            print(
                f"  {rec['model_set']}: delta C bootstrap mean={rec['mean']:+.8f}; "
                f"95%=[{rec['ci_low']:+.8f}, {rec['ci_high']:+.8f}]"
            )

    return full_rows, bootstrap_rows, manuscript_cmp


def _audit_one_oof(
    frame: pd.DataFrame,
    label: str,
    model_col: str,
    clinical_col: str,
    modality: str,
    audit: Audit,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"repeat", "fold", "sample_id", "time_months", "event", model_col, clinical_col}
    missing = required - set(frame.columns)
    if missing:
        audit.add(
            "FAIL",
            f"{label}/{modality}: OOF schema",
            f"Missing columns: {sorted(missing)}",
            True,
        )
        return {}, {}

    f = frame.copy()
    f["sample_id"] = f["sample_id"].astype(str)
    f["repeat"] = pd.to_numeric(f["repeat"], errors="coerce").astype("Int64")
    f["fold"] = pd.to_numeric(f["fold"], errors="coerce").astype("Int64")
    f["time_months"] = numeric(f["time_months"])
    f["event"] = numeric(f["event"])
    f[model_col] = numeric(f[model_col])
    f[clinical_col] = numeric(f[clinical_col])

    zero = f.loc[f["time_months"] <= 0].copy()
    unique_zero_ids = sorted(zero["sample_id"].dropna().astype(str).unique())

    zero_summary = {
        "analysis": label,
        "modality": modality,
        "rows_total": int(len(f)),
        "unique_patients": int(f["sample_id"].nunique()),
        "zero_or_negative_rows": int(len(zero)),
        "zero_or_negative_unique_patients": int(len(unique_zero_ids)),
        "zero_time_rows": int((f["time_months"] == 0).sum()),
        "negative_time_rows": int((f["time_months"] < 0).sum()),
        "zero_event_0_rows": int(((f["time_months"] <= 0) & (f["event"] == 0)).sum()),
        "zero_event_1_rows": int(((f["time_months"] <= 0) & (f["event"] == 1)).sum()),
        "hashed_patient_tokens": "|".join(short_id(x) for x in unique_zero_ids),
    }

    if len(zero) == 0:
        audit.add("PASS", f"{label}/{modality}: zero follow-up", "No time<=0 rows.", True)
    else:
        audit.add(
            "REVIEW",
            f"{label}/{modality}: zero follow-up",
            (
                f"time<=0 rows={len(zero)}; unique patients={len(unique_zero_ids)}; "
                f"event counts={Counter(zero['event'].astype(int).tolist())}"
            ),
            True,
        )

    def point_delta(data: pd.DataFrame) -> float:
        vals = []
        for _, r in data.groupby("repeat", sort=True):
            c = cindex(
                r["time_months"].to_numpy(float),
                r["event"].to_numpy(int),
                r[clinical_col].to_numpy(float),
            )
            m = cindex(
                r["time_months"].to_numpy(float),
                r["event"].to_numpy(int),
                r[model_col].to_numpy(float),
            )
            vals.append(m - c)
        return float(np.nanmean(vals))

    original = point_delta(f)
    without_zero = point_delta(f.loc[f["time_months"] > 0].copy())
    sensitivity = {
        "analysis": label,
        "modality": modality,
        "original_oof_mean_delta_c": original,
        "excluding_time_le_0_oof_mean_delta_c": without_zero,
        "absolute_change": abs(without_zero - original),
        "signed_change": without_zero - original,
        "zero_or_negative_unique_patients": len(unique_zero_ids),
        "interpretation": "Evaluation-only sensitivity; models were not refit.",
    }
    if len(zero):
        status = "PASS" if abs(without_zero - original) <= 0.0005 else "REVIEW"
        audit.add(
            status,
            f"{label}/{modality}: zero-follow-up point sensitivity",
            (
                f"delta C {original:+.8f} -> {without_zero:+.8f}; "
                f"change={without_zero-original:+.8f}; no refit"
            ),
            True,
        )
    return zero_summary, sensitivity


def audit_zero_followup(
    root: Path,
    out: Path,
    audit: Audit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("\n[2/3] Auditing zero-follow-up records in saved Track B OOF predictions")
    paths = track_b_paths(root)
    for key, path in paths.items():
        audit.require_file(path, f"Track B OOF file: {key}")

    summaries: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []

    if paths["m38_predictions"].is_file():
        frame = read_csv(paths["m38_predictions"])
        s, d = _audit_one_oof(
            frame,
            "combined_reconstructed",
            "model_risk",
            "clinical_risk",
            "Multimodal",
            audit,
        )
        if s:
            summaries.append(s)
        if d:
            sensitivities.append(d)

    if paths["m41_predictions"].is_file():
        frame = read_csv(paths["m41_predictions"])
        if "modality" not in frame.columns:
            audit.add("FAIL", "M41 OOF modality column", "Column 'modality' missing.", True)
        else:
            for modality, sub in frame.groupby("modality", sort=True):
                s, d = _audit_one_oof(
                    sub.copy(),
                    "modality_specific",
                    "clinical_modality_risk",
                    "clinical_risk",
                    str(modality),
                    audit,
                )
                if s:
                    summaries.append(s)
                if d:
                    sensitivities.append(d)

    save_csv(out / "zero_followup_summary.csv", summaries)
    save_csv(out / "zero_followup_point_sensitivity.csv", sensitivities)

    tokens: set[str] = set()
    for row in summaries:
        for token in str(row.get("hashed_patient_tokens", "")).split("|"):
            token = token.strip()
            if token:
                tokens.add(token)

    if len(tokens) == 1 and tokens:
        audit.add(
            "PASS",
            "Cross-analysis zero-follow-up identity",
            f"All affected analyses point to the same hashed patient token: {next(iter(tokens))}.",
            True,
        )
    elif len(tokens) > 1:
        audit.add(
            "REVIEW",
            "Cross-analysis zero-follow-up identity",
            f"Multiple hashed patient tokens observed: {sorted(tokens)}.",
            True,
        )
    else:
        audit.add(
            "INFO",
            "Cross-analysis zero-follow-up identity",
            "No time<=0 patient token found.",
            False,
        )

    for row in summaries:
        print(
            f"  {row['modality']}: time<=0 patients={row['zero_or_negative_unique_patients']}; "
            f"token={row['hashed_patient_tokens'] or 'none'}"
        )
    for row in sensitivities:
        print(
            f"  {row['modality']}: delta C {row['original_oof_mean_delta_c']:+.8f} -> "
            f"{row['excluding_time_le_0_oof_mean_delta_c']:+.8f} after evaluation-only exclusion"
        )

    return summaries, sensitivities


def file_manifest(root: Path, out: Path) -> None:
    rows = []
    for path in list(track_a_paths(root).values()) + list(track_b_paths(root).values()):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    save_csv(out / "input_manifest.csv", rows)


def write_report(
    out: Path,
    audit: Audit,
    track_a_full: list[dict[str, Any]],
    track_a_boot: list[dict[str, Any]],
    zero_summary: list[dict[str, Any]],
    zero_sens: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    counts = audit.counts()
    status = audit.overall()

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "checks_by_severity": dict(counts),
        "elapsed_seconds": round(elapsed, 3),
        "scope": "Read-only Track A M37 + Track B zero-follow-up evaluation sensitivity",
        "refitted_models": 0,
        "new_bootstrap_draws": 0,
        "patient_identifiers_in_share_bundle": False,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_csv(out / "checks.csv", audit.checks)

    lines = [
        "# Paper 2B -- M55b Track A + zero-follow-up audit",
        "",
        f"**Status: {status}**",
        "",
        "## Scope",
        "",
        "- Read-only audit: no prognostic model refit and no new bootstrap.",
        "- Track A bootstrap summaries are recomputed from the saved M37 draws.",
        "- Track A calibration slopes and other secondary point metrics are checked against the saved M37 result record; they are not independently recalculated.",
        "- Zero-follow-up sensitivity removes time<=0 rows only when reevaluating saved OOF predictions; training is unchanged.",
        "",
        "## Track A",
        "",
    ]

    if track_a_full:
        lines += [
            "| Model | Harrell C | Point ΔC vs clinical | Calibration slope | IPCW AUC 5y | IBS 1-10y |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(
            track_a_full,
            key=lambda x: MODEL_ORDER.index(x["model_set"]) if x["model_set"] in MODEL_ORDER else 99,
        ):
            lines.append(
                f"| {r['model_set']} | {r.get('harrell_c_index', math.nan):.6f} | "
                f"{r.get('point_delta_c_vs_clinical', math.nan):+.6f} | "
                f"{r.get('calibration_slope', math.nan):.6f} | "
                f"{r.get('ipcw_auc_5y', math.nan):.6f} | "
                f"{r.get('integrated_brier_1_to_10y', math.nan):.6f} |"
            )
        lines.append("")

    track_a_delta = [r for r in track_a_boot if r.get("metric") == "delta_c_index_vs_clinical"]
    if track_a_delta:
        lines += [
            "### Recomputed from saved 1000 bootstrap draws",
            "",
            "| Model | Mean ΔC | 95% interval |",
            "|---|---:|---:|",
        ]
        for r in sorted(
            track_a_delta,
            key=lambda x: MODEL_ORDER.index(x["model_set"]) if x["model_set"] in MODEL_ORDER else 99,
        ):
            lines.append(
                f"| {r['model_set']} | {r['mean']:+.8f} | "
                f"[{r['ci_low']:+.8f}, {r['ci_high']:+.8f}] |"
            )
        lines.append("")

    lines += [
        "## Zero-follow-up audit",
        "",
        "| Analysis | Modality | Unique time<=0 patients | Event=0 rows | Event=1 rows | Hashed token |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in zero_summary:
        lines.append(
            f"| {r['analysis']} | {r['modality']} | {r['zero_or_negative_unique_patients']} | "
            f"{r['zero_event_0_rows']} | {r['zero_event_1_rows']} | "
            f"{r['hashed_patient_tokens'] or '-'} |"
        )
    lines.append("")

    if zero_sens:
        lines += [
            "### Evaluation-only sensitivity",
            "",
            "| Modality | Original ΔC | Excluding time<=0 | Change |",
            "|---|---:|---:|---:|",
        ]
        for r in zero_sens:
            lines.append(
                f"| {r['modality']} | {r['original_oof_mean_delta_c']:+.8f} | "
                f"{r['excluding_time_le_0_oof_mean_delta_c']:+.8f} | "
                f"{r['signed_change']:+.8f} |"
            )
        lines.append("")

    lines += [
        "## Interpretation boundary",
        "",
        "- A matching saved calibration slope is not an independent validation of calibration; M37 did not save the patient-level Track A predictions required for a no-refit recalculation.",
        "- Excluding a zero-time row only at evaluation does not reproduce what would happen if model development were repeated without that patient.",
        "- This audit does not cover RFS, multiplicity, panel provenance, or corrected Track B IPCW AUC.",
        "",
        "## Check counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    lines += ["", f"Elapsed: {elapsed:.2f} seconds.", ""]
    (out / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def make_share_zip(out: Path) -> Path:
    share_names = [
        "audit_report.md",
        "audit_summary.json",
        "checks.csv",
        "trackA_full_results_audit.csv",
        "trackA_bootstrap_recomputed.csv",
        "trackA_saved_summary_comparison.csv",
        "trackA_manuscript_reference_check.csv",
        "trackA_model_registry_audit.csv",
        "zero_followup_summary.csv",
        "zero_followup_point_sensitivity.csv",
        "input_manifest.csv",
        "console.log",
    ]
    zip_path = out / "share_bundle_m55b.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in share_names:
            p = out / name
            if p.is_file():
                z.write(p, arcname=name)
    return zip_path


def main() -> int:
    args = parse_args()
    root = find_root(args.root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_parent = (
        args.report_root.resolve()
        if args.report_root is not None
        else root / "results" / "reports"
    )
    out = report_parent / f"paper2b_m55b_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    log_path = out / "console.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
            return len(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    with log_path.open("w", encoding="utf-8") as log:
        sys.stdout = Tee(original_stdout, log)
        sys.stderr = Tee(original_stderr, log)
        try:
            started = time.time()
            print("PAPER 2B / M55b / TRACK A + ZERO-FOLLOW-UP READ-ONLY AUDIT")
            print(f"Version: {SCRIPT_VERSION}")
            print(f"Root: {root}")
            print(f"Output: {out}")
            print("No model refit; no new bootstrap; no input file modification.")

            audit = Audit()
            file_manifest(root, out)

            track_a_full, track_a_boot, _ = audit_track_a(root, out, audit)
            zero_summary, zero_sens = audit_zero_followup(root, out, audit)

            print("\n[3/3] Writing report")
            elapsed = time.time() - started
            summary = write_report(
                out,
                audit,
                track_a_full,
                track_a_boot,
                zero_summary,
                zero_sens,
                elapsed,
            )
            zip_path = make_share_zip(out)

            counts = audit.counts()
            print("\n" + "=" * 78)
            print(f"AUDIT STATUS: {summary['status']}")
            print("; ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "No checks recorded")
            print("No models fitted; no bootstrap rerun; no existing input changed.")
            print(f"Report: {out / 'audit_report.md'}")
            print(f"Share:  {zip_path}")

            return 0 if summary["status"] != "NOT_COMPLETE" else 2
        except Exception as exc:
            print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
