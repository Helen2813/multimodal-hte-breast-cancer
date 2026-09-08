#!/usr/bin/env python3
"""
M56 -- Paper 2B read-only audit:
  A) RFS artifacts, point estimates, saved-bootstrap summaries, and manuscript references
  B) Proper censoring-aware 5-year cumulative/dynamic AUC for existing OS OOF predictions
  C) An inference registry for later multiplicity work (NO multiplicity correction is applied)

This script does NOT refit any prognostic model and does NOT rerun any bootstrap.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe .\\scripts\\m56_audit_rfs_and_ipcw_auc.py

Optional dependency for censoring-aware AUC:
    scikit-survival

If scikit-survival is absent, the RFS audit still runs and the IPCW-AUC section is
reported as NOT_VERIFIED. The script never installs packages automatically.

Important boundaries
--------------------
1. RFS is treated as a sensitivity endpoint. A positive RNA RFS result is not
   promoted to a confirmatory finding here.
2. No multiplicity family or correction is chosen by this script.
3. Saved bootstrap draws are summarized; model development is not repeated.
4. The legacy Track-B "5-year AUC" is the project's complete-case binary AUC:
   cases = event by 60 months; controls = observed beyond 60 months; patients
   censored before 60 months are excluded.
5. The new IPCW AUC uses cumulative_dynamic_auc within each held-out fold.
   The censoring distribution is estimated from the complement of that fold
   within the same repeat. Fold AUCs are test-n weighted within repeat, then
   repeat estimates are averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"
HORIZON_MONTHS = 60.0

# Rounded historical manuscript/Supplement values. These are comparison targets,
# not authoritative recalculations.
RFS_MANUSCRIPT_REFERENCES = {
    "RNA": {"delta_c_index": 0.0146, "ci_low": 0.0035, "ci_high": 0.0258},
    "Methylation": {"delta_c_index": -0.0170, "ci_low": -0.0319, "ci_high": -0.0017},
    "CNV": {"delta_c_index": 0.0059, "ci_low": -0.0026, "ci_high": 0.0144},
    "Mutation": {"delta_c_index": -0.0040, "ci_low": -0.0107, "ci_high": 0.0030},
    "Multimodal": {"delta_c_index": 0.0029, "ci_low": -0.0136, "ci_high": 0.0204},
}

MODALITY_CANON = {
    "rna": "RNA",
    "cnv": "CNV",
    "cna": "CNV",
    "copy_number": "CNV",
    "copynumber": "CNV",
    "methylation": "Methylation",
    "meth": "Methylation",
    "mutation": "Mutation",
    "mut": "Mutation",
    "multimodal": "Multimodal",
    "combined": "Multimodal",
}

PATIENT_ALIASES = ["sample_id", "patient_id", "case_id", "submitter_id"]
TIME_ALIASES = [
    "time_months",
    "rfs_months",
    "rfs_time_months",
    "rfs_time",
    "time",
    "duration",
]
EVENT_ALIASES = ["event", "rfs_event", "event_rfs", "status"]
CLINICAL_RISK_ALIASES = ["clinical_risk", "risk_clinical", "clinical_score"]
MODEL_RISK_ALIASES = [
    "clinical_modality_risk",
    "model_risk",
    "combined_risk",
    "augmented_risk",
    "omics_risk",
]
DELTA_C_ALIASES = [
    "delta_c_index",
    "delta_c_index_vs_clinical",
    "delta_c",
    "delta_harrell_c",
]
REPETITION_ALIASES = ["repetition", "bootstrap", "replicate", "draw", "iteration"]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper 2B M56 RFS + IPCW-AUC read-only audit.")
    p.add_argument("--root", type=Path, default=None, help="Repository root; default auto-detect.")
    p.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Parent for report folders; default results/reports.",
    )
    return p.parse_args()


def find_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "metabric_m7_config.json").is_file():
            raise FileNotFoundError(f"Not a recognized project root: {root}")
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
    raise FileNotFoundError("Could not auto-detect the multimodal-hte-breast-cancer root.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def save_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def finite(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    return a[np.isfinite(a)]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
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


def choose_alias(columns: Iterable[str], aliases: list[str]) -> Optional[str]:
    cols = list(map(str, columns))
    exact = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias.lower() in exact:
            return exact[alias.lower()]
    return None


def canonical_modality(value: Any, fallback_path: str = "") -> str:
    text = str(value or "").strip()
    low = text.lower()
    for key, val in MODALITY_CANON.items():
        if low == key or key in low:
            return val
    path_low = fallback_path.lower()
    for key, val in MODALITY_CANON.items():
        if re.search(rf"(^|[^a-z]){re.escape(key)}([^a-z]|$)", path_low):
            return val
    return text if text else "Unknown"


def cindex(time_values: np.ndarray, event_values: np.ndarray, risk_values: np.ndarray) -> float:
    t = np.asarray(time_values, dtype=float)
    e = np.asarray(event_values, dtype=int)
    r = np.asarray(risk_values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    if mask.sum() < 3 or int(e[mask].sum()) == 0:
        return float("nan")

    try:
        from sksurv.metrics import concordance_index_censored
        return float(
            concordance_index_censored(
                e[mask].astype(bool),
                t[mask].astype(float),
                r[mask].astype(float),
            )[0]
        )
    except Exception:
        try:
            from lifelines.utils import concordance_index
            return float(
                concordance_index(
                    t[mask].astype(float),
                    -r[mask].astype(float),
                    event_observed=e[mask].astype(int),
                )
            )
        except Exception as exc:
            raise RuntimeError("Need either scikit-survival or lifelines for Harrell C.") from exc


def legacy_binary_auc(
    time_values: np.ndarray,
    event_values: np.ndarray,
    risk_values: np.ndarray,
    horizon: float,
) -> tuple[float, int]:
    from sklearn.metrics import roc_auc_score

    t = np.asarray(time_values, dtype=float)
    e = np.asarray(event_values, dtype=int)
    r = np.asarray(risk_values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[mask], e[mask], r[mask]

    cases = (e == 1) & (t <= horizon)
    controls = t > horizon
    valid = cases | controls
    if valid.sum() < 10 or cases.sum() == 0 or controls.sum() == 0:
        return float("nan"), int(valid.sum())
    return float(roc_auc_score(cases[valid].astype(int), r[valid])), int(valid.sum())


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, status: str, check: str, details: str, required: bool = True) -> None:
        self.checks.append(
            {
                "status": status,
                "check": check,
                "details": details,
                "required": bool(required),
            }
        )

    def counts(self) -> Counter:
        return Counter(r["status"] for r in self.checks)

    def overall(self) -> str:
        c = self.counts()
        if c.get("FAIL", 0) or c.get("NOT_VERIFIED", 0):
            return "NOT_COMPLETE"
        if c.get("REVIEW", 0):
            return "REVIEW_NEEDED"
        return "CHECKED_ARTIFACTS_CONSISTENT"


# ---------------------------------------------------------------------------
# RFS discovery
# ---------------------------------------------------------------------------

def header_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def rfs_candidate_files(root: Path) -> list[Path]:
    results = root / "results"
    if not results.exists():
        return []
    candidates: list[Path] = []
    for p in results.rglob("*.csv"):
        low = str(p.relative_to(root)).lower()
        if any(tag in low for tag in ["rfs", "recurrence", "m54"]):
            candidates.append(p)
    return sorted(set(candidates))


def classify_rfs_candidate(path: Path, columns: list[str]) -> tuple[str, int]:
    low_path = str(path).lower()
    lower = {c.lower() for c in columns}

    patient = choose_alias(columns, PATIENT_ALIASES)
    time_col = choose_alias(columns, TIME_ALIASES)
    event_col = choose_alias(columns, EVENT_ALIASES)
    clinical = choose_alias(columns, CLINICAL_RISK_ALIASES)
    model = choose_alias(columns, MODEL_RISK_ALIASES)
    repeat = "repeat" if "repeat" in lower else None
    fold = "fold" if "fold" in lower else None
    delta = choose_alias(columns, DELTA_C_ALIASES)
    rep = choose_alias(columns, REPETITION_ALIASES)

    score_oof = sum(
        [
            4 if patient else 0,
            4 if time_col else 0,
            4 if event_col else 0,
            4 if clinical else 0,
            4 if model else 0,
            3 if repeat else 0,
            2 if fold else 0,
            2 if ("oof" in low_path or "prediction" in low_path) else 0,
        ]
    )
    score_boot = sum(
        [
            6 if delta else 0,
            4 if rep else 0,
            3 if ("bootstrap" in low_path or "draw" in low_path) else 0,
            2 if "m54" in low_path else 0,
        ]
    )

    if score_oof >= 20 and score_oof > score_boot:
        return "OOF_CANDIDATE", score_oof
    if score_boot >= 9:
        return "BOOTSTRAP_CANDIDATE", score_boot
    if "summary" in low_path:
        return "SUMMARY_OR_OTHER", max(score_oof, score_boot)
    return "OTHER_RFS", max(score_oof, score_boot)


def inventory_rfs(root: Path, out: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in rfs_candidate_files(root):
        cols = header_columns(path)
        role, score = classify_rfs_candidate(path, cols)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "columns": "|".join(cols),
                "candidate_role": role,
                "role_score": score,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "rfs_artifact_inventory.csv", index=False)
    return df


def infer_oof_schema(frame: pd.DataFrame) -> dict[str, Optional[str]]:
    return {
        "patient": choose_alias(frame.columns, PATIENT_ALIASES),
        "time": choose_alias(frame.columns, TIME_ALIASES),
        "event": choose_alias(frame.columns, EVENT_ALIASES),
        "clinical_risk": choose_alias(frame.columns, CLINICAL_RISK_ALIASES),
        "model_risk": choose_alias(frame.columns, MODEL_RISK_ALIASES),
        "repeat": "repeat" if "repeat" in frame.columns else None,
        "fold": "fold" if "fold" in frame.columns else None,
        "modality": "modality" if "modality" in frame.columns else None,
        "analysis": "analysis" if "analysis" in frame.columns else None,
    }


def validate_oof_schema(schema: dict[str, Optional[str]]) -> bool:
    return all(
        schema[k] is not None
        for k in ["patient", "time", "event", "clinical_risk", "model_risk", "repeat", "fold"]
    )


def repeat_pooled_delta(
    frame: pd.DataFrame,
    schema: dict[str, str],
) -> tuple[float, float]:
    pooled: list[float] = []
    within: list[float] = []
    for _, rep in frame.groupby(schema["repeat"], sort=True):
        c = cindex(
            num(rep[schema["time"]]).to_numpy(float),
            num(rep[schema["event"]]).fillna(0).to_numpy(int),
            num(rep[schema["clinical_risk"]]).to_numpy(float),
        )
        m = cindex(
            num(rep[schema["time"]]).to_numpy(float),
            num(rep[schema["event"]]).fillna(0).to_numpy(int),
            num(rep[schema["model_risk"]]).to_numpy(float),
        )
        pooled.append(m - c)

        fold_vals: list[float] = []
        fold_w: list[int] = []
        for _, fold in rep.groupby(schema["fold"], sort=True):
            cf = cindex(
                num(fold[schema["time"]]).to_numpy(float),
                num(fold[schema["event"]]).fillna(0).to_numpy(int),
                num(fold[schema["clinical_risk"]]).to_numpy(float),
            )
            mf = cindex(
                num(fold[schema["time"]]).to_numpy(float),
                num(fold[schema["event"]]).fillna(0).to_numpy(int),
                num(fold[schema["model_risk"]]).to_numpy(float),
            )
            if np.isfinite(cf) and np.isfinite(mf):
                fold_vals.append(mf - cf)
                fold_w.append(len(fold))
        within.append(
            float(np.average(fold_vals, weights=fold_w))
            if fold_vals
            else float("nan")
        )
    return float(np.nanmean(pooled)), float(np.nanmean(within))


def audit_rfs_oof(
    root: Path,
    inventory: pd.DataFrame,
    out: Path,
    audit: Audit,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames_by_key: dict[str, pd.DataFrame] = {}
    if inventory.empty:
        audit.add("NOT_VERIFIED", "RFS artifact discovery", "No RFS/m54/recurrence CSV found.", True)
        return pd.DataFrame(), frames_by_key

    oof_paths = inventory.loc[inventory["candidate_role"] == "OOF_CANDIDATE", "path"].tolist()
    if not oof_paths:
        audit.add(
            "NOT_VERIFIED",
            "RFS OOF discovery",
            "No file satisfied the OOF schema among RFS/m54/recurrence artifacts.",
            True,
        )
        return pd.DataFrame(), frames_by_key

    candidate_results: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_frames: defaultdict[str, list[pd.DataFrame]] = defaultdict(list)

    for rel in oof_paths:
        path = root / rel
        frame = read_csv(path)
        schema0 = infer_oof_schema(frame)
        if not validate_oof_schema(schema0):
            audit.add("REVIEW", f"RFS OOF schema: {rel}", f"Detected={schema0}", False)
            continue
        schema = {k: v for k, v in schema0.items() if v is not None}

        if schema0["modality"]:
            groups = list(frame.groupby(schema["modality"], sort=True))
        else:
            groups = [(canonical_modality("", rel), frame)]

        for raw_modality, sub in groups:
            modality = canonical_modality(raw_modality, rel)
            if modality == "Unknown":
                modality = canonical_modality("", rel)

            s = sub.copy()
            pid = schema["patient"]
            repeat = schema["repeat"]
            fold = schema["fold"]
            tcol = schema["time"]
            ecol = schema["event"]

            s[pid] = s[pid].astype(str)
            unique_n = int(s[pid].nunique())
            events_by_patient = (
                s.groupby(pid, observed=True)[ecol]
                .first()
                .pipe(pd.to_numeric, errors="coerce")
            )
            events = int(np.nansum(events_by_patient.to_numpy(float)))
            repeats = int(pd.to_numeric(s[repeat], errors="coerce").nunique())
            folds = int(pd.to_numeric(s[fold], errors="coerce").nunique())
            dup = int(s.duplicated([repeat, pid]).sum())
            time_nonpositive = int((num(s[tcol]) <= 0).sum())

            outcome_consistency = (
                s.groupby(pid, observed=True)
                .agg(
                    time_nunique=(tcol, lambda x: pd.to_numeric(x, errors="coerce").nunique(dropna=True)),
                    event_nunique=(ecol, lambda x: pd.to_numeric(x, errors="coerce").nunique(dropna=True)),
                )
            )
            inconsistent_outcomes = int(
                ((outcome_consistency["time_nunique"] > 1) | (outcome_consistency["event_nunique"] > 1)).sum()
            )

            try:
                pooled, within = repeat_pooled_delta(s, schema)
                metric_status = "PASS"
                metric_error = ""
            except Exception as exc:
                pooled = within = math.nan
                metric_status = "NOT_VERIFIED"
                metric_error = f"{type(exc).__name__}: {exc}"

            rec = {
                "source_path": rel,
                "modality": modality,
                "n_unique_patients": unique_n,
                "events": events,
                "repeats": repeats,
                "folds_observed": folds,
                "duplicate_repeat_patient_rows": dup,
                "inconsistent_patient_outcomes_across_repeats": inconsistent_outcomes,
                "time_le_0_rows": time_nonpositive,
                "pooled_oof_mean_delta_c": pooled,
                "within_fold_weighted_mean_delta_c": within,
                "metric_status": metric_status,
                "metric_error": metric_error,
            }
            candidate_results[modality].append(rec)
            temp = s.copy()
            temp.attrs["schema"] = schema
            temp.attrs["source_path"] = rel
            temp.attrs["modality"] = modality
            candidate_frames[modality].append(temp)

    # Resolve duplicate sources only if numerically consistent.
    chosen_rows: list[dict[str, Any]] = []
    for modality, recs in sorted(candidate_results.items()):
        if len(recs) == 1:
            chosen = recs[0]
            chosen["source_resolution"] = "UNIQUE"
            chosen_rows.append(chosen)
            frames_by_key[modality] = candidate_frames[modality][0]
            continue

        values = [r["pooled_oof_mean_delta_c"] for r in recs if np.isfinite(r["pooled_oof_mean_delta_c"])]
        ns = {r["n_unique_patients"] for r in recs}
        reps = {r["repeats"] for r in recs}
        consistent = (
            len(values) == len(recs)
            and (max(values) - min(values) <= 1e-12)
            and len(ns) == 1
            and len(reps) == 1
        )
        if consistent:
            idx = sorted(range(len(recs)), key=lambda i: recs[i]["source_path"])[0]
            chosen = recs[idx].copy()
            chosen["source_resolution"] = f"REDUNDANT_CONSISTENT_{len(recs)}"
            chosen_rows.append(chosen)
            frames_by_key[modality] = candidate_frames[modality][idx]
            audit.add(
                "PASS",
                f"RFS OOF source resolution: {modality}",
                f"{len(recs)} candidate files gave identical n/repeats/point estimate; using {chosen['source_path']}.",
                True,
            )
        else:
            for r in recs:
                r2 = r.copy()
                r2["source_resolution"] = "AMBIGUOUS_NOT_CHOSEN"
                rows.append(r2)
            audit.add(
                "NOT_VERIFIED",
                f"RFS OOF source resolution: {modality}",
                f"Multiple non-identical candidate files: {[r['source_path'] for r in recs]}",
                True,
            )

    rows.extend(chosen_rows)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rfs_oof_audit.csv", index=False)

    for r in chosen_rows:
        status = "PASS"
        details = (
            f"n={r['n_unique_patients']}, events={r['events']}, repeats={r['repeats']}, "
            f"pooled ΔC={r['pooled_oof_mean_delta_c']:+.8f}, "
            f"within-fold ΔC={r['within_fold_weighted_mean_delta_c']:+.8f}; "
            f"source={r['source_path']}"
        )
        if r["duplicate_repeat_patient_rows"] or r["inconsistent_patient_outcomes_across_repeats"]:
            status = "REVIEW"
        audit.add(status, f"RFS OOF audit: {r['modality']}", details, True)
        print(f"  RFS {r['modality']}: {details}")

    return df, frames_by_key


# ---------------------------------------------------------------------------
# RFS saved-bootstrap audit
# ---------------------------------------------------------------------------

def parse_bootstrap_file(path: Path) -> list[dict[str, Any]]:
    frame = read_csv(path)
    delta_col = choose_alias(frame.columns, DELTA_C_ALIASES)
    rep_col = choose_alias(frame.columns, REPETITION_ALIASES)
    if delta_col is None or rep_col is None:
        return []

    modality_col = "modality" if "modality" in frame.columns else None
    analysis_col = "analysis" if "analysis" in frame.columns else None

    if modality_col:
        groups = list(frame.groupby(modality_col, sort=True))
    elif analysis_col and frame[analysis_col].astype(str).nunique() > 1:
        groups = list(frame.groupby(analysis_col, sort=True))
    else:
        groups = [(canonical_modality("", str(path)), frame)]

    out: list[dict[str, Any]] = []
    for raw_modality, sub in groups:
        modality = canonical_modality(raw_modality, str(path))
        if modality == "Unknown":
            modality = canonical_modality("", str(path))
        stats = summarize(num(sub[delta_col]).to_numpy())
        out.append(
            {
                "source_path": str(path),
                "modality": modality,
                "delta_column": delta_col,
                "repetition_column": rep_col,
                "unique_repetitions": int(pd.to_numeric(sub[rep_col], errors="coerce").nunique()),
                **stats,
            }
        )
    return out


def audit_rfs_bootstrap(
    root: Path,
    inventory: pd.DataFrame,
    out: Path,
    audit: Audit,
) -> pd.DataFrame:
    candidates = (
        inventory.loc[inventory["candidate_role"] == "BOOTSTRAP_CANDIDATE", "path"].tolist()
        if not inventory.empty
        else []
    )
    rows: list[dict[str, Any]] = []
    for rel in candidates:
        try:
            parsed = parse_bootstrap_file(root / rel)
            for r in parsed:
                r["source_path"] = rel
                rows.append(r)
        except Exception as exc:
            audit.add(
                "REVIEW",
                f"RFS bootstrap candidate: {rel}",
                f"{type(exc).__name__}: {exc}",
                False,
            )

    if not rows:
        audit.add(
            "NOT_VERIFIED",
            "RFS saved-bootstrap discovery",
            "No RFS candidate file exposed a repetition column plus a ΔC column.",
            True,
        )
        df = pd.DataFrame()
        df.to_csv(out / "rfs_bootstrap_recomputed.csv", index=False)
        return df

    # Resolve duplicate groups per modality if consistent.
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["modality"]].append(r)

    chosen: list[dict[str, Any]] = []
    for modality, recs in sorted(grouped.items()):
        if len(recs) == 1:
            r = recs[0].copy()
            r["source_resolution"] = "UNIQUE"
            chosen.append(r)
            continue
        means = [r["mean"] for r in recs if np.isfinite(r["mean"])]
        cis = [(r["ci_low"], r["ci_high"]) for r in recs if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"])]
        consistent = (
            len(means) == len(recs)
            and max(means) - min(means) <= 1e-12
            and len({(round(a, 12), round(b, 12)) for a, b in cis}) == 1
        )
        if consistent:
            idx = sorted(range(len(recs)), key=lambda i: recs[i]["source_path"])[0]
            r = recs[idx].copy()
            r["source_resolution"] = f"REDUNDANT_CONSISTENT_{len(recs)}"
            chosen.append(r)
            audit.add(
                "PASS",
                f"RFS bootstrap source resolution: {modality}",
                f"{len(recs)} files yielded identical summaries; using {r['source_path']}.",
                True,
            )
        else:
            audit.add(
                "NOT_VERIFIED",
                f"RFS bootstrap source resolution: {modality}",
                f"Non-identical candidate summaries from {[r['source_path'] for r in recs]}",
                True,
            )

    df = pd.DataFrame(chosen)
    df.to_csv(out / "rfs_bootstrap_recomputed.csv", index=False)

    for r in chosen:
        audit.add(
            "PASS",
            f"RFS saved bootstrap: {r['modality']}",
            (
                f"draws={r['unique_repetitions']}; mean ΔC={r['mean']:+.8f}; "
                f"95%=[{r['ci_low']:+.8f},{r['ci_high']:+.8f}]; source={r['source_path']}"
            ),
            True,
        )
        print(
            f"  RFS bootstrap {r['modality']}: mean ΔC={r['mean']:+.8f}; "
            f"95%=[{r['ci_low']:+.8f},{r['ci_high']:+.8f}] "
            f"({r['unique_repetitions']} saved draws)"
        )
    return df


def manuscript_rfs_check(
    oof: pd.DataFrame,
    boot: pd.DataFrame,
    out: Path,
    audit: Audit,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for modality, ref in RFS_MANUSCRIPT_REFERENCES.items():
        oo = (
            oof.loc[
                (oof.get("modality", pd.Series(dtype=str)).astype(str) == modality)
                & (oof.get("source_resolution", pd.Series(dtype=str)) != "AMBIGUOUS_NOT_CHOSEN")
            ]
            if not oof.empty and "modality" in oof.columns
            else pd.DataFrame()
        )
        bb = (
            boot.loc[boot["modality"].astype(str) == modality]
            if not boot.empty and "modality" in boot.columns
            else pd.DataFrame()
        )

        point = float(oo.iloc[0]["pooled_oof_mean_delta_c"]) if len(oo) == 1 else math.nan
        boot_mean = float(bb.iloc[0]["mean"]) if len(bb) == 1 else math.nan
        low = float(bb.iloc[0]["ci_low"]) if len(bb) == 1 else math.nan
        high = float(bb.iloc[0]["ci_high"]) if len(bb) == 1 else math.nan

        # Manuscript historically appears to use bootstrap mean as displayed ΔC.
        display_candidate = boot_mean if np.isfinite(boot_mean) else point
        diffs = {
            "delta": abs(display_candidate - ref["delta_c_index"]) if np.isfinite(display_candidate) else math.nan,
            "ci_low": abs(low - ref["ci_low"]) if np.isfinite(low) else math.nan,
            "ci_high": abs(high - ref["ci_high"]) if np.isfinite(high) else math.nan,
        }
        status = (
            "PASS"
            if all(np.isfinite(v) for v in diffs.values())
            and diffs["delta"] <= 0.00015
            and diffs["ci_low"] <= 0.00015
            and diffs["ci_high"] <= 0.00015
            else "NOT_VERIFIED"
            if not all(np.isfinite(v) for v in diffs.values())
            else "REVIEW"
        )
        row = {
            "modality": modality,
            "reference_delta_c": ref["delta_c_index"],
            "reference_ci_low": ref["ci_low"],
            "reference_ci_high": ref["ci_high"],
            "oof_direct_point_delta_c": point,
            "saved_bootstrap_mean_delta_c": boot_mean,
            "saved_bootstrap_ci_low": low,
            "saved_bootstrap_ci_high": high,
            "status": status,
            "absolute_diff_display_delta": diffs["delta"],
            "absolute_diff_ci_low": diffs["ci_low"],
            "absolute_diff_ci_high": diffs["ci_high"],
        }
        rows.append(row)
        audit.add(
            status,
            f"RFS manuscript reference: {modality}",
            (
                f"reference={ref}; OOF point={point}; bootstrap mean/CI="
                f"{boot_mean}, [{low},{high}]"
            ),
            True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "rfs_manuscript_reference_check.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# OS OOF and censoring-aware AUC
# ---------------------------------------------------------------------------

def os_oof_sources(root: Path) -> dict[str, Path]:
    return {
        "Multimodal": root / "results" / "tables" / "metabric_m7" / "m38_oof_predictions_LOCAL_ONLY.csv",
        "ModalitySpecific": root / "results" / "tables" / "metabric_m8" / "m41_oof_predictions_LOCAL_ONLY.csv",
    }


def make_surv(event: np.ndarray, time_values: np.ndarray):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(event, dtype=bool), time=np.asarray(time_values, dtype=float))


def fold_ipcw_auc(
    rep: pd.DataFrame,
    schema: dict[str, str],
    horizon: float,
    remove_nonpositive: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    from sksurv.metrics import cumulative_dynamic_auc

    details: list[dict[str, Any]] = []
    errors: list[str] = []
    fold_col = schema["fold"]
    all_folds = sorted(pd.to_numeric(rep[fold_col], errors="coerce").dropna().unique())

    for fold_value in all_folds:
        test = rep.loc[pd.to_numeric(rep[fold_col], errors="coerce") == fold_value].copy()
        train = rep.loc[pd.to_numeric(rep[fold_col], errors="coerce") != fold_value].copy()

        if remove_nonpositive:
            test = test.loc[num(test[schema["time"]]) > 0].copy()
            train = train.loc[num(train[schema["time"]]) > 0].copy()

        def valid_frame(df: pd.DataFrame, risk_col: str) -> pd.DataFrame:
            mask = (
                num(df[schema["time"]]).notna()
                & num(df[schema["event"]]).notna()
                & num(df[risk_col]).notna()
            )
            return df.loc[mask].copy()

        test_both = test.loc[
            num(test[schema["time"]]).notna()
            & num(test[schema["event"]]).notna()
            & num(test[schema["clinical_risk"]]).notna()
            & num(test[schema["model_risk"]]).notna()
        ].copy()
        train_surv = train.loc[
            num(train[schema["time"]]).notna()
            & num(train[schema["event"]]).notna()
        ].copy()

        if len(test_both) < 10 or len(train_surv) < 20:
            errors.append(f"fold={fold_value}: insufficient train/test size")
            continue

        y_train = make_surv(
            num(train_surv[schema["event"]]).fillna(0).to_numpy(int),
            num(train_surv[schema["time"]]).to_numpy(float),
        )
        y_test = make_surv(
            num(test_both[schema["event"]]).fillna(0).to_numpy(int),
            num(test_both[schema["time"]]).to_numpy(float),
        )

        try:
            auc_c, _ = cumulative_dynamic_auc(
                y_train,
                y_test,
                num(test_both[schema["clinical_risk"]]).to_numpy(float),
                np.asarray([horizon], dtype=float),
            )
            auc_m, _ = cumulative_dynamic_auc(
                y_train,
                y_test,
                num(test_both[schema["model_risk"]]).to_numpy(float),
                np.asarray([horizon], dtype=float),
            )
            clinical_ipcw = float(np.asarray(auc_c).ravel()[0])
            model_ipcw = float(np.asarray(auc_m).ravel()[0])
        except Exception as exc:
            errors.append(f"fold={fold_value}: {type(exc).__name__}: {exc}")
            continue

        legacy_c, legacy_n_c = legacy_binary_auc(
            num(test_both[schema["time"]]).to_numpy(float),
            num(test_both[schema["event"]]).fillna(0).to_numpy(int),
            num(test_both[schema["clinical_risk"]]).to_numpy(float),
            horizon,
        )
        legacy_m, legacy_n_m = legacy_binary_auc(
            num(test_both[schema["time"]]).to_numpy(float),
            num(test_both[schema["event"]]).fillna(0).to_numpy(int),
            num(test_both[schema["model_risk"]]).to_numpy(float),
            horizon,
        )

        details.append(
            {
                "fold": int(fold_value),
                "test_n": int(len(test_both)),
                "train_n": int(len(train_surv)),
                "ipcw_auc_clinical": clinical_ipcw,
                "ipcw_auc_model": model_ipcw,
                "ipcw_delta_auc": model_ipcw - clinical_ipcw,
                "legacy_binary_auc_clinical": legacy_c,
                "legacy_binary_auc_model": legacy_m,
                "legacy_delta_auc": legacy_m - legacy_c if np.isfinite(legacy_c) and np.isfinite(legacy_m) else math.nan,
                "legacy_valid_n_clinical": legacy_n_c,
                "legacy_valid_n_model": legacy_n_m,
                "remove_nonpositive_time": int(remove_nonpositive),
            }
        )
    return details, errors


def pooled_legacy_repeat_auc(
    rep: pd.DataFrame,
    schema: dict[str, str],
    horizon: float,
) -> dict[str, float]:
    c, nc = legacy_binary_auc(
        num(rep[schema["time"]]).to_numpy(float),
        num(rep[schema["event"]]).fillna(0).to_numpy(int),
        num(rep[schema["clinical_risk"]]).to_numpy(float),
        horizon,
    )
    m, nm = legacy_binary_auc(
        num(rep[schema["time"]]).to_numpy(float),
        num(rep[schema["event"]]).fillna(0).to_numpy(int),
        num(rep[schema["model_risk"]]).to_numpy(float),
        horizon,
    )
    return {
        "clinical": c,
        "model": m,
        "delta": m - c if np.isfinite(c) and np.isfinite(m) else math.nan,
        "valid_n": min(nc, nm),
    }


def evaluate_auc_dataset(
    frame: pd.DataFrame,
    modality: str,
    endpoint: str,
    source_path: str,
    out_details: list[dict[str, Any]],
    audit: Audit,
    sksurv_available: bool,
) -> dict[str, Any]:
    schema0 = infer_oof_schema(frame)
    if not validate_oof_schema(schema0):
        audit.add(
            "NOT_VERIFIED",
            f"{endpoint} AUC schema: {modality}",
            f"Detected={schema0}; source={source_path}",
            True,
        )
        return {}
    schema = {k: v for k, v in schema0.items() if v is not None}

    repeat_summaries: list[dict[str, Any]] = []
    errors_all: list[str] = []

    for repeat_value, rep in frame.groupby(schema["repeat"], sort=True):
        legacy_pooled = pooled_legacy_repeat_auc(rep, schema, HORIZON_MONTHS)

        if sksurv_available:
            details, errors = fold_ipcw_auc(rep, schema, HORIZON_MONTHS, remove_nonpositive=False)
            # If some/all folds fail and there are nonpositive times, also try explicit sensitivity.
            positive_details: list[dict[str, Any]] = []
            positive_errors: list[str] = []
            if errors and int((num(rep[schema["time"]]) <= 0).sum()) > 0:
                positive_details, positive_errors = fold_ipcw_auc(
                    rep,
                    schema,
                    HORIZON_MONTHS,
                    remove_nonpositive=True,
                )

            for d in details:
                d.update(
                    {
                        "endpoint": endpoint,
                        "modality": modality,
                        "source_path": source_path,
                        "repeat": int(repeat_value),
                    }
                )
                out_details.append(d)
            for d in positive_details:
                d.update(
                    {
                        "endpoint": endpoint,
                        "modality": modality,
                        "source_path": source_path,
                        "repeat": int(repeat_value),
                    }
                )
                out_details.append(d)

            chosen = details
            sensitivity_used = 0
            if not chosen and positive_details:
                chosen = positive_details
                sensitivity_used = 1

            if chosen:
                weights = np.asarray([d["test_n"] for d in chosen], dtype=float)
                ipcw_c = float(np.average([d["ipcw_auc_clinical"] for d in chosen], weights=weights))
                ipcw_m = float(np.average([d["ipcw_auc_model"] for d in chosen], weights=weights))
                ipcw_delta = ipcw_m - ipcw_c
                legacy_fold_c = float(
                    np.average(
                        [d["legacy_binary_auc_clinical"] for d in chosen if np.isfinite(d["legacy_binary_auc_clinical"])],
                        weights=[
                            d["test_n"]
                            for d in chosen
                            if np.isfinite(d["legacy_binary_auc_clinical"])
                        ],
                    )
                )
                legacy_fold_m = float(
                    np.average(
                        [d["legacy_binary_auc_model"] for d in chosen if np.isfinite(d["legacy_binary_auc_model"])],
                        weights=[
                            d["test_n"]
                            for d in chosen
                            if np.isfinite(d["legacy_binary_auc_model"])
                        ],
                    )
                )
                repeat_summaries.append(
                    {
                        "repeat": int(repeat_value),
                        "ipcw_clinical": ipcw_c,
                        "ipcw_model": ipcw_m,
                        "ipcw_delta": ipcw_delta,
                        "legacy_fold_clinical": legacy_fold_c,
                        "legacy_fold_model": legacy_fold_m,
                        "legacy_fold_delta": legacy_fold_m - legacy_fold_c,
                        "legacy_pooled_clinical": legacy_pooled["clinical"],
                        "legacy_pooled_model": legacy_pooled["model"],
                        "legacy_pooled_delta": legacy_pooled["delta"],
                        "successful_ipcw_folds": len(chosen),
                        "positive_time_sensitivity_used": sensitivity_used,
                    }
                )
            errors_all.extend([f"repeat={repeat_value}: {x}" for x in errors])
            if positive_errors:
                errors_all.extend([f"repeat={repeat_value} positive-time sensitivity: {x}" for x in positive_errors])
        else:
            repeat_summaries.append(
                {
                    "repeat": int(repeat_value),
                    "ipcw_clinical": math.nan,
                    "ipcw_model": math.nan,
                    "ipcw_delta": math.nan,
                    "legacy_fold_clinical": math.nan,
                    "legacy_fold_model": math.nan,
                    "legacy_fold_delta": math.nan,
                    "legacy_pooled_clinical": legacy_pooled["clinical"],
                    "legacy_pooled_model": legacy_pooled["model"],
                    "legacy_pooled_delta": legacy_pooled["delta"],
                    "successful_ipcw_folds": 0,
                    "positive_time_sensitivity_used": 0,
                }
            )

    if not repeat_summaries:
        audit.add(
            "NOT_VERIFIED",
            f"{endpoint} AUC evaluation: {modality}",
            f"No repeat summary could be calculated; source={source_path}; errors={errors_all[:3]}",
            True,
        )
        return {}

    repdf = pd.DataFrame(repeat_summaries)
    result = {
        "endpoint": endpoint,
        "modality": modality,
        "source_path": source_path,
        "repeats": int(len(repdf)),
        "horizon_months": HORIZON_MONTHS,
        "legacy_pooled_binary_auc_clinical_mean": float(repdf["legacy_pooled_clinical"].mean()),
        "legacy_pooled_binary_auc_model_mean": float(repdf["legacy_pooled_model"].mean()),
        "legacy_pooled_binary_delta_auc_mean": float(repdf["legacy_pooled_delta"].mean()),
        "legacy_foldweighted_binary_auc_clinical_mean": float(repdf["legacy_fold_clinical"].mean())
        if repdf["legacy_fold_clinical"].notna().any()
        else math.nan,
        "legacy_foldweighted_binary_auc_model_mean": float(repdf["legacy_fold_model"].mean())
        if repdf["legacy_fold_model"].notna().any()
        else math.nan,
        "legacy_foldweighted_binary_delta_auc_mean": float(repdf["legacy_fold_delta"].mean())
        if repdf["legacy_fold_delta"].notna().any()
        else math.nan,
        "ipcw_auc_clinical_mean": float(repdf["ipcw_clinical"].mean())
        if repdf["ipcw_clinical"].notna().any()
        else math.nan,
        "ipcw_auc_model_mean": float(repdf["ipcw_model"].mean())
        if repdf["ipcw_model"].notna().any()
        else math.nan,
        "ipcw_delta_auc_mean": float(repdf["ipcw_delta"].mean())
        if repdf["ipcw_delta"].notna().any()
        else math.nan,
        "successful_ipcw_fold_evaluations": int(repdf["successful_ipcw_folds"].sum()),
        "repeats_using_positive_time_sensitivity": int(repdf["positive_time_sensitivity_used"].sum()),
        "ipcw_error_count": len(errors_all),
        "ipcw_error_examples": " || ".join(errors_all[:5]),
        "ipcw_status": (
            "PASS"
            if sksurv_available and repdf["ipcw_clinical"].notna().all()
            else "REVIEW"
            if sksurv_available and repdf["ipcw_clinical"].notna().any()
            else "NOT_VERIFIED"
        ),
        "note": (
            "IPCW AUC is fold-contained cumulative/dynamic AUC; censoring distribution estimated "
            "from complement folds within the same repeat. Legacy binary AUC excludes censoring "
            "before horizon and is shown only for comparison."
        ),
    }

    audit.add(
        result["ipcw_status"],
        f"{endpoint} censoring-aware AUC: {modality}",
        (
            f"legacy pooled ΔAUC={result['legacy_pooled_binary_delta_auc_mean']}; "
            f"IPCW fold-contained ΔAUC={result['ipcw_delta_auc_mean']}; "
            f"successful fold evals={result['successful_ipcw_fold_evaluations']}; "
            f"errors={result['ipcw_error_count']}"
        ),
        endpoint == "OS",
    )
    return result


def audit_os_auc(
    root: Path,
    out: Path,
    audit: Audit,
) -> tuple[pd.DataFrame, bool]:
    print("\n[3/5] Re-evaluating existing OS OOF predictions with censoring-aware AUC")

    try:
        import sksurv  # noqa: F401
        sksurv_available = True
        audit.add("PASS", "scikit-survival availability", "Available for cumulative_dynamic_auc.", True)
    except Exception as exc:
        sksurv_available = False
        audit.add(
            "NOT_VERIFIED",
            "scikit-survival availability",
            (
                f"{type(exc).__name__}: {exc}. RFS audit will continue. "
                "Install with: .\\.venv\\Scripts\\python.exe -m pip install scikit-survival"
            ),
            True,
        )

    details: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    m38 = root / "results" / "tables" / "metabric_m7" / "m38_oof_predictions_LOCAL_ONLY.csv"
    m41 = root / "results" / "tables" / "metabric_m8" / "m41_oof_predictions_LOCAL_ONLY.csv"

    if m38.is_file():
        f = read_csv(m38)
        r = evaluate_auc_dataset(
            f, "Multimodal", "OS", str(m38.relative_to(root)), details, audit, sksurv_available
        )
        if r:
            results.append(r)
    else:
        audit.add("NOT_VERIFIED", "OS multimodal OOF source", f"Missing {m38}", True)

    if m41.is_file():
        f = read_csv(m41)
        if "modality" not in f.columns:
            audit.add("FAIL", "OS modality OOF schema", "m41 file lacks modality column.", True)
        else:
            for modality, sub in f.groupby("modality", sort=True):
                canon = canonical_modality(modality)
                r = evaluate_auc_dataset(
                    sub.copy(),
                    canon,
                    "OS",
                    str(m41.relative_to(root)),
                    details,
                    audit,
                    sksurv_available,
                )
                if r:
                    results.append(r)
    else:
        audit.add("NOT_VERIFIED", "OS modality OOF source", f"Missing {m41}", True)

    save_csv(out / "os_auc_fold_details.csv", details)
    df = pd.DataFrame(results)
    df.to_csv(out / "os_auc_comparison.csv", index=False)

    for r in results:
        if np.isfinite(r["ipcw_delta_auc_mean"]):
            print(
                f"  OS {r['modality']}: legacy pooled ΔAUC="
                f"{r['legacy_pooled_binary_delta_auc_mean']:+.6f}; "
                f"IPCW fold-contained ΔAUC={r['ipcw_delta_auc_mean']:+.6f}"
            )
        else:
            print(
                f"  OS {r['modality']}: legacy pooled ΔAUC="
                f"{r['legacy_pooled_binary_delta_auc_mean']:+.6f}; IPCW NOT_VERIFIED"
            )
    return df, sksurv_available


def optionally_audit_rfs_auc(
    rfs_frames: dict[str, pd.DataFrame],
    out: Path,
    audit: Audit,
    sksurv_available: bool,
) -> pd.DataFrame:
    details: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if not rfs_frames:
        pd.DataFrame().to_csv(out / "rfs_auc_comparison_optional.csv", index=False)
        return pd.DataFrame()

    for modality, frame in sorted(rfs_frames.items()):
        source = str(frame.attrs.get("source_path", "auto-discovered RFS OOF"))
        r = evaluate_auc_dataset(
            frame, modality, "RFS", source, details, audit, sksurv_available
        )
        if r:
            results.append(r)
    save_csv(out / "rfs_auc_fold_details_optional.csv", details)
    df = pd.DataFrame(results)
    df.to_csv(out / "rfs_auc_comparison_optional.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Inference registry
# ---------------------------------------------------------------------------

def track_a_registry_rows(root: Path) -> list[dict[str, Any]]:
    base = root / "results" / "tables" / "metabric_m7"
    full_path = base / "m37_track_a_full_results.csv"
    draws_path = base / "m37_track_a_paired_deltas_1000.csv"
    if not full_path.is_file() or not draws_path.is_file():
        return []
    full = read_csv(full_path)
    draws = read_csv(draws_path)

    clinical = full.loc[full["model_set"].astype(str) == "clinical"]
    if len(clinical) != 1:
        return []
    c0 = float(num(clinical["harrell_c_index"]).iloc[0])

    rows: list[dict[str, Any]] = []
    for model in ["clinical_cna", "clinical_rna", "clinical_rna_cna"]:
        fr = full.loc[full["model_set"].astype(str) == model]
        dr = draws.loc[draws["model_set"].astype(str) == model]
        if len(fr) != 1 or dr.empty:
            continue
        vals = summarize(num(dr["delta_c_index_vs_clinical"]).to_numpy())
        point = float(num(fr["harrell_c_index"]).iloc[0]) - c0
        rows.append(
            {
                "endpoint": "OS",
                "track": "Track A fixed transport",
                "analysis": model,
                "modality": canonical_modality(model),
                "direct_point_delta_c": point,
                "bootstrap_mean_delta_c": vals["mean"],
                "ci_low": vals["ci_low"],
                "ci_high": vals["ci_high"],
                "bootstrap_draws": vals["n"],
                "inference_type": "paired patient bootstrap of fixed transported predictions",
                "multiplicity_adjustment": "NOT_APPLIED",
                "family_assignment": "UNASSIGNED_FOR_LATER_INFERENCE_PLAN",
            }
        )
    return rows


def os_trackb_registry_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Direct point estimates from OOF predictions.
    points: dict[tuple[str, str], float] = {}

    m38 = root / "results" / "tables" / "metabric_m7" / "m38_oof_predictions_LOCAL_ONLY.csv"
    if m38.is_file():
        f = read_csv(m38)
        schema0 = infer_oof_schema(f)
        if validate_oof_schema(schema0):
            schema = {k: v for k, v in schema0.items() if v is not None}
            p, _ = repeat_pooled_delta(f, schema)
            points[("combined_reconstructed", "Multimodal")] = p

    m41 = root / "results" / "tables" / "metabric_m8" / "m41_oof_predictions_LOCAL_ONLY.csv"
    if m41.is_file():
        f = read_csv(m41)
        if "modality" in f.columns:
            for modality, sub in f.groupby("modality", sort=True):
                schema0 = infer_oof_schema(sub)
                if validate_oof_schema(schema0):
                    schema = {k: v for k, v in schema0.items() if v is not None}
                    p, _ = repeat_pooled_delta(sub, schema)
                    points[("modality_specific", canonical_modality(modality))] = p

    draws_path = (
        root
        / "results"
        / "tables"
        / "metabric_m9"
        / "m46_oof_patient_bootstrap_2000.csv"
    )
    if not draws_path.is_file():
        return rows
    draws = read_csv(draws_path)
    if not {"analysis", "modality", "delta_c_index"}.issubset(draws.columns):
        return rows

    for (analysis, modality_raw), sub in draws.groupby(["analysis", "modality"], sort=True):
        modality = canonical_modality(modality_raw)
        vals = summarize(num(sub["delta_c_index"]).to_numpy())
        point = points.get((str(analysis), modality), math.nan)
        rows.append(
            {
                "endpoint": "OS",
                "track": "Track B reconstructed",
                "analysis": str(analysis),
                "modality": modality,
                "direct_point_delta_c": point,
                "bootstrap_mean_delta_c": vals["mean"],
                "ci_low": vals["ci_low"],
                "ci_high": vals["ci_high"],
                "bootstrap_draws": vals["n"],
                "inference_type": "conditional patient bootstrap of locked repeated OOF predictions",
                "multiplicity_adjustment": "NOT_APPLIED",
                "family_assignment": "UNASSIGNED_FOR_LATER_INFERENCE_PLAN",
            }
        )
    return rows


def rfs_registry_rows(oof: pd.DataFrame, boot: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modalities = set()
    if not oof.empty and "modality" in oof.columns:
        modalities.update(oof["modality"].dropna().astype(str))
    if not boot.empty and "modality" in boot.columns:
        modalities.update(boot["modality"].dropna().astype(str))

    for modality in sorted(modalities):
        oo = (
            oof.loc[
                (oof["modality"].astype(str) == modality)
                & (oof.get("source_resolution", pd.Series(index=oof.index, dtype=str)) != "AMBIGUOUS_NOT_CHOSEN")
            ]
            if not oof.empty
            else pd.DataFrame()
        )
        bb = boot.loc[boot["modality"].astype(str) == modality] if not boot.empty else pd.DataFrame()
        point = float(oo.iloc[0]["pooled_oof_mean_delta_c"]) if len(oo) == 1 else math.nan
        bmean = float(bb.iloc[0]["mean"]) if len(bb) == 1 else math.nan
        low = float(bb.iloc[0]["ci_low"]) if len(bb) == 1 else math.nan
        high = float(bb.iloc[0]["ci_high"]) if len(bb) == 1 else math.nan
        nboot = int(bb.iloc[0]["n"]) if len(bb) == 1 else 0
        rows.append(
            {
                "endpoint": "RFS",
                "track": "RFS sensitivity",
                "analysis": "sensitivity_endpoint",
                "modality": modality,
                "direct_point_delta_c": point,
                "bootstrap_mean_delta_c": bmean,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_draws": nboot,
                "inference_type": "conditional patient bootstrap of locked repeated OOF predictions",
                "multiplicity_adjustment": "NOT_APPLIED",
                "family_assignment": "UNASSIGNED_FOR_LATER_INFERENCE_PLAN",
            }
        )
    return rows


def build_inference_registry(
    root: Path,
    rfs_oof: pd.DataFrame,
    rfs_boot: pd.DataFrame,
    out: Path,
) -> pd.DataFrame:
    rows = track_a_registry_rows(root) + os_trackb_registry_rows(root) + rfs_registry_rows(rfs_oof, rfs_boot)
    df = pd.DataFrame(rows)
    df.to_csv(out / "inference_registry.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Report and share bundle
# ---------------------------------------------------------------------------

def write_report(
    out: Path,
    audit: Audit,
    inventory: pd.DataFrame,
    rfs_oof: pd.DataFrame,
    rfs_boot: pd.DataFrame,
    rfs_refs: pd.DataFrame,
    os_auc: pd.DataFrame,
    registry: pd.DataFrame,
    sksurv_available: bool,
    elapsed: float,
) -> dict[str, Any]:
    counts = audit.counts()
    status = audit.overall()
    summary = {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "checks_by_severity": dict(counts),
        "elapsed_seconds": round(elapsed, 3),
        "scope": "RFS saved-artifact audit + OS censoring-aware AUC + inference registry",
        "model_refits": 0,
        "new_bootstrap_draws": 0,
        "multiplicity_correction_applied": False,
        "scikit_survival_available": bool(sksurv_available),
        "patient_ids_in_share_bundle": False,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_csv(out / "checks.csv", audit.checks)

    lines = [
        "# Paper 2B -- M56 RFS and censoring-aware AUC audit",
        "",
        f"**Status: {status}**",
        "",
        "## Scope",
        "",
        "- No model refit.",
        "- No new bootstrap.",
        "- RFS saved OOF predictions and saved bootstrap draws are audited if auto-discovered unambiguously.",
        "- OS 5-year AUC is re-evaluated with fold-contained cumulative/dynamic IPCW AUC when scikit-survival is available.",
        "- No multiplicity correction or inferential family is chosen here.",
        "",
        "## RFS artifact discovery",
        "",
        f"Candidate RFS/m54/recurrence CSV files: {len(inventory)}.",
        "",
    ]

    if not rfs_oof.empty:
        lines += [
            "### RFS OOF",
            "",
            "| Modality | n | events | repeats | pooled ΔC | within-fold ΔC | source resolution |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for _, r in rfs_oof.iterrows():
            if str(r.get("source_resolution", "")) == "AMBIGUOUS_NOT_CHOSEN":
                continue
            lines.append(
                f"| {r.get('modality')} | {int(r.get('n_unique_patients',0))} | "
                f"{int(r.get('events',0))} | {int(r.get('repeats',0))} | "
                f"{r.get('pooled_oof_mean_delta_c', math.nan):+.6f} | "
                f"{r.get('within_fold_weighted_mean_delta_c', math.nan):+.6f} | "
                f"{r.get('source_resolution','')} |"
            )
        lines.append("")

    if not rfs_boot.empty:
        lines += [
            "### RFS saved bootstrap",
            "",
            "| Modality | saved draws | mean ΔC | 95% interval |",
            "|---|---:|---:|---:|",
        ]
        for _, r in rfs_boot.iterrows():
            lines.append(
                f"| {r['modality']} | {int(r['unique_repetitions'])} | "
                f"{r['mean']:+.6f} | [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] |"
            )
        lines.append("")

    if not rfs_refs.empty:
        lines += [
            "### Historical manuscript-reference check",
            "",
            "| Modality | reference | observed saved-bootstrap | status |",
            "|---|---:|---:|---|",
        ]
        for _, r in rfs_refs.iterrows():
            lines.append(
                f"| {r['modality']} | {r['reference_delta_c']:+.4f} "
                f"[{r['reference_ci_low']:+.4f},{r['reference_ci_high']:+.4f}] | "
                f"{r['saved_bootstrap_mean_delta_c']:+.4f} "
                f"[{r['saved_bootstrap_ci_low']:+.4f},{r['saved_bootstrap_ci_high']:+.4f}] | "
                f"{r['status']} |"
            )
        lines.append("")

    lines += [
        "## OS AUC re-evaluation",
        "",
        f"scikit-survival available: **{sksurv_available}**.",
        "",
    ]

    if not os_auc.empty:
        lines += [
            "| Modality | legacy pooled binary ΔAUC | IPCW fold-contained ΔAUC | IPCW status |",
            "|---|---:|---:|---|",
        ]
        for _, r in os_auc.iterrows():
            ipcw = r["ipcw_delta_auc_mean"]
            ipcw_text = f"{ipcw:+.6f}" if np.isfinite(ipcw) else "NOT_VERIFIED"
            lines.append(
                f"| {r['modality']} | {r['legacy_pooled_binary_delta_auc_mean']:+.6f} | "
                f"{ipcw_text} | {r['ipcw_status']} |"
            )
        lines.append("")

    lines += [
        "## Inference registry",
        "",
        f"Rows prepared for later multiplicity planning: {len(registry)}.",
        "",
        "No multiplicity correction was applied. `family_assignment` remains explicitly unassigned.",
        "",
        "## Interpretation boundaries",
        "",
        "- RFS remains a sensitivity endpoint even if an unadjusted interval excludes zero.",
        "- AUC re-evaluation changes an evaluation metric only; it does not refit the prognostic models.",
        "- The IPCW AUC here is fold-contained to avoid estimating the censoring distribution from the held-out fold itself.",
        "- Conditional bootstrap intervals do not include full model-development uncertainty.",
        "- Any future multiplicity procedure must be specified as a separate inferential extension.",
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
    names = [
        "audit_report.md",
        "audit_summary.json",
        "checks.csv",
        "rfs_artifact_inventory.csv",
        "rfs_oof_audit.csv",
        "rfs_bootstrap_recomputed.csv",
        "rfs_manuscript_reference_check.csv",
        "os_auc_comparison.csv",
        "rfs_auc_comparison_optional.csv",
        "inference_registry.csv",
        "console.log",
    ]
    zpath = out / "share_bundle_m56.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = out / name
            if p.is_file():
                z.write(p, arcname=name)
    return zpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    root = find_root(args.root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_parent = args.report_root.resolve() if args.report_root else root / "results" / "reports"
    out = report_parent / f"paper2b_m56_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    log_path = out / "console.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr

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
            audit = Audit()

            print("PAPER 2B / M56 / RFS + CENSORING-AWARE AUC READ-ONLY AUDIT")
            print(f"Version: {SCRIPT_VERSION}")
            print(f"Root: {root}")
            print(f"Output: {out}")
            print("No model refit; no new bootstrap; no multiplicity correction.")

            print("\n[1/5] Discovering RFS artifacts")
            inventory = inventory_rfs(root, out)
            print(f"  Found {len(inventory)} RFS/m54/recurrence CSV candidates.")

            print("\n[2/5] Auditing RFS OOF and saved bootstrap")
            rfs_oof, rfs_frames = audit_rfs_oof(root, inventory, out, audit)
            rfs_boot = audit_rfs_bootstrap(root, inventory, out, audit)
            rfs_refs = manuscript_rfs_check(rfs_oof, rfs_boot, out, audit)

            os_auc, sksurv_available = audit_os_auc(root, out, audit)

            print("\n[4/5] Optional RFS censoring-aware AUC from discovered OOF")
            rfs_auc = optionally_audit_rfs_auc(rfs_frames, out, audit, sksurv_available)
            if not rfs_auc.empty:
                for _, r in rfs_auc.iterrows():
                    print(
                        f"  RFS {r['modality']}: legacy pooled ΔAUC="
                        f"{r['legacy_pooled_binary_delta_auc_mean']:+.6f}; "
                        f"IPCW ΔAUC="
                        f"{r['ipcw_delta_auc_mean']:+.6f}"
                        if np.isfinite(r["ipcw_delta_auc_mean"])
                        else f"  RFS {r['modality']}: IPCW NOT_VERIFIED"
                    )

            print("\n[5/5] Building inference registry and report")
            registry = build_inference_registry(root, rfs_oof, rfs_boot, out)

            elapsed = time.time() - started
            summary = write_report(
                out,
                audit,
                inventory,
                rfs_oof,
                rfs_boot,
                rfs_refs,
                os_auc,
                registry,
                sksurv_available,
                elapsed,
            )
            share = make_share_zip(out)

            counts = audit.counts()
            print("\n" + "=" * 78)
            print(f"AUDIT STATUS: {summary['status']}")
            print("; ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "No checks")
            print("No models fitted; no bootstrap rerun; no multiplicity correction.")
            print(f"Report: {out / 'audit_report.md'}")
            print(f"Share:  {share}")
            if not sksurv_available:
                print(
                    "\nIPCW AUC was not verified because scikit-survival is unavailable.\n"
                    "Install into the same environment and rerun:\n"
                    r"  .\.venv\Scripts\python.exe -m pip install scikit-survival"
                )
            return 0 if summary["status"] != "NOT_COMPLETE" else 2

        except Exception as exc:
            print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
