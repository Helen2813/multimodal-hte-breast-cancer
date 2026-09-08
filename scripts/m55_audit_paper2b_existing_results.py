#!/usr/bin/env python3
"""Paper 2B -- first, read-only audit of existing METABRIC results (M7--M9).

Save as: scripts/m55_audit_paper2b_existing_results.py
Run from the existing project: python scripts/m55_audit_paper2b_existing_results.py
Alternative: python <this-file> --root <project-directory>
Requires Python >=3.10, numpy and pandas already used by the project.

NO fitting, tuning, feature selection, bootstrapping, downloading, or modification
of existing data, configurations, scripts, or results. Only a NEW report folder
is created. Reports contain aggregates, not patient IDs or patient predictions.

Scope:
* inspect current M7/M8/M9 configurations and saved M36/M40/M45 protocols;
* inspect actual M41 penalties, not just the configured fallback sequence;
* validate OS OOF schemas, patient/repeat/fold structure and outcome agreement;
* recompute pooled and comparable-pair-weighted within-test-fold Harrell C;
* reconstruct the NPI-observed/common-repeat population without imputing NPI;
* recompute M46 bootstrap summaries FROM SAVED DRAWS (no new resampling);
* compare checkpoint metrics and clearly separate point estimates from the
  mean of bootstrap estimates used in the original M46 reporting code;
* inventory Track A bootstrap files. Track A models are NOT rerun;
* reproduce only the old known-status binary AUC, explicitly NOT an IPCW AUC.

NOT a full scientific validation: this file does not certify leakage freedom,
training-set membership, censoring assumptions, statistical coverage, prespecifi-
cation chronology, panel provenance, RFS results, or manuscript readiness.
The saved OOF rows alone cannot establish those claims.

Schemas checked against repository commit:
73855f6e14c1bebfaa4c92e7e9169de1742454e5
Relevant sources: m38_run_track_b_full_repeated_nested.py,
m41_run_modality_specific_repeated_nested.py, _metabric_m8_utils.py,
_metabric_m9_utils.py, m46_bootstrap_repeated_oof_predictions.py,
m51_metabric_m10b_npi_benchmark.py. Source files are read, never executed.

Exit codes: 0 = audit executed without FAIL/MISSING (review WARNs separately);
1 = an inconsistency or unusable input was detected; 2 = required audit evidence
is missing; 3 = invocation/dependency/unexpected execution error.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print("Missing dependency: " + str(exc.name), file=sys.stderr)
    print("Use the existing project environment containing numpy and pandas. "
          "This script installs nothing.", file=sys.stderr)
    raise SystemExit(3)

VERSION = "1.0.0"
ATOL = 1e-8
MODALITIES = ("RNA", "CNV", "Methylation", "Mutation")
# Historical values from the supplied manuscript/discussion, NOT ground truth.
# They are comparison annotations only; no result is forced to match them.
REPORTED_OS = {
    "RNA": (1980, 0.0001),
    "CNV": (1981, 0.0024),
    "Methylation": (1417, -0.0217),
    "Mutation": (1905, -0.0014),
    "Multimodal": (1904, -0.0128),
}
TABLES = (
    "checks", "input_manifest", "config_values", "protocol_comparison",
    "penalizer_by_fold", "penalizer_summary", "cohort_audit",
    "npi_population_audit", "repeat_metrics", "fold_metrics", "metric_summary",
    "checkpoint_comparison", "bootstrap_audit", "bootstrap_summary_comparison",
    "reported_value_comparison", "source_hash_checks",
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_nested(obj: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def to_text(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return str(value)


def finite_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def close(a: Any, b: Any, atol: float = ATOL) -> bool:
    x, y = finite_float(a), finite_float(b)
    if math.isnan(x) and math.isnan(y):
        return True
    return math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, abs_tol=atol, rel_tol=1e-10)


def harrell_components(times: np.ndarray, events: np.ndarray,
                       risks: np.ndarray) -> tuple[float, float, int]:
    """C, concordance credit, comparable pairs. Larger risk = earlier event.

    Event/event ties in time are not comparable. Event/censor ties in time are
    comparable, as in lifelines' concordance_index. Exact risk ties get 1/2.
    A Fenwick tree gives O(n log n) complexity. No observations are discarded.
    """
    t, event_input, r = np.asarray(times, float), np.asarray(events, float), np.asarray(risks, float)
    if not (np.isfinite(event_input).all() and np.isin(event_input, [0, 1]).all()):
        raise ValueError("C-index events must be exactly 0 or 1")
    e = event_input.astype(int)
    if not (t.ndim == e.ndim == r.ndim == 1 and len(t) == len(e) == len(r)):
        raise ValueError("C-index inputs must be equal-length one-dimensional arrays")
    if not (np.isfinite(t).all() and np.isfinite(r).all() and np.isin(e, [0, 1]).all()):
        raise ValueError("C-index inputs must be finite with binary events")
    if len(t) == 0:
        return float("nan"), 0.0, 0
    _, rank = np.unique(r, return_inverse=True)
    rank = rank + 1
    tree = np.zeros(int(rank.max()) + 1, dtype=np.int64)

    def add(k: int) -> None:
        while k < len(tree):
            tree[k] += 1
            k += k & -k

    def count(k: int) -> int:
        total = 0
        while k:
            total += int(tree[k])
            k -= k & -k
        return total

    order = np.argsort(-t, kind="mergesort")
    ends = np.r_[np.flatnonzero(np.diff(t[order]) != 0) + 1, len(t)]
    start, at_risk, twice_credit, pairs = 0, 0, 0, 0
    for end in ends:
        group = order[start:end]
        censored = group[e[group] == 0]
        deaths = group[e[group] == 1]
        # Censors at the event time are comparable with the tied events.
        for i in censored:
            add(int(rank[i]))
            at_risk += 1
        for i in deaths:
            k = int(rank[i])
            lower = count(k - 1)
            equal = count(k) - lower
            twice_credit += 2 * lower + equal
            pairs += at_risk
        for i in deaths:
            add(int(rank[i]))
            at_risk += 1
        start = int(end)
    credit = twice_credit / 2.0
    return credit / pairs if pairs else float("nan"), credit, pairs


def known_status_auc(times: np.ndarray, events: np.ndarray,
                     risks: np.ndarray, horizon: float) -> tuple[float, int, int, int]:
    """Reproduce the OLD binary AUC only; NO IPCW/censoring adjustment.

    Cases: event by horizon; controls: observed time strictly after horizon.
    Others are excluded. Original code required >=10 evaluable observations.
    """
    cases = (events == 1) & (times <= horizon)
    controls = times > horizon
    keep = cases | controls
    ncase, ncontrol = int(cases.sum()), int(controls.sum())
    n = ncase + ncontrol
    if n < 10 or ncase == 0 or ncontrol == 0:
        return float("nan"), ncase, ncontrol, int((~keep).sum())
    r, y = risks[keep], cases[keep].astype(int)
    order = np.argsort(r, kind="mergesort")
    ends = np.r_[np.flatnonzero(np.diff(r[order]) != 0) + 1, len(order)]
    start, previous_controls, credit = 0, 0, 0.0
    for end in ends:
        labels = y[order[start:end]]
        a, b = int(labels.sum()), int(len(labels) - labels.sum())
        credit += a * (previous_controls + b / 2.0)
        previous_controls += b
        start = int(end)
    return credit / (ncase * ncontrol), ncase, ncontrol, int((~keep).sum())


def metric_pair(frame: pd.DataFrame, model_col: str, horizon: float) -> dict:
    t = frame["time_months"].to_numpy(float)
    e = frame["event"].to_numpy(int)
    c = frame["clinical_risk"].to_numpy(float)
    m = frame[model_col].to_numpy(float)
    cc, cn, cd = harrell_components(t, e, c)
    mc, mn, md = harrell_components(t, e, m)
    if cd != md:
        raise RuntimeError("Paired C-index denominators differ")
    ca, nc, nn, excluded = known_status_auc(t, e, c, horizon)
    ma, _, _, _ = known_status_auc(t, e, m, horizon)
    return {
        "n": len(frame), "events": int(e.sum()), "comparable_pairs": cd,
        "clinical_c": cc, "model_c": mc, "delta_c": mc - cc,
        "clinical_concordance_credit": cn, "model_concordance_credit": mn,
        "known_status_clinical_auc": ca, "known_status_model_auc": ma,
        "known_status_delta_auc": ma - ca, "auc_cases": nc,
        "auc_controls": nn, "auc_excluded": excluded,
        "auc_definition": "known-status binary AUC; NOT IPCW",
    }


def metrics_self_test() -> int:
    """Independent brute-force checks including tied times/risks/censoring."""
    rng = np.random.default_rng(55001)
    checked = 0
    for n in (0, 1, 2, 3, 11, 37):
        for _ in range(30):
            t = rng.integers(0, 12, n).astype(float)
            e = rng.integers(0, 2, n)
            r = rng.integers(-3, 4, n).astype(float)
            credit, pairs = 0.0, 0
            for i in range(n):
                for j in range(i + 1, n):
                    earlier = None
                    if t[i] < t[j] and e[i] == 1:
                        earlier = (i, j)
                    elif t[j] < t[i] and e[j] == 1:
                        earlier = (j, i)
                    elif t[i] == t[j] and e[i] != e[j]:
                        earlier = (i, j) if e[i] == 1 else (j, i)
                    if earlier is not None:
                        a, b = earlier
                        pairs += 1
                        credit += float(r[a] > r[b]) + 0.5 * float(r[a] == r[b])
            value, numerator, denominator = harrell_components(t, e, r)
            expected = credit / pairs if pairs else float("nan")
            if not (close(value, expected) and numerator == credit and denominator == pairs):
                raise AssertionError("Harrell C self-test failed")
            auc, nc, nn, _ = known_status_auc(t, e, r, 6.0)
            cases = np.flatnonzero((e == 1) & (t <= 6.0))
            controls = np.flatnonzero(t > 6.0)
            expected_auc = float("nan")
            if nc + nn >= 10 and nc and nn:
                expected_auc = sum(float(r[i] > r[j]) + 0.5 * float(r[i] == r[j])
                                   for i in cases for j in controls) / (nc * nn)
            if not close(auc, expected_auc):
                raise AssertionError("Known-status AUC self-test failed")
            checked += 1
    return checked


class Audit:
    def __init__(self, root: Path, out: Path):
        self.root, self.out = root, out
        self.tables: dict[str, list[dict]] = {name: [] for name in TABLES}
        self.messages: list[str] = []
        self.inputs: dict[Path, dict] = {}
        self.configs: dict[str, dict] = {}
        self.protocols: dict[str, dict] = {}
        self.valid_frames: dict[str, pd.DataFrame] = {}
        self.points: dict[str, float] = {}
        self.checkpoints: dict[str, pd.DataFrame] = {}
        self.horizon = 60.0
        self.started = time.monotonic()

    def log(self, text: str) -> None:
        print(text, flush=True)
        self.messages.append(text)

    def label(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return "<external>/" + path.name

    def path(self, value: str) -> Path:
        p = Path(str(value))
        return p.resolve() if p.is_absolute() else (self.root / p).resolve()

    def add(self, scope: str, check: str, severity: str,
            observed: Any = "", expected: Any = "", note: str = "") -> None:
        self.tables["checks"].append({
            "scope": scope, "check": check, "severity": severity,
            "observed": to_text(observed), "expected": to_text(expected), "note": note,
        })
        if severity in {"FAIL", "MISSING", "WARN"}:
            self.log(f"[{severity}] {scope}: {check} -- {note or to_text(observed)}")

    def test(self, ok: bool, scope: str, check: str, observed: Any = "",
             expected: Any = "", note: str = "", bad: str = "FAIL") -> bool:
        self.add(scope, check, "PASS" if ok else bad, observed, expected, note if not ok else "")
        return bool(ok)

    def read(self, path: Path, kind: str, required: bool = True) -> Any:
        scope = self.label(path)
        if not path.is_file():
            self.add(scope, "file_available", "MISSING" if required else "INFO",
                     note="Required audit input missing" if required else "Optional evidence not available")
            if path not in self.inputs:
                self.inputs[path] = {"path": scope, "exists": False, "required": required}
            return None
        try:
            before = path.stat()
            h = digest_file(path)
            if kind == "json":
                obj = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(obj, dict):
                    raise ValueError("Expected JSON object")
            elif kind == "csv":
                with path.open("r", encoding="utf-8-sig", newline="") as f:
                    header = next(csv.reader(f), [])
                if len(header) != len(set(header)):
                    self.add(scope, "unique_column_names", "FAIL", note="Duplicated CSV headers")
                    return None
                # Keep IDs as text. Do not silently coerce missing IDs to 'nan'.
                dtype = {"sample_id": "string"} if "sample_id" in header else None
                obj = pd.read_csv(path, encoding="utf-8-sig", dtype=dtype, low_memory=False)
            else:
                obj = path.read_text(encoding="utf-8-sig")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                self.add(scope, "stable_input", "FAIL", note="File changed while being read; stop other writers")
                return None
            self.inputs[path] = {"path": scope, "exists": True, "required": required,
                                 "size_bytes": before.st_size, "sha256": h,
                                 "rows": len(obj) if isinstance(obj, pd.DataFrame) else "",
                                 "mtime_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat()}
            return obj
        except Exception as exc:
            # Do not copy arbitrary CSV values/patient rows into the shareable log.
            self.add(scope, "readable_input", "FAIL", type(exc).__name__,
                     note="Input could not be parsed; file left unchanged")
            return None

    def csv(self, path: Path, required: bool = True) -> pd.DataFrame | None:
        return self.read(path, "csv", required)

    def require_cols(self, frame: pd.DataFrame, cols: set[str], scope: str) -> bool:
        missing = sorted(cols - set(frame.columns))
        return self.test(not missing, scope, "required_columns", missing, [],
                         "Required columns absent; this analysis is not silently adapted")

    def numeric(self, frame: pd.DataFrame, cols: list[str], scope: str) -> bool:
        ok = True
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            invalid = int((~np.isfinite(frame[col].to_numpy(float))).sum())
            ok = self.test(invalid == 0, scope, "finite_" + col, invalid, 0,
                           "Invalid values are not dropped or imputed by the audit") and ok
        return ok

    def load_configs(self) -> None:
        self.log("[1/5] Reading configurations, saved protocols and input identities")
        for stage in ("m7", "m8", "m9"):
            cfg = self.read(self.root / f"metabric_{stage}_config.json", "json")
            self.configs[stage] = cfg or {}
            for section in ("track_a", "track_b", "modality_analysis", "bootstrap"):
                for key, value in (cfg or {}).get(section, {}).items():
                    self.tables["config_values"].append({
                        "source": f"metabric_{stage}_config.json", "section": section,
                        "key": key, "value": to_text(value), "evidence": "CURRENT_CONFIG_NOT_EXECUTION_LOG"})
        c7, c8, c9 = (self.configs[k] for k in ("m7", "m8", "m9"))
        self.m7dir = self.path(c7.get("output_dir", "results/tables/metabric_m7"))
        self.m8dir = self.path(c8.get("output_dir", "results/tables/metabric_m8"))
        self.m9dir = self.path(c9.get("output_dir", "results/tables/metabric_m9"))
        f9 = c9.get("files", {})
        self.pred7 = self.path(f9.get("m7_combined_predictions", str(self.m7dir / "m38_oof_predictions_LOCAL_ONLY.csv")))
        self.pred8 = self.path(f9.get("m8_modality_predictions", str(self.m8dir / "m41_oof_predictions_LOCAL_ONLY.csv")))
        protocol_paths = {
            "m7": self.path(get_nested(c8, "files", "m7_protocol", default=str(self.pred7.parent / "m36_m7_full_core_protocol.json"))),
            "m8": self.path(f9.get("m8_protocol", str(self.pred8.parent / "m40_m8_protocol.json"))),
            "m9": self.m9dir / "m45_m9_final_inference_protocol.json",
        }
        for stage, path in protocol_paths.items():
            # M45 naming is checked conservatively: discover only unambiguous M45 JSON.
            if stage == "m9" and not path.is_file():
                candidates = sorted(self.m9dir.glob("m45*protocol*.json"))
                if len(candidates) == 1:
                    path = candidates[0]
                elif len(candidates) > 1:
                    self.add("m9", "protocol_choice", "WARN", len(candidates), 1,
                             "Ambiguous M45 protocols; no automatic choice")
                    continue
            self.protocols[stage] = self.read(path, "json", required=stage != "m9") or {}
        for label, cfgdir, preddir in (("M7", self.m7dir, self.pred7.parent), ("M8", self.m8dir, self.pred8.parent)):
            self.test(cfgdir == preddir, label, "config_output_matches_M9_input_directory",
                      self.label(preddir), self.label(cfgdir),
                      "Different directories are a provenance question, not proof of incorrect results", bad="WARN")
        value = get_nested(c9, "bootstrap", "five_year_months", default=60.0)
        self.horizon = finite_float(value)
        if not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("Invalid evaluation horizon in M9 configuration")
        self.expected = {}
        for name, stage, cfgsection, protocolsection in (
                ("Multimodal", "m7", "track_b", "track_b"),
                ("modalities", "m8", "modality_analysis", "modality_specific_analysis")):
            cfgs = self.configs[stage].get(cfgsection, {})
            ps = self.protocols[stage].get(protocolsection, {})
            settings = {}
            for key in ("outer_repeats", "outer_folds"):
                pv, cv = ps.get(key), cfgs.get(key)
                self.tables["protocol_comparison"].append({
                    "analysis": name, "parameter": key, "saved_protocol": pv,
                    "current_config": cv, "effective_source": "saved_protocol" if pv is not None else "current_config_only"})
                if pv is not None and cv is not None:
                    self.test(pv == cv, name, "protocol_vs_config_" + key, cv, pv,
                              "Current config differs from saved protocol; do not overwrite either", bad="WARN")
                settings[key] = pv if pv is not None else cv
            seeds = ps.get("repeat_seeds")
            if seeds is None and all(k in cfgs for k in ("repeat_seed_start", "outer_repeats")):
                seeds = list(range(int(cfgs["repeat_seed_start"]), int(cfgs["repeat_seed_start"]) + int(cfgs["outer_repeats"])))
            settings["repeat_seeds"] = seeds
            self.expected[name] = settings
        self.add("scope", "configuration_is_not_execution", "INFO",
                 note="Configured bootstrap counts/penalties are not labelled observed execution counts")
        self.verify_m10b_hashes()

    def verify_m10b_hashes(self) -> None:
        path = self.root / "scripts/m51_metabric_m10b_npi_benchmark.py"
        text = self.read(path, "text", required=False)
        if text is None:
            return
        try:
            values = None
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXPECTED_HASHES" for t in node.targets):
                    values = ast.literal_eval(node.value)
            if not isinstance(values, dict):
                self.add("M10B", "declared_hashes", "INFO", note="No literal EXPECTED_HASHES found; script not executed")
                return
            for rel, expected in values.items():
                p = self.path(str(rel))
                exists = p.is_file()
                observed = digest_file(p) if exists else ""
                status = "PASS" if observed == expected else ("FAIL" if exists else "MISSING")
                self.tables["source_hash_checks"].append({"source": self.label(path), "input": self.label(p),
                                                         "expected_sha256": expected, "observed_sha256": observed,
                                                         "status": status})
                self.add("M10B", "declared_input_hash:" + self.label(p), status,
                         note="Comparison to the hashes declared by the saved M10B source, not proof of chronology")
        except (SyntaxError, ValueError, TypeError):
            self.add("M10B", "declared_hashes", "WARN", note="Could not read literal hashes; no code was executed")

    def audit_penalties(self, frame: pd.DataFrame | None) -> None:
        if frame is None:
            return
        scope = "M41 penalties"
        cols = {"modality", "repeat", "fold", "clinical_penalizer", "modality_penalizer", "clinical_modality_penalizer"}
        if not self.require_cols(frame, cols, scope):
            return
        f = frame.copy()
        if not self.numeric(f, ["repeat", "fold", "clinical_penalizer", "modality_penalizer", "clinical_modality_penalizer"], scope):
            return
        duplicates = int(f.duplicated(["modality", "repeat", "fold"], keep=False).sum())
        if not self.test(duplicates == 0, scope, "one_checkpoint_per_fold", duplicates, 0):
            return
        sequence = get_nested(self.configs["m8"], "modality_analysis", "cox_penalizer_sequence", default=[])
        first = finite_float(sequence[0]) if sequence else float("nan")
        for mod, g in f.groupby("modality", sort=True):
            mismatch = ~np.isclose(g["clinical_penalizer"], g["clinical_modality_penalizer"], atol=1e-12, rtol=0)
            for role in ("clinical", "modality", "clinical_modality"):
                vals = g[role + "_penalizer"].to_numpy(float)
                for val, count in zip(*np.unique(vals, return_counts=True)):
                    self.tables["penalizer_summary"].append({"modality": mod, "model": role,
                        "actual_penalizer": val, "folds": int(count), "fraction": count / len(g),
                        "first_current_config_value": first,
                        "different_from_first_config_value": not close(val, first)})
                self.test(bool((vals >= 0).all()), str(mod), "nonnegative_" + role + "_penalty")
            for (_, row), different in zip(g.iterrows(), mismatch):
                self.tables["penalizer_by_fold"].append({"modality": mod, "repeat": row["repeat"],
                    "fold": row["fold"], "clinical_penalizer": row["clinical_penalizer"],
                    "modality_penalizer": row["modality_penalizer"],
                    "clinical_modality_penalizer": row["clinical_modality_penalizer"],
                    "clinical_vs_augmented_differ": bool(different),
                    "stored_delta_c": row.get("delta_c_index_vs_clinical", "")})
            self.test(int(mismatch.sum()) == 0, str(mod), "paired_models_same_actual_penalizer",
                      int(mismatch.sum()), 0,
                      "Different fallback penalties need reporting; not automatically an invalid comparison", bad="WARN")
            self.log(f"  {mod}: {len(g)} checkpoint folds; paired penalty differences={int(mismatch.sum())}")
        self.add(scope, "fallback_definition", "INFO",
                 note="The source returns the first fit without an exception; this is not inner-CV tuning. "
                      "A non-throwing fit is not a certificate that no convergence warning occurred.")

    def prepare_predictions(self, frame: pd.DataFrame, model_col: str, scope: str) -> pd.DataFrame | None:
        required = {"sample_id", "repeat", "fold", "time_months", "event", "clinical_risk", model_col}
        if not self.require_cols(frame, required, scope):
            return None
        f = frame.copy()
        missing_id = f["sample_id"].isna() | f["sample_id"].astype("string").str.strip().eq("").fillna(True)
        if not self.test(not missing_id.any(), scope, "nonmissing_patient_ids", int(missing_id.sum()), 0):
            return None
        f["sample_id"] = f["sample_id"].astype(str)
        whitespace = int(f["sample_id"].ne(f["sample_id"].str.strip()).sum())
        if not self.test(whitespace == 0, scope, "no_ID_whitespace", whitespace, 0,
                         "IDs are not silently normalized"):
            return None
        if not self.numeric(f, ["repeat", "fold", "time_months", "event", "clinical_risk", model_col], scope):
            return None
        ok = True
        for col in ("repeat", "fold"):
            valid = (f[col] >= 1) & (f[col] == np.floor(f[col]))
            ok = self.test(bool(valid.all()), scope, "positive_integer_" + col, int((~valid).sum()), 0) and ok
        ok = self.test(bool(f["event"].isin([0, 1]).all()), scope, "binary_events") and ok
        ok = self.test(bool((f["time_months"] >= 0).all()), scope, "nonnegative_followup") and ok
        duplicates = int(f.duplicated(["repeat", "sample_id"], keep=False).sum())
        ok = self.test(duplicates == 0, scope, "one_OOF_row_per_patient_per_repeat", duplicates, 0) and ok
        if not ok:
            return None
        f["repeat"] = f["repeat"].astype(int)
        f["fold"] = f["fold"].astype(int)
        f["event"] = f["event"].astype(int)
        changes = f.groupby("sample_id", sort=False)[["time_months", "event"]].nunique(dropna=False)
        if not self.test(bool((changes == 1).all().all()), scope, "outcomes_constant_across_repeats",
                         int((changes > 1).any(axis=1).sum()), 0):
            return None
        sets = [set(g["sample_id"]) for _, g in f.groupby("repeat", sort=True)]
        union, intersection = set.union(*sets), set.intersection(*sets)
        self.test(union == intersection, scope, "same_patients_in_every_repeat", len(union) - len(intersection), 0,
                  "Repeat populations differ; no all-repeat pooled summary will be certified")
        ex = self.expected["Multimodal" if scope == "Multimodal" else "modalities"]
        for key, col in (("outer_repeats", "repeat"), ("outer_folds", "fold")):
            wanted = ex.get(key)
            if wanted is not None:
                if col == "repeat":
                    good = set(f[col]) == set(range(1, int(wanted) + 1))
                else:
                    good = all(set(g[col]) == set(range(1, int(wanted) + 1)) for _, g in f.groupby("repeat"))
                self.test(good, scope, "expected_" + key, f[col].nunique(), wanted)
        if "seed" in f:
            f["seed"] = pd.to_numeric(f["seed"], errors="coerce")
            seeds = ex.get("repeat_seeds")
            for rep, g in f.groupby("repeat", sort=True):
                values = g["seed"].drop_duplicates().tolist()
                desired = seeds[rep - 1] if seeds and 0 < rep <= len(seeds) else None
                good = len(values) == 1 and np.isfinite(values[0]) and (desired is None or values[0] == desired)
                self.test(good, scope, f"repeat_{rep}_seed", values, desired)
        self.tables["cohort_audit"].append({"analysis": scope, "endpoint": "OS", "rows": len(f),
            "unique_patients_union": len(union), "common_patients_all_repeats": len(intersection),
            "repeats": f["repeat"].nunique(), "folds_min": f.groupby("repeat")["fold"].nunique().min(),
            "folds_max": f.groupby("repeat")["fold"].nunique().max(),
            "events_unique_union": int(f.drop_duplicates("sample_id")["event"].sum()),
            "zero_followup_unique": int(f.drop_duplicates("sample_id")["time_months"].eq(0).sum()),
            "reported_primary_n_not_ground_truth": REPORTED_OS.get(scope, (None, None))[0]})
        for role in ("clinical_risk", model_col):
            nbad = int(f[role].le(0).sum())
            if nbad:
                self.add(scope, "nonpositive_" + role, "WARN", nbad,
                         note="Finite scores retained for ranking; verify whether these are hazards or transformed scores")
        return f

    def audit_predictions(self) -> None:
        self.log("[2/5] Reading checkpoints and OS out-of-fold predictions")
        cp7 = self.csv(self.pred7.parent / "m38_fold_checkpoint.csv")
        cp8 = self.csv(self.pred8.parent / "m41_fold_checkpoint.csv")
        self.audit_penalties(cp8)
        p7, p8 = self.csv(self.pred7), self.csv(self.pred8)
        jobs = []
        if p7 is not None and len(p7):
            jobs.append(("Multimodal", p7, "model_risk", cp7))
        elif p7 is not None:
            self.add("Multimodal", "nonempty_predictions", "FAIL")
        if p8 is not None and self.require_cols(p8, {"modality"}, "M41 OOF"):
            empty_modality = p8["modality"].isna() | p8["modality"].astype("string").str.strip().eq("").fillna(True)
            self.test(not empty_modality.any(), "M41 OOF", "nonmissing_modality_labels", int(empty_modality.sum()), 0)
            mods = set(p8["modality"].dropna().astype(str))
            self.test(mods == set(MODALITIES), "M41 OOF", "expected_modalities", sorted(mods), list(MODALITIES))
            for mod, g in p8.groupby("modality", sort=True):
                if mod not in MODALITIES:
                    continue
                checkpoint = None
                if cp8 is not None and "modality" in cp8:
                    checkpoint = cp8.loc[cp8["modality"] == mod].copy()
                jobs.append((str(mod), g.copy(), "clinical_modality_risk", checkpoint))
        for name, frame, mcol, checkpoint in jobs:
            if len(frame) == 0:
                self.add(name, "nonempty_predictions", "FAIL")
                continue
            f = self.prepare_predictions(frame, mcol, name)
            if f is None:
                continue
            self.valid_frames[name] = f
            if checkpoint is not None:
                self.checkpoints[name] = checkpoint
            self.evaluate(name, f, mcol, checkpoint)
        self.audit_clinical_and_npi()

    def evaluate(self, name: str, f: pd.DataFrame, model_col: str,
                 checkpoint: pd.DataFrame | None) -> None:
        self.log(f"  Recomputing {name}: {f['sample_id'].nunique()} patients, {f['repeat'].nunique()} repeats")
        cp_map = {}
        if checkpoint is not None and self.require_cols(checkpoint, {"repeat", "fold"}, name + " checkpoint"):
            cp = checkpoint.copy()
            if self.numeric(cp, ["repeat", "fold"], name + " checkpoint"):
                dup = int(cp.duplicated(["repeat", "fold"], keep=False).sum())
                valid_keys = bool(((cp["repeat"] > 0) & (cp["repeat"] % 1 == 0) &
                                   (cp["fold"] > 0) & (cp["fold"] % 1 == 0)).all())
                if self.test(dup == 0 and valid_keys, name, "checkpoint_keys_valid", dup, 0):
                    cp_map = {(int(r["repeat"]), int(r["fold"])): r for _, r in cp.iterrows()}
                    oof_keys = set(zip(f["repeat"], f["fold"]))
                    self.test(set(cp_map) == oof_keys, name, "OOF_vs_checkpoint_fold_keys",
                              len(set(cp_map) ^ oof_keys), 0)
        repeat_results = []
        reference_library_checked = False
        for rep, group in f.groupby("repeat", sort=True):
            pooled = metric_pair(group, model_col, self.horizon)
            fold_results = []
            for fold, g in group.groupby("fold", sort=True):
                vals = metric_pair(g, model_col, self.horizon)
                self.tables["fold_metrics"].append({"analysis": name, "repeat": rep, "fold": fold, **vals})
                fold_results.append(vals)
                saved = cp_map.get((rep, fold))
                if saved is not None:
                    mapping = {"test_n": "n", "test_events": "events", "delta_c_index_vs_clinical": "delta_c",
                               "delta_auc_5y_vs_clinical": "known_status_delta_auc"}
                    mapping.update({"clinical_only_c_index": "clinical_c", "harrell_c_index": "model_c"}
                                   if name == "Multimodal" else
                                   {"clinical_c_index": "clinical_c", "clinical_modality_c_index": "model_c"})
                    for oldkey, newkey in mapping.items():
                        if oldkey not in saved:
                            continue
                        good = close(saved[oldkey], vals[newkey])
                        self.tables["checkpoint_comparison"].append({"analysis": name, "repeat": rep,
                            "fold": fold, "metric": oldkey, "stored": saved[oldkey], "recomputed": vals[newkey],
                            "difference": finite_float(vals[newkey]) - finite_float(saved[oldkey]), "agrees": good})
                        if not good:
                            self.add(name, f"checkpoint_metric:{oldkey}:r{rep}:f{fold}", "FAIL",
                                     vals[newkey], saved[oldkey], "Stored checkpoint and recomputed OOF metric differ")
                    for oldkey, actual in (("train_n", len(group) - len(g)),
                                           ("train_events", int(group["event"].sum() - g["event"].sum()))):
                        if oldkey in saved:
                            self.test(close(saved[oldkey], actual), name,
                                      f"checkpoint_{oldkey}:r{rep}:f{fold}", actual, saved[oldkey])
            pairs = sum(x["comparable_pairs"] for x in fold_results)
            cwithin = sum(x["clinical_concordance_credit"] for x in fold_results) / pairs if pairs else float("nan")
            mwithin = sum(x["model_concordance_credit"] for x in fold_results) / pairs if pairs else float("nan")
            row = {"analysis": name, "repeat": rep, **pooled,
                   "within_fold_comparable_pairs": pairs,
                   "within_fold_clinical_c": cwithin, "within_fold_model_c": mwithin,
                   "within_fold_delta_c": mwithin - cwithin,
                   "pooled_minus_within_fold_delta_c": pooled["delta_c"] - (mwithin - cwithin)}
            self.tables["repeat_metrics"].append(row)
            repeat_results.append(row)
            if not reference_library_checked:
                reference_library_checked = True
                try:
                    from lifelines.utils import concordance_index
                except ImportError:
                    self.add(name, "optional_lifelines_crosscheck", "INFO",
                             note="lifelines unavailable; native metric passed independent brute-force self-tests")
                else:
                    for role, col in (("clinical_c", "clinical_risk"), ("model_c", model_col)):
                        if pooled["comparable_pairs"]:
                            library = concordance_index(group["time_months"].to_numpy(float),
                                -group[col].to_numpy(float), event_observed=group["event"].to_numpy(int))
                            self.test(close(library, pooled[role]), name, "lifelines_" + role,
                                      pooled[role], library, "Independent library C-index disagrees")
        if not repeat_results:
            return
        rr = pd.DataFrame(repeat_results)
        same = f.groupby("sample_id")["repeat"].nunique().eq(f["repeat"].nunique()).all()
        point = float(rr["delta_c"].mean())
        within = float(rr["within_fold_delta_c"].mean())
        finite = bool(np.isfinite(rr[["clinical_c", "model_c", "within_fold_delta_c"]]).all().all())
        self.test(finite, name, "estimable_C_in_all_repeats", note="No comparable pairs in at least one evaluation")
        self.tables["metric_summary"].append({"analysis": name, "endpoint": "OS", "repeats": len(rr),
            "n_unique": f["sample_id"].nunique(), "common_repeat_population": bool(same),
            "mean_clinical_c_point": rr["clinical_c"].mean(), "mean_model_c_point": rr["model_c"].mean(),
            "mean_pooled_delta_c_POINT": point, "mean_within_fold_delta_c_POINT": within,
            "pooled_minus_within_fold_delta_c": point - within,
            "repeat_delta_sd_DESCRIPTIVE_NOT_SE": rr["delta_c"].std(ddof=1),
            "mean_known_status_delta_auc": rr["known_status_delta_auc"].mean(),
            "horizon_months": self.horizon,
            "note": "Point means across fitted repeats. No CI or full-pipeline uncertainty estimated here."})
        if same and finite:
            self.points[name] = point
        self.log(f"    pooled point delta C={point:+.8f}; within-fold={within:+.8f}")

    def audit_clinical_and_npi(self) -> None:
        self.log("[3/5] Comparing outcome records and reconstructing NPI-restricted populations")
        c7 = self.configs["m7"]
        p = self.path(get_nested(c7, "files", "clinical_master",
                                default="results/tables/metabric_m2/m06_metabric_clinical_master_LOCAL_ONLY.csv"))
        clinical = self.csv(p)
        if clinical is None or not self.require_cols(clinical, {"sample_id", "os_months", "os_event", "npi"}, "clinical master"):
            return
        if clinical["sample_id"].isna().any() or clinical["sample_id"].duplicated().any():
            self.add("clinical master", "unique_nonmissing_IDs", "FAIL",
                     note="Unsafe clinical join; no ID deduplication performed")
            return
        c = clinical.set_index("sample_id")
        for name, f in self.valid_frames.items():
            one = f.drop_duplicates("sample_id").set_index("sample_id")
            missing = one.index.difference(c.index)
            self.test(len(missing) == 0, name, "all_OOF_IDs_in_clinical_master", len(missing), 0)
            overlap = one.index.intersection(c.index)
            a, b = one.loc[overlap], c.loc[overlap]
            bt = pd.to_numeric(b["os_months"], errors="coerce").to_numpy(float)
            be = pd.to_numeric(b["os_event"], errors="coerce").to_numpy(float)
            valid = np.isfinite(bt) & np.isfinite(be)
            self.test(bool(valid.all()), name, "clinical_outcome_present_for_OOF_patients", int((~valid).sum()), 0)
            mismatch = ((~np.isclose(a["time_months"].to_numpy(float), bt, atol=1e-8, rtol=0)) |
                        (a["event"].to_numpy(int) != be)) & valid
            self.test(not mismatch.any(), name, "OOF_outcomes_equal_clinical_master", int(mismatch.sum()), 0)
            npi_present = set(b.index[b["npi"].notna()])
            rawsets = [set(g["sample_id"]) for _, g in f.groupby("repeat", sort=True)]
            union, common = set.union(*rawsets), set.intersection(*rawsets)
            # Exact M10B population rule: observed NPI, then common across repeats.
            retained = common & npi_present
            excluded_missing_master = union - set(c.index)
            excluded_missing_npi = (union & set(c.index)) - npi_present
            excluded_repeat = (union & npi_present) - common
            npi_numeric = pd.to_numeric(b["npi"], errors="coerce")
            invalid_observed = b["npi"].notna() & (~np.isfinite(npi_numeric.to_numpy(float)))
            self.test(not invalid_observed.any(), name, "observed_NPI_is_finite_numeric", int(invalid_observed.sum()), 0)
            self.tables["npi_population_audit"].append({
                "analysis": name, "primary_unique_union_n": len(union),
                "primary_common_all_repeats_n": len(common), "npi_common_population_n": len(retained),
                "excluded_ID_absent_from_master": len(excluded_missing_master),
                "excluded_missing_NPI": len(excluded_missing_npi),
                "excluded_not_in_every_repeat_after_NPI": len(excluded_repeat),
                "total_removed": len(union) - len(retained),
                "retained_events": int(one.loc[sorted(retained), "event"].sum()) if retained else 0,
                "reason_sets_disjoint": not bool(excluded_missing_master & excluded_missing_npi or excluded_missing_npi & excluded_repeat),
                "note": "Reconstructed M10B eligibility; not a comparison to an unavailable M10B output registry."})
            self.log(f"  {name}: primary union n={len(union)} -> NPI/common n={len(retained)}; "
                     f"missing NPI={len(excluded_missing_npi)}, repeat-intersection losses={len(excluded_repeat)}")
        names = list(self.valid_frames)
        for i, left in enumerate(names):
            a = self.valid_frames[left].drop_duplicates("sample_id").set_index("sample_id")
            for right in names[i + 1:]:
                b = self.valid_frames[right].drop_duplicates("sample_id").set_index("sample_id")
                common = a.index.intersection(b.index)
                bad = (~np.isclose(a.loc[common, "time_months"], b.loc[common, "time_months"], rtol=0, atol=1e-8)
                       | (a.loc[common, "event"].to_numpy() != b.loc[common, "event"].to_numpy()))
                self.test(not bad.any(), left + "/" + right, "shared_patients_same_OS", int(bad.sum()), 0)

    def audit_bootstrap(self) -> None:
        self.log("[4/5] Inspecting saved bootstrap draws; no bootstrap is rerun")
        draws = self.csv(self.m9dir / "m46_oof_patient_bootstrap_2000.csv")
        summaries = self.csv(self.m9dir / "m46_oof_patient_bootstrap_summary.csv")
        self.bootstrap_means = {}
        if draws is not None and self.require_cols(draws, {"analysis", "modality", "bootstrap", "delta_c_index"}, "M46 draws"):
            metrics = ("mean_clinical_c_index", "mean_model_c_index", "delta_c_index",
                       "mean_clinical_auc_5y", "mean_model_auc_5y", "delta_auc_5y")
            expected_groups = {("modality_specific", m) for m in MODALITIES} | {("combined_reconstructed", "Multimodal")}
            observed_groups = set(zip(draws["analysis"].astype(str), draws["modality"].astype(str)))
            self.test(observed_groups == expected_groups, "M46 draws", "complete_analysis_groups",
                      sorted(observed_groups), sorted(expected_groups))
            if summaries is not None:
                self.require_cols(summaries, {"analysis", "modality", "metric", "mean", "ci_low", "ci_high"}, "M46 summaries")
            for (analysis, mod), g in draws.groupby(["analysis", "modality"], sort=True):
                scope = "M46/" + str(mod)
                b = pd.to_numeric(g["bootstrap"], errors="coerce")
                valid_b = np.isfinite(b) & b.ge(1) & b.eq(np.floor(b))
                self.test(bool(valid_b.all()) and not b.duplicated().any(), scope, "unique_valid_draw_numbers")
                if bool(valid_b.all()):
                    self.test(set(b.astype(int)) == set(range(1, len(g) + 1)), scope,
                              "contiguous_draw_numbers", int(b.nunique()), len(g),
                              "Saved draw IDs have gaps, duplicates, or a non-one origin")
                for clinical_key, model_key, delta_key in (
                        ("mean_clinical_c_index", "mean_model_c_index", "delta_c_index"),
                        ("mean_clinical_auc_5y", "mean_model_auc_5y", "delta_auc_5y")):
                    if all(k in g for k in (clinical_key, model_key, delta_key)):
                        cv = pd.to_numeric(g[clinical_key], errors="coerce").to_numpy(float)
                        mv = pd.to_numeric(g[model_key], errors="coerce").to_numpy(float)
                        dv = pd.to_numeric(g[delta_key], errors="coerce").to_numpy(float)
                        finite = np.isfinite(cv) & np.isfinite(mv) & np.isfinite(dv)
                        bad_delta = ~np.isclose(mv[finite] - cv[finite], dv[finite], atol=ATOL, rtol=1e-10)
                        self.test(not bad_delta.any(), scope, "paired_identity_" + delta_key,
                                  int(bad_delta.sum()), 0, "Stored paired delta does not equal model minus clinical")
                configured = get_nested(self.configs["m9"], "bootstrap", "repetitions")
                if configured is not None:
                    self.test(len(g) == int(configured), scope, "draw_count_vs_current_config", len(g), configured,
                              "Configured count is not automatically the count used for a published interval", bad="WARN")
                for metric in metrics:
                    if metric not in g:
                        continue
                    all_v = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
                    v = all_v[np.isfinite(all_v)]
                    self.test(len(v) == len(all_v), scope, "finite_draws_" + metric, len(v), len(all_v),
                              "Finite draws only are summarized, matching the historical summarizer", bad="WARN")
                    if len(v) == 0:
                        self.add(scope, "estimable_" + metric, "FAIL")
                        continue
                    values = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                              "median": float(np.median(v)), "ci_low": float(np.quantile(v, 0.025)),
                              "ci_high": float(np.quantile(v, 0.975)), "fraction_positive": float(np.mean(v > 0))}
                    self.tables["bootstrap_audit"].append({"analysis": analysis, "modality": mod,
                        "metric": metric, "rows": len(g), "valid_draws": len(v), **values,
                        "note": "Descriptive re-summary of stored conditional draws; fraction_positive is NOT a p-value"})
                    if metric == "delta_c_index":
                        self.bootstrap_means[str(mod)] = values["mean"]
                    if summaries is not None and {"analysis", "modality", "metric"}.issubset(summaries.columns):
                        hit = summaries[(summaries["analysis"] == analysis) & (summaries["modality"] == mod) & (summaries["metric"] == metric)]
                        if not self.test(len(hit) == 1, scope, "one_summary_row_" + metric, len(hit), 1):
                            continue
                        for statistic, computed in values.items():
                            if statistic not in hit:
                                continue
                            stored = hit.iloc[0][statistic]
                            good = close(stored, computed)
                            self.tables["bootstrap_summary_comparison"].append({"modality": mod,
                                "metric": metric, "statistic": statistic, "stored": stored, "recomputed": computed,
                                "difference": computed - finite_float(stored), "agrees": good})
                            if not good:
                                self.add(scope, metric + ":" + statistic, "FAIL", computed, stored,
                                         "Summary is inconsistent with saved draws")
        # Track A: inventory only, no assumptions about unavailable model objects.
        paths = sorted(self.pred7.parent.glob("m37*bootstrap*.csv"))
        if not paths:
            self.add("Track A", "saved_bootstrap_draw_files", "MISSING",
                     note="No m37*bootstrap*.csv found; configured 1000 is not treated as observed")
        for path in paths:
            frame = self.csv(path, required=False)
            if frame is None:
                continue
            bcol = next((col for col in ("bootstrap", "bootstrap_id", "replicate", "iteration") if col in frame), None)
            self.tables["bootstrap_audit"].append({"analysis": "Track A file inventory", "file": self.label(path),
                "rows": len(frame), "bootstrap_id_column": bcol or "not_recognized",
                "unique_draw_numbers": frame[bcol].nunique() if bcol else "",
                "configured_repetitions_NOT_observed": get_nested(self.configs["m7"], "track_a", "bootstrap_repetitions"),
                "note": "Inventory only: model-specific completeness and Track A metric reconstruction not certified"})
        for mod in MODALITIES + ("Multimodal",):
            expected_n, reported_delta = REPORTED_OS[mod]
            point, bmean = self.points.get(mod), self.bootstrap_means.get(mod)
            n = self.valid_frames[mod]["sample_id"].nunique() if mod in self.valid_frames else None
            match = close(bmean, reported_delta, atol=0.00005001) if bmean is not None else None
            self.tables["reported_value_comparison"].append({"analysis": mod,
                "reported_primary_n": expected_n, "observed_primary_n": n,
                "reported_delta_4dp": reported_delta, "recomputed_OOF_POINT_delta": point,
                "stored_draws_BOOTSTRAP_MEAN_delta": bmean,
                "bootstrap_mean_minus_point": bmean - point if bmean is not None and point is not None else "",
                "bootstrap_mean_matches_reported_4dp": match,
                "note": "Reported values are supplied historical targets, not truth. M46 prints bootstrap means, not OOF points."})
            if n is not None and n != expected_n:
                self.add(mod, "reported_vs_observed_primary_n", "WARN", n, expected_n,
                         "Check manuscript version and cohort definition; no row is removed to force agreement")
            if match is False:
                self.add(mod, "reported_delta_vs_bootstrap_mean", "WARN", bmean, reported_delta,
                         "Historical rounded manuscript value differs; verify source/version before editing")

    def finish(self) -> int:
        self.log("[5/5] Writing aggregate reports and a shareable ZIP")
        # Detect source changes during the audit (for example, a concurrent old runner).
        for path, record in self.inputs.items():
            if record.get("exists") and record.get("sha256"):
                unchanged = path.is_file() and digest_file(path) == record["sha256"]
                record["unchanged_at_audit_end"] = unchanged
                self.test(unchanged, record["path"], "input_unchanged_during_audit",
                          note="Input changed during audit; rerun only after other writers stop")
        self.tables["input_manifest"] = list(self.inputs.values())
        counts = Counter(row["severity"] for row in self.tables["checks"])
        if counts["FAIL"]:
            status, code = "INCONSISTENCIES_DETECTED", 1
        elif counts["MISSING"]:
            status, code = "INCOMPLETE_MISSING_EVIDENCE", 2
        elif counts["WARN"]:
            status, code = "COMPLETED_REVIEW_WARNINGS", 0
        else:
            status, code = "CHECKED_ARTIFACTS_CONSISTENT", 0
        summary = {"script_version": VERSION, "status": status, "exit_code": code,
                   "checks_by_severity": dict(counts), "analyses_with_valid_OOF_schema": sorted(self.valid_frames),
                   "elapsed_seconds": round(time.monotonic() - self.started, 2),
                   "scope": "Read-only OS M7--M9 / NPI audit; not certification of the paper",
                   "refitted_models": 0, "new_bootstrap_draws": 0,
                   "patient_level_outputs_written": False,
                   "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
        for name, rows in self.tables.items():
            path = self.out / (name + ".csv")
            if rows:
                pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
            else:
                pd.DataFrame(columns=["no_rows_for_this_check"]).to_csv(path, index=False, encoding="utf-8-sig")
        (self.out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        sections = ["# Paper 2B -- first audit of saved results", "", f"**Status: {status}**", "",
                    "This status covers only the checks listed below, not manuscript readiness.", "",
                    "## Scope and interpretation", "",
                    "- Inputs were read only. No models were fitted and no bootstrap was rerun.",
                    "- OS M7/M8/M9 and reconstructed NPI eligibility are covered. RFS is not covered.",
                    "- Native Harrell C passed brute-force tests; available lifelines is also cross-checked.",
                    "- Within-fold C pools concordance credits/comparable pairs within test folds, then averages repeats.",
                    "- Known-status AUC reproduces the OLD binary calculation; it is not IPCW AUC.",
                    "- A point estimate from OOF predictions and the mean of bootstrap draws are different quantities.",
                    "- M46 reporting uses bootstrap means. Their difference from an OOF point is not automatically a bug.",
                    "- No inference on pooled-vs-within-fold differences is claimed by this audit.",
                    "- NPI counts are reconstructed from observed NPI plus common IDs across repeats.",
                    "  They are not silently substituted for the primary population.",
                    "- Hashes document available files, not prospective registration or temporal order.",
                    "- OOF rows cannot prove that preprocessing/feature selection used training data only.",
                    "- RFS, panel provenance, multiplicity, IPCW AUC, full-pipeline coverage and refit experiments are deferred.",
                    "", "## Findings needing attention", ""]
        attention = [r for r in self.tables["checks"] if r["severity"] in {"FAIL", "MISSING", "WARN"}]
        if not attention:
            sections.append("No discrepancies in the checks that could be executed.")
        for row in attention:
            sections.append(f"- **{row['severity']}** {row['scope']} / {row['check']}: "
                            f"observed={row['observed']}; expected={row['expected']}. {row['note']}")
        sections.extend(["", "## Recomputed point comparisons", "",
                         "| Analysis | n | Pooled point delta C | Within-fold point delta C |",
                         "|---|---:|---:|---:|"])
        for r in self.tables["metric_summary"]:
            sections.append(f"| {r['analysis']} | {r['n_unique']} | {r['mean_pooled_delta_c_POINT']:+.8f} | "
                            f"{r['mean_within_fold_delta_c_POINT']:+.8f} |")
        sections.extend(["", "## NPI eligibility reconstruction", "",
                         "| Analysis | Primary union n | NPI/common n | Missing NPI | Other exclusions |",
                         "|---|---:|---:|---:|---:|"])
        for r in self.tables["npi_population_audit"]:
            sections.append(f"| {r['analysis']} | {r['primary_unique_union_n']} | {r['npi_common_population_n']} | "
                            f"{r['excluded_missing_NPI']} | "
                            f"{r['excluded_ID_absent_from_master'] + r['excluded_not_in_every_repeat_after_NPI']} |")
        sections.extend(["", "## Files", "",
                         "All CSVs, audit_summary.json, and console.log are aggregate reports.",
                         "share_bundle.zip contains these reports, not input patient files or copies of the old code.",
                         "Input-manifest file hashes are hashes of whole files, not hashes of individual patient IDs.",
                         "", "## Stop rule", "",
                         "Resolve FAILs and missing required evidence before treating the numerical audit as complete.",
                         "WARNs request interpretation; a fallback penalty or distinct NPI population is not automatically invalid."])
        (self.out / "audit_report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
        self.log(f"\nSTATUS: {status}; FAIL={counts['FAIL']}; MISSING={counts['MISSING']}; WARN={counts['WARN']}")
        self.log("Reports: " + self.label(self.out))
        self.log("Share: " + self.label(self.out / "share_bundle.zip"))
        (self.out / "console.log").write_text("\n".join(self.messages) + "\n", encoding="utf-8")
        files = sorted(p for p in self.out.iterdir() if p.is_file() and p.suffix != ".zip")
        with zipfile.ZipFile(self.out / "share_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, arcname=p.name)
        return code


def locate_root(value: str | None) -> Path:
    markers = ("metabric_m7_config.json", "metabric_m8_config.json", "metabric_m9_config.json")
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("--root is not an existing directory")
        if not any((path / marker).is_file() for marker in markers):
            raise ValueError("--root contains none of the M7/M8/M9 configuration files")
        return path
    bases = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    for base in bases:
        for path in (base, *base.parents):
            if all((path / marker).is_file() for marker in markers):
                return path
    raise ValueError("Project root not found. Run from the existing repository or provide --root.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Existing multimodal-hte-breast-cancer directory")
    parser.add_argument("--output-dir", help="NEW report directory; must not already exist (relative to project root)")
    parser.add_argument("--self-test", action="store_true", help="Test metrics with synthetic arrays only; no project files required")
    args = parser.parse_args()
    try:
        checked = metrics_self_test()
        if args.self_test:
            print(f"PASS: {checked} synthetic tied-time/risk/censoring cases for C-index and known-status AUC.")
            return 0
        root = locate_root(args.root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out = Path(args.output_dir) if args.output_dir else Path("results/reports") / ("paper2b_audit_" + stamp)
        out = out.resolve() if out.is_absolute() else (root / out).resolve()
        if out.exists():
            raise ValueError("Output directory already exists. No files were overwritten; choose a new directory.")
        if out == root or root.is_relative_to(out):
            raise ValueError("Unsafe output directory")
        out.mkdir(parents=True, exist_ok=False)
        audit = Audit(root, out)
        audit.log("PAPER 2B / M55 / READ-ONLY AUDIT v" + VERSION)
        audit.log(f"Metric self-tests passed: {checked}; no new model fits or resampling.")
        try:
            audit.load_configs()
            audit.audit_predictions()
            audit.audit_bootstrap()
        except Exception as exc:
            safe_frames = "; ".join(f"{Path(f.filename).name}:{f.lineno}({f.name})"
                                    for f in traceback.extract_tb(exc.__traceback__)[-5:])
            audit.add("execution", "unhandled_processing_error", "FAIL", type(exc).__name__,
                      note="Audit stopped early; partial report is not a passed audit. Locations: " + safe_frames)
            # Exception text is deliberately not copied: it may contain patient values.
        return audit.finish()
    except Exception as exc:
        print("Cannot run audit: " + str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
