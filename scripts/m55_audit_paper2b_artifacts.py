#!/usr/bin/env python3
"""Paper 2B: read-only audit of the saved METABRIC OS results (M7--M9).

Place in scripts/ and run from the existing project:
    python scripts/m55_audit_paper2b_artifacts.py

Python >=3.10; dependencies: numpy, pandas. No other project modules are imported.
No models are fitted. No new bootstrap samples, datasets, or network requests are
created. Existing files are never rewritten. Only a NEW timestamped audit folder
is written, including a ZIP of aggregate audit outputs (no patient rows or IDs).

Scope: M38/M41 OS OOF predictions and checkpoints; M36/M40/M45 protocols;
M37/M46 saved bootstrap summaries; the observed-NPI restriction used by M10B.
RFS, IPCW AUC, within-fold concordance sensitivity, source-panel reconstruction,
and proof of training/test isolation are deliberately OUTSIDE this first file.
OOF labels alone cannot establish absence of leakage or prospective locking.

Exit codes: 0 = completed without detected FAIL/required NOT_VERIFIED checks;
2 = completed with findings or missing required evidence; 1 = execution failure.
Neither exit code 0 nor matching hashes certifies the whole manuscript.

Schema references (read, not executed by this script):
  Helen2813/multimodal-hte-breast-cancer, commit
  73855f6e14c1bebfaa4c92e7e9169de1742454e5:
    scripts/m38_run_track_b_full_repeated_nested.py
    scripts/m41_run_modality_specific_repeated_nested.py
    scripts/m46_bootstrap_repeated_oof_predictions.py
    scripts/m51_metabric_m10b_npi_benchmark.py
Harrell tie convention:
  https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
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
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy or pandas. Use the project's existing .venv; "
        "this script does not install or upgrade packages. " + str(exc)
    )

VERSION = "1.0.0"
ATOL = 1e-8
TABLES = (
    "checks", "file_inventory", "config_protocol_comparison", "hash_checks",
    "cohort_summary", "fold_structure", "penalizer_summary", "penalizer_pairs",
    "clinical_selection_summary", "repeat_metrics", "metric_comparisons",
    "npi_restriction_summary", "bootstrap_summary_check", "estimate_origins",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(x) for x in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def text(value: Any) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), ensure_ascii=True, sort_keys=True)
    return str(value)


def nested(obj: dict, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def harrell_scores(time: np.ndarray, event: np.ndarray,
                   risks: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, int]:
    """Exact pair enumeration; larger risk predicts shorter survival.

    Comparable ordered pair (i,j): i dies before j's recorded time, OR i dies
    at the time j is censored. Two deaths at the same time are not comparable.
    Risk ties receive 0.5. No clipping, rank normalization, or row deletion.
    Chunking limits memory use. This is evaluation only, not a Cox fit.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=float)
    r = np.asarray(risks, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if t.ndim != 1 or len(t) != len(e) or len(t) != len(r):
        raise ValueError("Inconsistent arrays for concordance.")
    if not (np.isfinite(t).all() and np.isfinite(e).all() and np.isfinite(r).all()):
        raise ValueError("Non-finite values: concordance was not computed.")
    if (t < 0).any() or not np.isin(e, [0, 1]).all():
        raise ValueError("Invalid survival outcome: concordance was not computed.")
    deaths = np.flatnonzero(e == 1)
    numerator = np.zeros(r.shape[1], dtype=float)
    denominator = 0
    for start in range(0, len(deaths), block_size):
        idx = deaths[start:start + block_size]
        ti = t[idx, None]
        comparable = ((ti < t[None, :]) |
                      ((ti == t[None, :]) & (e[None, :] == 0)))
        denominator += int(comparable.sum())
        for k in range(r.shape[1]):
            left, right = r[idx, k, None], r[None, :, k]
            numerator[k] += np.count_nonzero(comparable & (left > right))
            numerator[k] += 0.5 * np.count_nonzero(comparable & (left == right))
    if denominator == 0:
        raise ValueError("No comparable survival pairs.")
    return numerator / denominator, denominator


def find_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("--root is not a directory.")
        return root
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    for start in starts:
        for candidate in [start, *start.parents]:
            if any((candidate / f"metabric_m{i}_config.json").is_file() for i in (7, 8, 9)):
                return candidate
    raise ValueError(
        "Project root not found. Put this file in scripts/ and run from the "
        "project root, or specify --root. No old analysis was executed."
    )


class Audit:
    def __init__(self, root: Path, output: Path):
        self.root, self.output = root, output
        output.mkdir(parents=True, exist_ok=False)
        self.rows: dict[str, list[dict]] = {name: [] for name in TABLES}
        self.inventory: dict[str, dict] = {}
        self.transcript = (output / "console.log").open("x", encoding="utf-8")
        self.points: dict[tuple[str, str], dict] = {}
        self.bootstrap_means: dict[tuple[str, str], dict] = {}

    def say(self, message: str) -> None:
        # Console/report messages never include sample identifiers or patient rows.
        print(message, flush=True)
        self.transcript.write(message + "\n")
        self.transcript.flush()

    def add(self, status: str, check: str, details: str, required: bool = True) -> None:
        self.rows["checks"].append(dict(status=status, check=check,
                                        details=details, required=required))
        if status != "PASS":
            self.say(f"[{status}] {check}: {details}")

    def path(self, value: Any) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("A non-empty local path was expected.")
        p = (self.root / Path(value.replace("\\", "/"))).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Path outside the specified project root was refused.")
        return p

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def record(self, path: Path, role: str, required: bool = False) -> dict:
        key = self.rel(path)
        if key in self.inventory:
            return self.inventory[key]
        row = dict(path=key, role=role, exists=path.is_file(), size_bytes=None,
                   sha256=None, mtime_utc=None)
        if path.is_file():
            try:
                st = path.stat()
                row.update(size_bytes=st.st_size, sha256=sha256(path),
                           mtime_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat())
            except OSError as exc:
                self.add("FAIL", f"read {key}", type(exc).__name__)
        else:
            self.add("NOT_VERIFIED" if required else "INFO", f"file {key}",
                     "File is absent; no replacement was guessed.", required)
        self.inventory[key] = row
        return row

    def load_json(self, path: Path, role: str, required: bool = False) -> dict:
        self.record(path, role, required)
        if not path.is_file():
            return {}
        try:
            with path.open(encoding="utf-8-sig") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError("Top-level JSON must be an object.")
            return value
        except (OSError, ValueError) as exc:
            self.add("FAIL", f"JSON {self.rel(path)}", type(exc).__name__)
            return {}

    def load_csv(self, path: Path, role: str, required: bool = False) -> pd.DataFrame | None:
        self.record(path, role, required)
        if not path.is_file():
            return None
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            if len(header) != len(set(header)):
                raise ValueError("Duplicate column names.")
            # Preserve IDs exactly; never strip, normalize, impute, or deduplicate.
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                                dtype={"sample_id": "string"})
            return frame
        except (OSError, ValueError, StopIteration) as exc:
            self.add("FAIL", f"CSV {self.rel(path)}", type(exc).__name__)
            return None

    def compare(self, category: str, check: str, observed: Any, saved: Any,
                tolerance: float = ATOL, required: bool = True) -> None:
        try:
            a, b = float(observed), float(saved)
            valid = math.isfinite(a) and math.isfinite(b)
            difference = abs(a - b) if valid else None
            status = "PASS" if valid and difference <= tolerance else "FAIL"
        except (ValueError, TypeError):
            difference, status = None, "NOT_VERIFIED"
        self.rows[category].append(dict(check=check, computed=observed, saved=saved,
                                        absolute_difference=difference, tolerance=tolerance,
                                        status=status))
        self.add(status, check, f"computed={text(observed)}; saved={text(saved)}", required)

    def finish(self) -> int:
        self.rows["file_inventory"] = list(self.inventory.values())
        counts = Counter(row["status"] for row in self.rows["checks"])
        fails = counts["FAIL"]
        unverified = sum(row["status"] == "NOT_VERIFIED" and row["required"]
                         for row in self.rows["checks"])
        reviews = counts["REVIEW"]
        status = ("ISSUES_FOUND" if fails else "INCOMPLETE" if unverified else
                  "REVIEW_NEEDED" if reviews else "CHECKS_PASSED_WITHIN_SCOPE")
        self.say("\n" + "=" * 78)
        self.say(f"AUDIT STATUS: {status}")
        self.say(f"FAIL={fails}; required NOT_VERIFIED={unverified}; REVIEW={reviews}")
        self.say("No models fitted; no bootstrap rerun; no existing input changed.")
        self.say("This is not certification of the whole paper or of absence of leakage.")
        env = {"python": platform.python_version(), "platform": platform.system()}
        for package in ("numpy", "pandas", "lifelines", "scikit-survival"):
            try:
                env[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                env[package] = "not installed"
        result = dict(script_version=VERSION, status=status, status_counts=dict(counts),
                      environment=env, scope="OS M7--M9 and candidate M10B NPI restriction",
                      limitations=[
                          "No raw-matrix or actual training-set inspection.",
                          "No RFS audit, IPCW AUC, or within-fold sensitivity in this file.",
                          "Hashes prove agreement with recorded artifacts, not chronology.",
                          "Bootstrap summaries are checked against saved draws, not rerun.",
                          "OOF point contrasts and bootstrap means are separate quantities.",
                          "The manuscript PDF is not automatically parsed or certified."],
                      checks=json_safe(self.rows["checks"]))
        (self.output / "audit_result.json").write_text(
            json.dumps(json_safe(result), indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8")
        for name in TABLES:
            rows = self.rows[name]
            fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
            with (self.output / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: text(v) if isinstance(v, (list, dict)) else v
                                  for k, v in json_safe(row).items()} for row in rows)
        lines = ["# Paper 2B: saved-artifact audit", "", f"**Status: {status}**", "",
                 "Read-only inspection of existing OS M7--M9 artifacts; no model refitting.",
                 "No patient-level records or identifiers are exported in this report.", "",
                 "## Scope limitations", *[f"- {v}" for v in result["limitations"]], "",
                 "## Observed results (not expected or hard-coded patient counts)", ""]
        for row in self.rows["cohort_summary"]:
            lines.append(f"- {row['analysis']}/{row['modality']}: n={row['n_union']}; "
                         f"events={row['events']}; repeats={row['repeats']}; "
                         f"complete repeated population={row['same_population_all_repeats']}.")
        lines += ["", "## Findings", ""]
        for row in self.rows["checks"]:
            if row["status"] != "PASS":
                lines.append(f"- **{row['status']}** {row['check']}: {row['details']}")
        lines += ["", "## Output tables", "",
                  "- penalizer_summary / penalizer_pairs: observed M41 penalties; M38 is not inferred.",
                  "- npi_restriction_summary: how observed-NPI restriction changes OOF populations.",
                  "- repeat_metrics / metric_comparisons: recomputed pooled OOF C-indices.",
                  "- bootstrap_summary_check: saved draws vs saved summary, not fresh bootstrap.",
                  "- estimate_origins: original point estimates vs bootstrap means (not interchangeable).",
                  "- config_protocol_comparison / hash_checks: version and evidence checks.", "",
                  "Do not interpret configuration differences alone as invalid numerical results.",
                  "Archive original results. Resolve FAIL and missing required evidence before new fits."]
        (self.output / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.say(f"Output: {self.output}")
        self.say("Send back audit_bundle.zip; do NOT upload the original LOCAL_ONLY predictions.")
        self.transcript.close()
        # Explicit allow-list: never include inputs, arbitrary directory contents, or IDs.
        names = [f"{name}.csv" for name in TABLES] + ["audit_result.json", "audit_summary.md", "console.log"]
        with zipfile.ZipFile(self.output / "audit_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(self.output / name, arcname=name)
        return 2 if fails or unverified else 0


def compare_setting(a: Audit, label: str, current: Any, locked: Any) -> None:
    if locked is None:
        status = "NOT_VERIFIED"
    else:
        status = "PASS" if current == locked else "REVIEW"
    a.rows["config_protocol_comparison"].append(dict(
        setting=label, current_config=text(current), saved_protocol=text(locked), status=status))
    a.add(status, label, f"current={text(current)}; protocol={text(locked)}", required=False)


def expected_design(cfg: dict, protocol: dict, key: str) -> dict:
    p = protocol.get(key, {})
    return dict(
        repeats=p.get("outer_repeats", cfg.get("outer_repeats")),
        folds=p.get("outer_folds", cfg.get("outer_folds")),
        seeds=p.get("repeat_seeds"),
        seed_start=cfg.get("repeat_seed_start"),
        source="saved protocol" if p else "current config only",
        modalities=p.get("modalities"),
    )


def valid_integers(frame: pd.DataFrame, columns: list[str]) -> bool:
    for column in columns:
        vals = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not (np.isfinite(vals).all() and (vals > 0).all() and (vals == np.floor(vals)).all()):
            return False
        frame[column] = vals.astype(np.int64)
    return True


def require_columns(a: Audit, frame: pd.DataFrame, columns: list[str], label: str) -> bool:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        a.add("FAIL", label, "Missing columns: " + ", ".join(missing))
    return not missing


def audit_penalties(a: Audit, checkpoints: pd.DataFrame | None, sequence: Any) -> None:
    if checkpoints is None:
        return
    label = "M41 checkpoint penalties"
    cols = ["modality", "repeat", "fold", "clinical_penalizer",
            "modality_penalizer", "clinical_modality_penalizer"]
    if not require_columns(a, checkpoints, cols, label):
        return
    if checkpoints.duplicated(["modality", "repeat", "fold"]).any():
        a.add("FAIL", label, "Duplicate modality/repeat/fold rows; penalty counts not summarized.")
        return
    allowed = np.asarray(sequence, float) if isinstance(sequence, list) and sequence else None
    for modality, frame in checkpoints.groupby("modality", sort=True):
        frame = frame.copy()
        for col in cols[3:]:
            vals = pd.to_numeric(frame[col], errors="coerce")
            if not (np.isfinite(vals.to_numpy(float)).all() and (vals >= 0).all()):
                a.add("FAIL", f"{modality}/{col}", "Missing, negative, or invalid actual penalty.")
                continue
            frame[col] = vals
            for value, count in vals.value_counts().sort_index().items():
                in_config = bool(np.isclose(float(value), allowed, rtol=0, atol=1e-12).any()) if allowed is not None else None
                a.rows["penalizer_summary"].append(dict(
                    modality=modality, model=col, actual_penalizer=float(value), folds=int(count),
                    fraction=float(count / len(vals)), in_current_config_sequence=in_config,
                    evidence="m41_fold_checkpoint.csv; observed, not inferred"))
                if in_config is False:
                    a.add("REVIEW", f"{modality}/{col}",
                          f"Recorded value {value} is not in the CURRENT config sequence; inspect run version.")
        c = pd.to_numeric(frame["clinical_penalizer"], errors="coerce")
        m = pd.to_numeric(frame["clinical_modality_penalizer"], errors="coerce")
        valid = np.isfinite(c) & np.isfinite(m)
        n_different = int((~np.isclose(c[valid], m[valid], atol=1e-12, rtol=0)).sum())
        for (cp, mp), subset in frame.groupby(["clinical_penalizer", "clinical_modality_penalizer"], dropna=False):
            a.rows["penalizer_pairs"].append(dict(modality=modality, clinical_penalizer=cp,
                                                   augmented_penalizer=mp, folds=len(subset)))
        a.add("REVIEW" if n_different else "PASS", f"{modality}: paired penalties",
              f"Different recorded clinical/augmented penalties in {n_different}/{len(frame)} folds. "
              "A difference is not by itself an invalid comparison.")
    a.say("Recorded M41 penalty distributions have been summarized (not treated as tuning results).")


def npi_audit(a: Audit, frame: pd.DataFrame, clinical: pd.DataFrame | None,
              analysis: str, modality: str) -> None:
    if clinical is None:
        return
    required = ["sample_id", "npi", "os_months", "os_event"]
    if not require_columns(a, clinical, required, "clinical master for OS/NPI audit"):
        return
    if clinical["sample_id"].isna().any() or clinical["sample_id"].duplicated().any():
        a.add("FAIL", "clinical master IDs", "Missing or duplicate IDs; merge refused.")
        return
    unique = frame.drop_duplicates("sample_id")  # only after within-ID outcome consistency was checked
    merged = unique[["sample_id", "time_months", "event"]].merge(
        clinical[required], on="sample_id", how="left", validate="one_to_one", indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    tc = pd.to_numeric(merged["os_months"], errors="coerce").to_numpy(float)
    ec = pd.to_numeric(merged["os_event"], errors="coerce").to_numpy(float)
    to = merged["time_months"].to_numpy(float)
    eo = merged["event"].to_numpy(float)
    finite = np.isfinite(tc) & np.isfinite(ec)
    invalid = int((~finite).sum())
    mismatches = int((finite & ((np.abs(tc - to) > ATOL) | (ec != eo))).sum())
    a.add("PASS" if not (missing or invalid or mismatches) else "FAIL",
          f"{analysis}/{modality}: OS master agreement",
          f"IDs absent={missing}; invalid master outcomes={invalid}; outcome mismatches={mismatches}.")
    raw_present = merged["npi"].notna()
    numeric = pd.to_numeric(merged["npi"], errors="coerce")
    npi_valid = np.isfinite(numeric.to_numpy(float))
    malformed = int((raw_present & ~npi_valid).sum())
    if malformed:
        a.add("FAIL", f"{analysis}/{modality}: NPI", "Present but non-finite/nonnumeric NPI values.")
    observed_ids = set(merged.loc[raw_present, "sample_id"])
    sets = [set(g["sample_id"]) & observed_ids for _, g in frame.groupby("repeat")]
    union = set.union(*sets) if sets else set()
    common = set.intersection(*sets) if sets else set()
    a.rows["npi_restriction_summary"].append(dict(
        analysis=analysis, modality=modality, original_unique_patients=len(unique),
        missing_from_clinical_master=missing, npi_missing_after_merge=int((~raw_present).sum()),
        npi_present_but_invalid=malformed, npi_observed_union=len(union),
        excluded_by_repeat_intersection=len(union - common), npi_common_repeated_population=len(common),
        total_excluded=len(unique) - len(common),
        interpretation="Reconstructed M10B restriction; not independently matched to the M10B output registry."))
    a.say(f"{modality}: primary OOF n={len(unique)}; NPI-common n={len(common)}; "
          f"NPI-missing={int((~raw_present).sum())}; repeat-intersection loss={len(union-common)}.")


def audit_oof(a: Audit, frame: pd.DataFrame | None, checkpoints: pd.DataFrame | None,
              analysis: str, design: dict, clinical: pd.DataFrame | None) -> None:
    if frame is None:
        return
    molecular = analysis == "modality_specific"
    augmented = "clinical_modality_risk" if molecular else "model_risk"
    risk_cols = ["clinical_risk", augmented]
    cols = ["sample_id", "repeat", "fold", "time_months", "event", *risk_cols]
    if molecular:
        cols.append("modality")
    if not require_columns(a, frame, cols, f"{analysis}: OOF schema"):
        return
    frame = frame.copy()
    if frame["sample_id"].isna().any() or (frame["sample_id"].str.len() == 0).any():
        a.add("FAIL", f"{analysis}: IDs", "Missing/empty IDs; evaluation refused.")
        return
    if (frame["sample_id"] != frame["sample_id"].str.strip()).any():
        a.add("FAIL", f"{analysis}: IDs", "Whitespace in IDs; no automatic normalization was performed.")
        return
    integer_cols = ["repeat", "fold"] + (["seed"] if "seed" in frame else [])
    if not valid_integers(frame, integer_cols):
        a.add("FAIL", f"{analysis}: design", "Non-positive or non-integer repeat/fold/seed.")
        return
    if len(frame) == 0:
        a.add("FAIL", f"{analysis}: OOF population", "Empty prediction file.")
        return
    if molecular and frame["modality"].isna().any():
        a.add("FAIL", f"{analysis}: modality", "Missing modality labels.")
        return
    if molecular:
        expected_modalities = design.get("modalities")
        if not expected_modalities:
            expected_modalities = ["RNA", "CNV", "Methylation", "Mutation"]
            a.add("REVIEW", "M41 expected modalities",
                  "Saved protocol lacks modality labels; using the four documented M41 labels explicitly.")
        observed_modalities = set(frame["modality"].astype(str))
        a.add("PASS" if observed_modalities == set(expected_modalities) else "FAIL",
              "M41 modality completeness",
              f"Observed={sorted(observed_modalities)}; expected={sorted(expected_modalities)}.")
    groups = frame.groupby("modality", sort=True) if molecular else [("Multimodal", frame)]
    for modality, g in groups:
        g = g.copy()
        label = f"{analysis}/{modality}"
        if g.duplicated(["repeat", "sample_id"]).any():
            a.add("FAIL", f"{label}: OOF uniqueness", "Duplicate patient/repeat; no rows were dropped.")
            continue
        for col in ["time_months", "event", *risk_cols]:
            g[col] = pd.to_numeric(g[col], errors="coerce")
        good = (np.isfinite(g[["time_months", "event", *risk_cols]].to_numpy(float)).all()
                and g["event"].isin([0, 1]).all() and (g["time_months"] >= 0).all())
        if not good:
            a.add("FAIL", f"{label}: numerical validity",
                  "Non-finite risk/time, negative time, or event outside {0,1}; evaluation refused.")
            continue
        if (g["time_months"] == 0).any():
            a.add("REVIEW", f"{label}: zero follow-up", "Zero-time rows are present and retained, not silently removed.")
        if (g[risk_cols] <= 0).any().any():
            a.add("REVIEW", f"{label}: risk scale", "Nonpositive scores: check whether saved risks are log risks; rankings remain evaluable.")
        by_id = g.groupby("sample_id")[["time_months", "event"]].agg(["min", "max"])
        consistent = ((by_id[("time_months", "max")] - by_id[("time_months", "min")] <= ATOL).all()
                      and (by_id[("event", "max")] == by_id[("event", "min")]).all())
        if not consistent:
            a.add("FAIL", f"{label}: outcomes across repeats", "The same ID has different outcomes across repeats.")
            continue
        repeats = sorted(g["repeat"].unique().tolist())
        sets = [set(x["sample_id"]) for _, x in g.groupby("repeat")]
        union, common = set.union(*sets), set.intersection(*sets)
        same = len(union) == len(common)
        events = int(by_id[("event", "min")].sum())
        a.rows["cohort_summary"].append(dict(
            analysis=analysis, modality=modality, rows=len(g), n_union=len(union), n_common=len(common),
            events=events, repeats=len(repeats), same_population_all_repeats=same,
            expectation_source=design["source"]))
        a.add("PASS" if same else "FAIL", f"{label}: repeated population",
              f"Union n={len(union)}; intersection n={len(common)}; repeats={len(repeats)}.")
        nr, nf = design.get("repeats"), design.get("folds")
        if nr is not None:
            expected_repeats = list(range(1, int(nr) + 1))
            a.add("PASS" if repeats == expected_repeats else "FAIL", f"{label}: repeat completeness",
                  f"Observed repeats={repeats}; expected={expected_repeats} from {design['source']}.")
        else:
            a.add("NOT_VERIFIED", f"{label}: expected repeats", "No usable protocol/config setting.")
        cp = checkpoints
        if cp is not None:
            cp = cp.copy()
            needed = ["repeat", "fold"] + (["modality"] if molecular else [])
            if not require_columns(a, cp, needed, label + ": checkpoint schema"):
                cp = None
            elif not valid_integers(cp, ["repeat", "fold"]):
                a.add("FAIL", label + ": checkpoint design", "Invalid repeat/fold labels.")
                cp = None
            elif molecular:
                cp = cp.loc[cp["modality"] == modality].copy()
            if cp is not None and cp.duplicated(["repeat", "fold"]).any():
                a.add("FAIL", label + ": checkpoint uniqueness", "Duplicate repeat/fold rows.")
                cp = None
        if cp is not None:
            pkeys = set(map(tuple, g[["repeat", "fold"]].drop_duplicates().to_numpy()))
            ckeys = set(map(tuple, cp[["repeat", "fold"]].to_numpy()))
            a.add("PASS" if pkeys == ckeys else "FAIL", label + ": OOF/checkpoint folds",
                  f"OOF folds={len(pkeys)}; checkpoint folds={len(ckeys)}; symmetric difference={len(pkeys^ckeys)}.")
        values = []
        for repeat, r in g.groupby("repeat", sort=True):
            folds = sorted(r["fold"].unique().tolist())
            fold_ok = nf is not None and folds == list(range(1, int(nf) + 1))
            a.add("PASS" if fold_ok else "FAIL" if nf is not None else "NOT_VERIFIED",
                  f"{label}/repeat {repeat}: folds", f"Observed={folds}; configured/locked count={nf}.")
            expected_seeds = design.get("seeds")
            if isinstance(expected_seeds, list) and 0 < repeat <= len(expected_seeds):
                expected_seed = int(expected_seeds[repeat - 1])
            elif design.get("seed_start") is not None:
                expected_seed = int(design["seed_start"]) + repeat - 1
            else:
                expected_seed = None
            if "seed" in r and expected_seed is not None:
                a.add("PASS" if r["seed"].eq(expected_seed).all() else "FAIL", f"{label}/repeat {repeat}: seed",
                      f"Expected seed={expected_seed}; observed unique count={r['seed'].nunique()}.")
            for fold, f in r.groupby("fold", sort=True):
                row = dict(analysis=analysis, modality=modality, repeat=int(repeat), fold=int(fold),
                           test_n=len(f), test_events=int(f["event"].sum()),
                           inferred_train_n=len(r) - len(f),
                           train_membership="inferred complement only; actual training IDs not inspected")
                a.rows["fold_structure"].append(row)
                if cp is not None:
                    saved = cp.loc[(cp["repeat"] == repeat) & (cp["fold"] == fold)]
                    if len(saved) == 1:
                        for key, val in [("test_n", len(f)), ("test_events", f["event"].sum()),
                                         ("train_n", len(r) - len(f)),
                                         ("train_events", r["event"].sum() - f["event"].sum())]:
                            if key in saved:
                                a.compare("metric_comparisons", f"{label}/r{repeat}/f{fold}: {key}",
                                          int(val), saved.iloc[0][key], tolerance=0)
            try:
                c, pairs = harrell_scores(r["time_months"].to_numpy(), r["event"].to_numpy(),
                                          r[risk_cols].to_numpy())
            except ValueError:
                a.add("FAIL", f"{label}/repeat {repeat}: C-index", "No comparable pairs or invalid arrays.")
                continue
            row = dict(analysis=analysis, modality=modality, repeat=int(repeat), n=len(r),
                       events=int(r["event"].sum()), clinical_c_index=float(c[0]),
                       model_c_index=float(c[1]), delta_c_index=float(c[1] - c[0]),
                       comparable_pairs=pairs, aggregation="pooled OOF within each repeat")
            a.rows["repeat_metrics"].append(row)
            values.append(row)
        if values and len(values) == len(repeats) and same:
            point = dict(clinical_c_index=float(np.mean([x["clinical_c_index"] for x in values])),
                         model_c_index=float(np.mean([x["model_c_index"] for x in values])),
                         delta_c_index=float(np.mean([x["delta_c_index"] for x in values])))
            a.points[(analysis, str(modality))] = point
            a.say(f"{modality}: original OOF mean delta C={point['delta_c_index']:+.8f}")
        npi_audit(a, g, clinical, analysis, str(modality))
        if cp is not None and "selected_clinical" in cp and "clinical_candidates" in cp:
            for _, row in cp.iterrows():
                a.rows["clinical_selection_summary"].append(dict(
                    analysis=analysis, modality=modality, repeat=int(row["repeat"]), fold=int(row["fold"]),
                    clinical_candidates=row["clinical_candidates"], selected_clinical=row["selected_clinical"],
                    interpretation="Recorded clinical retention; selected model is not assumed to retain every clinical predictor."))


def compare_repeat_table(a: Audit, saved: pd.DataFrame | None, analysis: str) -> None:
    if saved is None:
        return
    keys = ["repeat"] + (["modality"] if analysis == "modality_specific" else [])
    if not require_columns(a, saved, keys, analysis + ": repeat summary"):
        return
    saved = saved.copy()
    if not valid_integers(saved, ["repeat"]) or saved.duplicated(keys).any():
        a.add("FAIL", analysis + ": repeat summary", "Invalid or duplicate repeat keys.")
        return
    for row in a.rows["repeat_metrics"]:
        if row["analysis"] != analysis:
            continue
        subset = saved.loc[saved["repeat"] == row["repeat"]]
        if "modality" in keys:
            subset = subset.loc[subset["modality"] == row["modality"]]
        label = f"{analysis}/{row['modality']}/r{row['repeat']}: saved repeat"
        if len(subset) != 1:
            a.add("FAIL", label, "Exactly one matching saved row was expected.")
            continue
        mapping = {"n": "n", "events": "events", "delta_c_index": "delta_c_index_vs_clinical"}
        mapping.update({"clinical_c_index": "clinical_c_index", "model_c_index": "clinical_modality_c_index"}
                       if analysis == "modality_specific" else
                       {"clinical_c_index": "clinical_only_c_index", "model_c_index": "harrell_c_index"})
        for computed, original in mapping.items():
            if original in saved:
                a.compare("metric_comparisons", label + "/" + original,
                          row[computed], subset.iloc[0][original])
            else:
                a.add("NOT_VERIFIED", label, f"Missing saved column {original}.")


def audit_bootstrap(a: Audit, draws: pd.DataFrame | None, summary: pd.DataFrame | None,
                    track: str, expected: Any) -> None:
    if draws is None:
        return
    track_a = track == "A"
    keys = ["model_set"] if track_a else ["analysis", "modality"]
    repcol = "repetition" if track_a else "bootstrap"
    metrics = ["delta_c_index_vs_clinical"] if track_a else [
        "mean_clinical_c_index", "mean_model_c_index", "delta_c_index"]
    if not require_columns(a, draws, keys + [repcol] + metrics, f"Track {track}: bootstrap draws"):
        return
    draws = draws.copy()
    if not valid_integers(draws, [repcol]):
        a.add("FAIL", f"Track {track}: bootstrap IDs", "Invalid replication labels.")
        return
    if draws[keys].isna().any().any():
        a.add("FAIL", f"Track {track}: bootstrap groups", "Missing group labels.")
        return
    groupby_key = keys[0] if len(keys) == 1 else keys
    for group, d in draws.groupby(groupby_key, sort=True):
        labels = (group,) if track_a else group
        group_dict = dict(zip(keys, labels))
        label = f"Track {track} bootstrap " + "/".join(map(str, labels))
        reps = sorted(d[repcol].unique().tolist())
        if len(reps) != len(d):
            a.add("FAIL", label, "Duplicated bootstrap iteration IDs.")
            continue
        a.add("PASS" if expected is not None and reps == list(range(1, int(expected)+1))
              else "FAIL" if expected is not None else "NOT_VERIFIED", label + ": count",
              f"Observed={len(d)}; configured/locked={expected}. Actual rows control the count.")
        s = summary
        if s is not None and set(keys + ["metric"]).issubset(s.columns):
            for key, val in group_dict.items():
                s = s.loc[s[key] == val]
        else:
            s = None
        for metric in metrics:
            numeric = pd.to_numeric(d[metric], errors="coerce").to_numpy(float)
            finite = numeric[np.isfinite(numeric)]
            if len(finite) != len(numeric):
                a.add("REVIEW", label + "/" + metric,
                      f"Finite draws={len(finite)}/{len(numeric)}; summary uses finite draws explicitly.")
            if len(finite) < 2:
                a.add("FAIL", label + "/" + metric, "Fewer than two finite draws.")
                continue
            vals = dict(mean=float(finite.mean()), sd=float(finite.std(ddof=1)),
                        median=float(np.median(finite)), ci_low=float(np.quantile(finite, .025)),
                        ci_high=float(np.quantile(finite, .975)), repetitions=len(finite))
            a.rows["bootstrap_summary_check"].append(dict(
                track=track, **group_dict, metric=metric, **vals, rows=len(d),
                evidence="recomputed from existing draws; no new resampling"))
            match = s.loc[s["metric"] == metric] if s is not None else None
            if match is None or len(match) != 1:
                a.add("NOT_VERIFIED", label + "/" + metric, "No unique matching saved summary row.")
            else:
                if not set(["mean", "ci_low", "ci_high"]).issubset(match.columns):
                    a.add("NOT_VERIFIED", label + "/" + metric,
                          "Saved summary lacks mean/ci_low/ci_high columns; schema needs review.")
                for field, value in vals.items():
                    if field in match:
                        a.compare("metric_comparisons", label + "/" + metric + "/" + field,
                                  value, match.iloc[0][field])
            if not track_a and metric == "delta_c_index":
                a.bootstrap_means[(str(group_dict["analysis"]), str(group_dict["modality"]))] = vals


def check_manifests(a: Audit, paths: list[Path]) -> None:
    for path in paths:
        df = a.load_csv(path, "saved hash manifest", required=False)
        if df is None:
            continue
        if not require_columns(a, df, ["path", "sha256"], a.rel(path)):
            continue
        for _, row in df.iterrows():
            try:
                target = a.path(str(row["path"]))
            except ValueError:
                a.rows["hash_checks"].append(dict(manifest=a.rel(path), path="OUTSIDE_ROOT",
                                                  status="NOT_VERIFIED", reason="Path not opened."))
                continue
            key = a.rel(target)
            # Only already enumerated small protocol/evaluation artifacts; never raw omics.
            entry = a.inventory.get(key)
            expected = str(row["sha256"]).strip().lower()
            if entry is None:
                status, actual = "OUT_OF_SCOPE", None
            elif not entry["exists"]:
                status, actual = "NOT_VERIFIED", None
            elif len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                status, actual = "NOT_VERIFIED", entry["sha256"]
            else:
                actual = entry["sha256"]
                status = "PASS" if actual == expected else "REVIEW"
            a.rows["hash_checks"].append(dict(manifest=a.rel(path), path=key, expected_sha256=expected,
                                              observed_sha256=actual, status=status))
            if status == "REVIEW":
                a.add("REVIEW", "historical hash " + key,
                      "Does not match this saved manifest; inspect version lineage, do not overwrite inputs.")


def check_m10_hashes(a: Audit, script: Path) -> None:
    a.record(script, "M10B historical hash anchors", required=False)
    if not script.is_file():
        return
    # Static parsing only: never import/execute the old stage or any of its functions.
    try:
        tree = ast.parse(script.read_text(encoding="utf-8-sig"))
        expected = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXPECTED_HASHES"
                                                    for t in node.targets):
                expected = ast.literal_eval(node.value)
        if not isinstance(expected, dict):
            return
        for value, digest in expected.items():
            path = a.path(value)
            entry = a.inventory.get(a.rel(path))
            actual = entry.get("sha256") if entry else None
            status = "NOT_VERIFIED" if not actual else "PASS" if actual == digest else "REVIEW"
            a.rows["hash_checks"].append(dict(manifest=a.rel(script) + ":EXPECTED_HASHES", path=a.rel(path),
                                              expected_sha256=digest, observed_sha256=actual, status=status))
            a.add(status, "M10B historical anchor " + a.rel(path),
                  "Historical version comparison only; a mismatch may reflect a later documented version.", False)
    except (OSError, SyntaxError, ValueError, TypeError):
        a.add("NOT_VERIFIED", "M10B static hash extraction", "Could not read a literal EXPECTED_HASHES mapping.", False)


def run(a: Audit) -> None:
    a.say("Paper 2B / METABRIC -- first read-only artifact audit")
    a.say("Scope: primary OS M7--M9 + the observed-NPI population restriction.")
    cfg = {i: a.load_json(a.path(f"metabric_m{i}_config.json"), f"current M{i} config", True)
           for i in (7, 8, 9)}
    m7 = a.path(cfg[7].get("output_dir", "results/tables/metabric_m7"))
    m8 = a.path(cfg[8].get("output_dir", "results/tables/metabric_m8"))
    m9 = a.path(cfg[9].get("output_dir", "results/tables/metabric_m9"))
    files8, files9 = cfg[8].get("files", {}), cfg[9].get("files", {})

    def consumer_path(mapping: dict, key: str, fallback: Path) -> Path:
        if key in mapping:
            return a.path(mapping[key])
        a.add("REVIEW", "path " + key, "Missing configured path; explicitly using documented canonical filename.")
        return fallback

    p7path = consumer_path(files8, "m7_protocol", m7 / "m36_m7_full_core_protocol.json")
    p8path = consumer_path(files9, "m8_protocol", m8 / "m40_m8_protocol.json")
    p9path = m9 / "m45_m9_protocol.json"
    p7 = a.load_json(p7path, "saved M7 protocol", True)
    p8 = a.load_json(p8path, "saved M8 protocol", True)
    p9 = a.load_json(p9path, "saved M9 protocol", True)
    for name, c, p, ck, pk in [("M7", cfg[7], p7, "track_b", "track_b"),
                               ("M8", cfg[8], p8, "modality_analysis", "modality_specific_analysis")]:
        for key in ("outer_repeats", "outer_folds"):
            compare_setting(a, name + "/" + key, nested(c, ck, key), nested(p, pk, key))
    compare_setting(a, "Track A/bootstrap repetitions", nested(cfg[7], "track_a", "bootstrap_repetitions"),
                    nested(p7, "track_a", "paired_patient_bootstrap_repetitions"))
    compare_setting(a, "M9/bootstrap repetitions", nested(cfg[9], "bootstrap", "repetitions"),
                    nested(p9, "sampling_uncertainty", "repetitions"))
    for name, settings in [("M7/track_b", cfg[7].get("track_b", {})),
                           ("M8/modality_analysis", cfg[8].get("modality_analysis", {}))]:
        for key in ("cox_penalizer", "cox_penalizer_sequence", "clinical_features", "engine", "historical_alpha"):
            if key in settings:
                a.rows["config_protocol_comparison"].append(dict(
                    setting=name + "/" + key, current_config=text(settings[key]),
                    saved_protocol="NOT_CHECKED_IN_THIS_FIELD", status="CONFIG_ONLY_NOT_ACTUAL_FIT"))
    a.add("INFO", "M38 actual penalty", "M38 checkpoints do not record per-fit penalties in the reviewed schema. "
          "Its config value will not be misreported as observed fit evidence.", False)

    op7 = consumer_path(files9, "m7_combined_predictions", m7 / "m38_oof_predictions_LOCAL_ONLY.csv")
    op8 = consumer_path(files9, "m8_modality_predictions", m8 / "m41_oof_predictions_LOCAL_ONLY.csv")
    # Follow the final inference's explicit inputs, never select by newest timestamp.
    for label, observed, configured_dir in [("M7 predictions", op7, m7), ("M8 predictions", op8, m8),
                                            ("M8 protocol", p8path, m8)]:
        a.add("PASS" if observed.parent == configured_dir else "REVIEW", label + ": path linkage",
              f"Consumed={a.rel(observed)}; current producer directory={a.rel(configured_dir)}.")
    c7 = a.load_csv(op7.parent / "m38_fold_checkpoint.csv", "M38 fold checkpoint", True)
    c8 = a.load_csv(op8.parent / "m41_fold_checkpoint.csv", "M41 fold checkpoint", True)
    o7 = a.load_csv(op7, "M38 OS OOF predictions", True)
    o8 = a.load_csv(op8, "M41 OS OOF predictions", True)
    masterpath = consumer_path(files8, "clinical_master",
                               a.path("results/tables/metabric_m2/m06_metabric_clinical_master_LOCAL_ONLY.csv"))
    clinical = a.load_csv(masterpath, "clinical master for OS/NPI reconciliation", True)
    audit_penalties(a, c8, nested(cfg[8], "modality_analysis", "cox_penalizer_sequence"))
    audit_oof(a, o7, c7, "combined_reconstructed", expected_design(cfg[7].get("track_b", {}), p7, "track_b"), clinical)
    audit_oof(a, o8, c8, "modality_specific", expected_design(cfg[8].get("modality_analysis", {}), p8,
                                                           "modality_specific_analysis"), clinical)
    repeat7 = a.load_csv(op7.parent / "m38_repeat_level_oof_results.csv", "saved M38 repeat metrics", True)
    # M41's exact repeat filename is obtained from its existing script, not guessed.
    script41 = a.path("scripts/m41_run_modality_specific_repeated_nested.py")
    a.record(script41, "M41 stage source", False)
    repeat8_path = op8.parent / "m41_repeat_level_results.csv"
    if script41.is_file():
        import re
        source = script41.read_text(encoding="utf-8-sig")
        names = sorted(set(re.findall(r'[\'"](m41_[A-Za-z0-9_]*repeat[A-Za-z0-9_]*\.csv)[\'"]', source)))
        # In the reviewed code the repeat table is m41_repeat_level_results.csv.
        if len(names) == 1:
            repeat8_path = op8.parent / names[0]
        elif names and repeat8_path.name not in names:
            a.add("REVIEW", "M41 repeat-table filename", "Ambiguous script filenames; documented filename retained explicitly.")
    repeat8 = a.load_csv(repeat8_path, "saved M41 repeat metrics", True)
    compare_repeat_table(a, repeat7, "combined_reconstructed")
    compare_repeat_table(a, repeat8, "modality_specific")

    dA = a.load_csv(m7 / "m37_track_a_paired_deltas_1000.csv", "saved Track A bootstrap deltas", False)
    sA = a.load_csv(consumer_path(files9, "m7_track_a_deltas", m7 / "m37_track_a_paired_delta_summary.csv"),
                    "saved Track A delta summary", True)
    bA = nested(p7, "track_a", "paired_patient_bootstrap_repetitions")
    audit_bootstrap(a, dA, sA, "A", bA if bA is not None else nested(cfg[7], "track_a", "bootstrap_repetitions"))
    if dA is None:
        a.add("NOT_VERIFIED", "Track A executed bootstrap count", "Config/summary alone does not confirm the completed saved draws.", True)
    dB = a.load_csv(m9 / "m46_oof_patient_bootstrap_2000.csv", "saved M46 bootstrap draws", True)
    sB = a.load_csv(m9 / "m46_oof_patient_bootstrap_summary.csv", "saved M46 bootstrap summary", True)
    bB = nested(p9, "sampling_uncertainty", "repetitions")
    expected_labels = nested(p8, "modality_specific_analysis", "modalities") or ["RNA", "CNV", "Methylation", "Mutation"]
    if dB is not None and {"analysis", "modality"}.issubset(dB.columns):
        observed_groups = set(map(tuple, dB[["analysis", "modality"]].drop_duplicates().to_numpy()))
        expected_groups = {("modality_specific", label) for label in expected_labels}
        expected_groups.add(("combined_reconstructed", "Multimodal"))
        a.add("PASS" if observed_groups == expected_groups else "FAIL", "M46 bootstrap group completeness",
              f"Observed groups={len(observed_groups)}; expected={len(expected_groups)}; "
              f"missing={len(expected_groups-observed_groups)}; extra={len(observed_groups-expected_groups)}.")
    audit_bootstrap(a, dB, sB, "B", bB if bB is not None else nested(cfg[9], "bootstrap", "repetitions"))
    for key in sorted(set(a.points) | set(a.bootstrap_means)):
        point, boot = a.points.get(key, {}), a.bootstrap_means.get(key, {})
        delta = point.get("delta_c_index")
        mean = boot.get("mean")
        a.rows["estimate_origins"].append(dict(
            analysis=key[0], modality=key[1], original_mean_oof_delta=delta,
            saved_bootstrap_mean_delta=mean, bootstrap_ci_low=boot.get("ci_low"),
            bootstrap_ci_high=boot.get("ci_high"),
            bootstrap_minus_original=(mean - delta) if delta is not None and mean is not None else None,
            interpretation="Different statistical summaries, not automatically an error. Check which is printed in the manuscript."))
    manifests = [m7 / "m36_input_hash_manifest.csv", m8 / "m40_input_hash_manifest.csv",
                 m9 / "m45_input_hash_manifest.csv"]
    check_manifests(a, manifests)
    check_m10_hashes(a, a.path("scripts/m51_metabric_m10b_npi_benchmark.py"))
    # Detect a concurrent producer overwriting files during the audit.
    changed = 0
    for relative, entry in list(a.inventory.items()):
        if entry["exists"] and entry["sha256"]:
            p = a.path(relative)
            try:
                unchanged = p.is_file() and sha256(p) == entry["sha256"]
            except OSError:
                unchanged = False
            if not unchanged:
                changed += 1
                a.add("FAIL", "input changed during audit " + relative,
                      "Concurrent modification/removal detected; this run is not a consistent snapshot.")
    a.add("PASS" if changed == 0 else "FAIL", "input snapshot stability",
          f"Files changed between first and final hash check: {changed}.")
    a.add("INFO", "Next stage", "IPCW AUC, within-fold concordance sensitivity, RFS, new fits and random panels were NOT run.", False)
    a.add("INFO", "Training isolation", "Unique OOF IDs and fold counts do not prove which patients trained each model. "
          "Actual train/test membership requires separate evidence.", False)
    a.add("INFO", "Bootstrap inference", "Agreement with saved conditional-bootstrap summaries does not validate "
          "full-pipeline confidence-interval coverage or account for model-development uncertainty.", False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Existing multimodal-hte-breast-cancer project directory.")
    parser.add_argument("--out", help="NEW output directory (relative to project root or absolute). Must not exist.")
    args = parser.parse_args()
    audit = None
    try:
        root = find_root(args.root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
        output = ((root / args.out).resolve() if args.out else
                  root / "results" / "reports" / f"paper2b_audit_{stamp}")
        if output.exists():
            raise ValueError("Output already exists. No overwrite is allowed; choose a new directory or omit --out.")
        # Never place audit outputs among frozen input artifacts.
        forbidden = [root / "data", root / "scripts", root / "results" / "tables",
                     root / "configs", root / ".git", root / ".venv"]
        if any(output.is_relative_to(p.resolve()) for p in forbidden):
            raise ValueError("Output inside an input/code directory was refused. Use results/reports or a new external directory.")
        audit = Audit(root, output)
        run(audit)
        return audit.finish()
    except KeyboardInterrupt:
        if audit is not None:
            audit.add("NOT_VERIFIED", "execution", "Interrupted; this is an incomplete audit, not a negative scientific result.")
            audit.finish()
        return 1
    except Exception as exc:
        if audit is not None:
            # Do not serialize exception payloads that might contain patient rows/IDs.
            import traceback
            frames = traceback.extract_tb(exc.__traceback__)
            location = "; ".join(f"{Path(f.filename).name}:{f.lineno} ({f.name})" for f in frames[-3:])
            audit.add("FAIL", "execution", f"Unexpected {type(exc).__name__}; audit incomplete. "
                      f"Location: {location}. No original file was changed.")
            audit.finish()
        else:
            print(f"Cannot start audit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
