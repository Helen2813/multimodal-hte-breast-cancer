from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def root_dir() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    return json.loads((root / "paper_a_stage24_config.json").read_text(encoding="utf-8"))


def out_dir(root: Path, cfg: dict) -> Path:
    path = root / cfg["output_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = list(fieldnames or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
        return
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> None:
    if not rows:
        print("<empty>")
        return
    shown = list(rows if max_rows is None else rows[:max_rows])
    widths = {c: len(c) for c in columns}
    rendered = []
    for row in shown:
        item = {c: str(row.get(c, "")) for c in columns}
        rendered.append(item)
        for c in columns:
            widths[c] = min(60, max(widths[c], len(item[c])))
    print("  ".join(c[: widths[c]].ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rendered:
        print("  ".join(row[c][: widths[c]].ljust(widths[c]) for c in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows)-max_rows} additional rows written to CSV")


def normalize_tcga_id(value: object) -> str:
    text = str(value).strip().upper().replace("_", "-")
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", text)
    return match.group(1) if match else text


def flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            out.extend(flatten_strings(key))
            out.extend(flatten_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(flatten_strings(item))
    return out


def binary_series(series: pd.Series) -> pd.Series | None:
    if series.dtype == bool:
        return series.astype(int)
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0", "yes": "1", "no": "0", "treated": "1", "control": "0"})
    )
    numeric = pd.to_numeric(normalized, errors="coerce")
    valid = numeric.dropna()
    if len(valid) == 0 or not set(valid.unique()).issubset({0, 1}):
        return None
    return numeric


def best_id_column(frame: pd.DataFrame) -> tuple[str, list[dict]]:
    rows = []
    for position, column in enumerate(frame.columns):
        sample = frame[column].dropna().astype(str).head(1000)
        if len(sample) == 0:
            continue
        frac = sample.str.contains(r"TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}", case=False, regex=True).mean()
        unique = sample.nunique() / max(1, len(sample))
        name = re.sub(r"[^A-Z0-9]+", "_", str(column).upper()).strip("_")
        bonus = 50 if name in {"PATIENT_ID", "CASE_ID", "SUBMITTER_ID", "SAMPLE", "UNNAMED_0"} else 0
        if any(token in name for token in ("PATIENT", "CASE", "SUBMITTER")):
            bonus += 20
        score = 200 * float(frac) + 20 * float(unique) + bonus - 0.001 * position
        rows.append({"column": column, "score": score, "tcga_fraction": float(frac), "uniqueness": float(unique)})
    rows.sort(key=lambda row: (-row["score"], str(row["column"])))
    if not rows:
        raise RuntimeError("No patient ID column found.")
    return str(rows[0]["column"]), rows


def binary_candidates(frame: pd.DataFrame, expected_sum: int, tokens: Sequence[str]) -> list[dict]:
    rows = []
    for column in frame.columns:
        parsed = binary_series(frame[column])
        if parsed is None:
            continue
        total = int(parsed.fillna(0).sum())
        token_score = sum(token in str(column).lower() for token in tokens)
        rows.append({
            "column": column,
            "sum": total,
            "exact_sum": total == expected_sum,
            "score": 1000 * int(total == expected_sum) + 25 * token_score - abs(total - expected_sum),
        })
    rows.sort(key=lambda row: (-row["score"], str(row["column"])))
    return rows


def candidate_paths(root: Path, cfg: dict) -> list[Path]:
    paths: set[Path] = set()
    manifest = root / cfg["candidate_v9_manifest"]
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for value in flatten_strings(data):
            if str(value).lower().endswith(".csv"):
                path = Path(value)
                if not path.is_absolute():
                    path = root / path
                if path.exists():
                    paths.add(path.resolve())
    for relative in cfg["candidate_search_roots"]:
        base = root / relative
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            if path.stat().st_size > 50 * 1024 * 1024:
                continue
            name = path.as_posix().lower()
            if any(token.lower() in name for token in cfg["candidate_name_tokens"]):
                paths.add(path.resolve())
    return sorted(paths)


def discover_v9_cohort(root: Path, cfg: dict, output: Path) -> dict:
    expected = cfg["expected_cohort"]
    manifest = root / cfg["candidate_v9_manifest"]
    if not manifest.exists():
        raise FileNotFoundError(f"Missing Candidate V9 manifest: {manifest}")
    write_csv(output / "s24_v9_lock_summary.csv", [{
        "manifest_path": rel(root, manifest), "manifest_sha256": sha256(manifest),
        "manifest_size_bytes": manifest.stat().st_size, "candidate_v9_modified": False,
    }])
    rows = []
    accepted = []
    for path in candidate_paths(root, cfg):
        try:
            frame = pd.read_csv(path, low_memory=False)
            id_col, _ = best_id_column(frame)
            t = binary_candidates(frame, int(expected["treated"]), ("treat", "initiated", "early", "day180", "a"))
            e = binary_candidates(frame, int(expected["events"]), ("event", "death", "status", "outcome"))
            t_col = t[0]["column"] if t else ""
            e_col = e[0]["column"] if e else ""
            t_sum = int(binary_series(frame[t_col]).fillna(0).sum()) if t_col else -1
            e_sum = int(binary_series(frame[e_col]).fillna(0).sum()) if e_col else -1
            unique_ids = int(frame[id_col].map(normalize_tcga_id).nunique())
            score = 5000 * int(len(frame) == int(expected["n"])) + 3000 * int(t_sum == int(expected["treated"])) + 1000 * int(e_sum == int(expected["events"])) + 200 * int(unique_ids == len(frame))
            row = {"path": rel(root, path), "rows": len(frame), "columns": len(frame.columns), "id_column": id_col,
                   "unique_ids": unique_ids, "treated_column": t_col, "treated_sum": t_sum,
                   "event_column": e_col, "event_sum": e_sum, "score": score, "sha256": sha256(path)}
            rows.append(row)
            if len(frame) == int(expected["n"]) and t_sum == int(expected["treated"]) and unique_ids == len(frame):
                accepted.append(row)
        except Exception as error:
            rows.append({"path": rel(root, path), "score": -1, "error": f"{type(error).__name__}: {error}"})
    rows.sort(key=lambda row: (-int(row.get("score", -1)), row.get("path", "")))
    write_csv(output / "s24_cohort_discovery_candidates.csv", rows)
    print("Top Candidate V9 cohort candidates")
    print_table(rows, ["path", "rows", "columns", "id_column", "unique_ids", "treated_column", "treated_sum", "event_column", "event_sum", "score"], max_rows=30)
    if not accepted:
        raise RuntimeError("No exact Candidate V9 cohort source discovered. Review s24_cohort_discovery_candidates.csv.")
    accepted.sort(key=lambda row: (-int(row["score"]), row["path"]))
    selected = accepted[0]
    registry = {
        "selected_cohort_path": selected["path"], "selected_cohort_sha256": selected["sha256"],
        "patient_id_column": selected["id_column"], "treatment_column": selected["treated_column"],
        "event_column": selected["event_column"], "n": selected["rows"], "treated": selected["treated_sum"],
        "control": int(selected["rows"]) - int(selected["treated_sum"]), "events": selected["event_sum"],
    }
    (output / "s24_selected_v9_cohort.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print("\nSelected Candidate V9 cohort")
    print(json.dumps(registry, indent=2))
    return registry


def find_clinical_columns(frame: pd.DataFrame) -> dict:
    columns = list(frame.columns)
    patient = next((c for c in columns if str(c).lower() == "cases.submitter_id"), None)
    if patient is None:
        patient = next((c for c in columns if "submitter_id" in str(c).lower() and "case" in str(c).lower()), None)
    start = next((c for c in columns if "days_to_treatment_start" in str(c).lower()), None)
    end = next((c for c in columns if "days_to_treatment_end" in str(c).lower()), None)
    text_cols = [c for c in columns if any(token in str(c).lower() for token in (
        "treatment_type", "therapy_type", "treatment_name", "drug_name", "pharmaceutical_therapy_drug_name", "regimen"))]
    if not patient or not start or not text_cols:
        raise RuntimeError(f"Clinical treatment columns unresolved: patient={patient}, start={start}, text={text_cols}")
    return {"patient": patient, "start": start, "end": end or "", "text": text_cols}


def classify_family(text: str) -> str:
    value = str(text).strip().lower()
    patterns = {
        "hormone": (r"\bhorm", r"\bendocr", r"tamox", r"anastro", r"letroz", r"exemestan", r"fulvestr", r"goserelin", r"leuprol", r"aromatase", r"ovarian suppression"),
        "chemo": (r"\bchemo", r"cyclophosph", r"doxorub", r"adriamycin", r"paclitax", r"docetax", r"carboplatin", r"cisplatin", r"capecitab", r"gemcitab", r"methotrex", r"fluorourac", r"epirub", r"\bcmf\b", r"\bfec\b", r"\bac\b", r"\btaxane"),
        "targeted": (r"\btarget", r"trastuz", r"herceptin", r"pertuz", r"lapatin", r"bevaciz"),
        "radiation": (r"radiat", r"radiotherap"),
    }
    for family, family_patterns in patterns.items():
        if any(re.search(pattern, value) for pattern in family_patterns):
            return family
    return "other_or_unknown"


def valid_day(value: object, low: float, high: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric) or numeric < low or numeric > high:
        return float("nan")
    return float(numeric)


def patient_sequence(patient_id: str, group: pd.DataFrame, landmark: float) -> dict:
    chemo = group[group["__family"] == "chemo"].copy()
    hormone = group[group["__family"] == "hormone"].copy()
    chemo_starts = chemo["__start_day"].dropna()
    hormone_starts = hormone["__start_day"].dropna()
    earliest_chemo = float(chemo_starts.min()) if len(chemo_starts) else float("nan")
    earliest_hormone = float(hormone_starts.min()) if len(hormone_starts) else float("nan")
    pre = chemo[chemo["__start_day"].notna() & (chemo["__start_day"] <= landmark)]
    after = chemo[chemo["__start_day"].notna() & (chemo["__start_day"] > landmark)]
    ongoing = pre[pre["__end_day"].notna() & (pre["__end_day"] >= landmark)]
    unknown_end = pre[pre["__end_day"].isna()]
    completed = pre[pre["__end_day"].notna() & (pre["__end_day"] < landmark)]
    if len(chemo) == 0:
        category = "no_recorded_chemo"
    elif len(chemo_starts) == 0:
        category = "chemo_start_missing_or_invalid"
    elif len(pre) == 0:
        category = "chemo_starts_after_day180_only"
    elif len(ongoing) > 0:
        category = "chemo_known_ongoing_at_day180"
    elif len(unknown_end) > 0:
        category = "chemo_started_before_day180_end_unknown"
    elif len(completed) == len(pre):
        category = "chemo_completed_before_day180"
    else:
        category = "chemo_interval_conflict"
    if len(chemo) == 0:
        sequence = "no_recorded_chemo"
    elif not np.isfinite(earliest_chemo):
        sequence = "chemo_timing_missing_or_invalid"
    elif not np.isfinite(earliest_hormone):
        sequence = "hormone_start_not_recorded"
    elif earliest_chemo < earliest_hormone:
        sequence = "chemo_started_before_hormone"
    elif earliest_hormone < earliest_chemo:
        sequence = "hormone_started_before_chemo"
    else:
        sequence = "chemo_and_hormone_same_day"
    return {
        "patient_id": patient_id, "chemo_records": len(chemo), "hormone_records": len(hormone),
        "chemo_records_valid_start": int(chemo["__start_day"].notna().sum()),
        "hormone_records_valid_start": int(hormone["__start_day"].notna().sum()),
        "chemo_records_negative_start": int(chemo["__start_negative"].sum()),
        "hormone_records_negative_start": int(hormone["__start_negative"].sum()),
        "earliest_chemo_start_day": earliest_chemo, "earliest_hormone_start_day": earliest_hormone,
        "any_recorded_chemo": int(len(chemo) > 0), "any_chemo_start_by_day180": int(len(pre) > 0),
        "any_chemo_start_after_day180": int(len(after) > 0), "chemo_known_ongoing_at_day180": int(len(ongoing) > 0),
        "chemo_pre180_end_unknown": int(len(unknown_end) > 0),
        "chemo_completed_before_day180": int(len(pre) > 0 and len(completed) == len(pre) and len(unknown_end) == 0),
        "chemo_before_hormone_start": int(np.isfinite(earliest_chemo) and np.isfinite(earliest_hormone) and earliest_chemo < earliest_hormone),
        "hormone_before_chemo_start": int(np.isfinite(earliest_chemo) and np.isfinite(earliest_hormone) and earliest_hormone < earliest_chemo),
        "complete_chemo_hormone_start_dates": int(np.isfinite(earliest_chemo) and np.isfinite(earliest_hormone)),
        "chemo_landmark_category": category, "chemo_hormone_sequence": sequence,
    }


def build_registry(root: Path, cfg: dict, output: Path, selected: dict) -> pd.DataFrame:
    cohort = pd.read_csv(root / selected["selected_cohort_path"], low_memory=False)
    cohort["patient_id"] = cohort[selected["patient_id_column"]].map(normalize_tcga_id)
    cohort["early_hormone_by_day180"] = binary_series(cohort[selected["treatment_column"]]).astype(int)
    cohort["post_landmark_event"] = binary_series(cohort[selected["event_column"]]) if selected["event_column"] else np.nan
    clinical_path = root / cfg["clinical_source"]
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    columns = find_clinical_columns(clinical)
    clinical["patient_id"] = clinical[columns["patient"]].map(normalize_tcga_id)
    clinical["__text"] = clinical[columns["text"]].fillna("").astype(str).apply(lambda row: " | ".join(v for v in row if v and v != "nan"), axis=1)
    clinical["__family"] = clinical["__text"].map(classify_family)
    low, high = float(cfg["valid_timing"]["minimum_day"]), float(cfg["valid_timing"]["maximum_day"])
    raw_start = pd.to_numeric(clinical[columns["start"]], errors="coerce")
    clinical["__start_negative"] = raw_start.notna() & (raw_start < 0)
    clinical["__start_day"] = raw_start.map(lambda x: valid_day(x, low, high))
    if columns["end"]:
        raw_end = pd.to_numeric(clinical[columns["end"]], errors="coerce")
        clinical["__end_day"] = raw_end.map(lambda x: valid_day(x, low, high))
    else:
        clinical["__end_day"] = np.nan
    write_csv(output / "s24_source_registry.csv", [{
        "clinical_source": rel(root, clinical_path), "clinical_rows": len(clinical),
        "patient_id_column": columns["patient"], "treatment_start_column": columns["start"],
        "treatment_end_column": columns["end"], "treatment_text_columns": " | ".join(columns["text"]),
        "cohort_source": selected["selected_cohort_path"], "cohort_treatment_column": selected["treatment_column"],
        "cohort_event_column": selected["event_column"],
    }])
    family_counts = clinical.groupby("__family").agg(records=("__family", "size"), patients=("patient_id", "nunique")).reset_index().rename(columns={"__family": "treatment_family"})
    write_csv(output / "s24_treatment_family_counts.csv", family_counts.to_dict("records"))
    clinical = clinical[clinical["patient_id"].isin(set(cohort["patient_id"]))]
    rows = [patient_sequence(pid, group, float(cfg["landmark_day"])) for pid, group in clinical.groupby("patient_id", sort=True)]
    sequence = pd.DataFrame(rows)
    merged = cohort[["patient_id", "early_hormone_by_day180", "post_landmark_event"]].merge(sequence, on="patient_id", how="left", validate="one_to_one")
    integer_columns = ["chemo_records", "hormone_records", "chemo_records_valid_start", "hormone_records_valid_start", "chemo_records_negative_start", "hormone_records_negative_start", "any_recorded_chemo", "any_chemo_start_by_day180", "any_chemo_start_after_day180", "chemo_known_ongoing_at_day180", "chemo_pre180_end_unknown", "chemo_completed_before_day180", "chemo_before_hormone_start", "hormone_before_chemo_start", "complete_chemo_hormone_start_dates"]
    for column in integer_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype(int)
    landmark = float(cfg["landmark_day"])
    merged["hormone_assignment_timing_consistent"] = (((merged["early_hormone_by_day180"] == 1) & merged["earliest_hormone_start_day"].between(0, landmark, inclusive="both")) | ((merged["early_hormone_by_day180"] == 0) & (merged["earliest_hormone_start_day"].isna() | (merged["earliest_hormone_start_day"] > landmark)))).astype(int)
    merged.to_csv(output / "s24_patient_sequence_registry_LOCAL_ONLY.csv", index=False, encoding="utf-8-sig")
    checks = [
        {"check": "rows", "observed": len(merged), "expected": cfg["expected_cohort"]["n"], "pass": len(merged) == int(cfg["expected_cohort"]["n"])},
        {"check": "treated", "observed": int(merged["early_hormone_by_day180"].sum()), "expected": cfg["expected_cohort"]["treated"], "pass": int(merged["early_hormone_by_day180"].sum()) == int(cfg["expected_cohort"]["treated"])},
        {"check": "hormone assignment timing consistent", "observed": int(merged["hormone_assignment_timing_consistent"].sum()), "expected": len(merged), "pass": bool(merged["hormone_assignment_timing_consistent"].all())},
    ]
    write_csv(output / "s24_registry_checks.csv", checks)
    print("\nRegistry checks")
    print_table(checks, ["check", "observed", "expected", "pass"])
    if not all(bool(r["pass"]) for r in checks):
        raise RuntimeError("Sequence registry consistency checks failed.")
    return merged


def smd_binary(treated: pd.Series, control: pd.Series) -> float:
    t = pd.to_numeric(treated, errors="coerce").dropna(); c = pd.to_numeric(control, errors="coerce").dropna()
    if len(t) == 0 or len(c) == 0:
        return float("nan")
    pt, pc = float(t.mean()), float(c.mean())
    var = (pt * (1 - pt) + pc * (1 - pc)) / 2
    if var <= 0:
        return 0.0 if pt == pc else float("inf")
    return (pt - pc) / math.sqrt(var)


def fisher_p(treated: pd.Series, control: pd.Series) -> float:
    t = pd.to_numeric(treated, errors="coerce").dropna().astype(int); c = pd.to_numeric(control, errors="coerce").dropna().astype(int)
    if len(t) == 0 or len(c) == 0:
        return float("nan")
    table = [[int((t == 1).sum()), int((t == 0).sum())], [int((c == 1).sum()), int((c == 0).sum())]]
    return float(stats.fisher_exact(table)[1])


def audit(frame: pd.DataFrame, cfg: dict, output: Path) -> dict:
    thresholds = cfg["review_thresholds"]
    variables = ["any_recorded_chemo", "any_chemo_start_by_day180", "any_chemo_start_after_day180", "chemo_known_ongoing_at_day180", "chemo_pre180_end_unknown", "chemo_completed_before_day180", "chemo_before_hormone_start", "hormone_before_chemo_start", "complete_chemo_hormone_start_dates"]
    treated = frame[frame["early_hormone_by_day180"] == 1]; control = frame[frame["early_hormone_by_day180"] == 0]
    rows = []
    for variable in variables:
        tf = pd.to_numeric(treated[variable], errors="coerce"); cf = pd.to_numeric(control[variable], errors="coerce")
        smd = smd_binary(tf, cf)
        rows.append({"variable": variable, "treated_fraction": float(tf.mean()), "control_fraction": float(cf.mean()),
                     "risk_difference": float(tf.mean() - cf.mean()), "smd": smd, "abs_smd": abs(smd),
                     "fisher_p": fisher_p(tf, cf), "smd_flag": abs(smd) >= float(thresholds["binary_smd_flag"]),
                     "smd_severe": abs(smd) >= float(thresholds["binary_smd_severe"])})
    write_csv(output / "s24_binary_sequence_imbalance.csv", rows)
    category_rows = []
    for category_col, output_name in (("chemo_landmark_category", "s24_chemo_landmark_categories.csv"), ("chemo_hormone_sequence", "s24_chemo_hormone_sequence_categories.csv")):
        current = []
        for group in (0, 1):
            subset = frame[frame["early_hormone_by_day180"] == group]
            counts = subset[category_col].fillna("missing").value_counts()
            for category, count in counts.items():
                current.append({"group": group, "category": str(category), "count": int(count), "group_n": len(subset), "fraction": float(count / len(subset))})
        write_csv(output / output_name, current)
        category_rows.extend(current)
    masks = {
        "all_candidate_v9": pd.Series(True, index=frame.index),
        "no_recorded_chemotherapy": frame["any_recorded_chemo"] == 0,
        "no_chemo_start_by_day180": frame["any_chemo_start_by_day180"] == 0,
        "chemo_started_by_day180": frame["any_chemo_start_by_day180"] == 1,
        "chemo_completed_before_day180": frame["chemo_completed_before_day180"] == 1,
        "chemo_known_ongoing_at_day180": frame["chemo_known_ongoing_at_day180"] == 1,
        "complete_chemo_and_hormone_start_dates": frame["complete_chemo_hormone_start_dates"] == 1,
        "chemo_started_before_hormone": frame["chemo_before_hormone_start"] == 1,
        "hormone_started_before_chemo": frame["hormone_before_chemo_start"] == 1,
    }
    populations = []
    for name, mask in masks.items():
        subset = frame[mask]; tr = int(subset["early_hormone_by_day180"].sum()); co = len(subset) - tr
        events = int(pd.to_numeric(subset["post_landmark_event"], errors="coerce").fillna(0).sum())
        feasible = min(tr, co) >= int(thresholds["minimum_arm_for_sensitivity"]) and events >= int(thresholds["minimum_events_for_sensitivity"])
        populations.append({"population": name, "n": len(subset), "treated": tr, "control": co, "events": events,
                            "minimum_arm": min(tr, co), "candidate_for_effect_sensitivity": feasible})
    write_csv(output / "s24_sensitivity_population_feasibility.csv", populations)
    print("\nBinary sequencing imbalance")
    print_table(rows, ["variable", "treated_fraction", "control_fraction", "risk_difference", "smd", "fisher_p", "smd_flag", "smd_severe"])
    print("\nPotential sensitivity populations")
    print_table(populations, ["population", "n", "treated", "control", "events", "minimum_arm", "candidate_for_effect_sensitivity"])
    max_smd = float(max(row["abs_smd"] for row in rows))
    severe = [row["variable"] for row in rows if row["smd_severe"]]
    flagged = [row["variable"] for row in rows if row["smd_flag"]]
    mismatches = int((frame["hormone_assignment_timing_consistent"] != 1).sum())
    unknown_fraction = float(frame["chemo_pre180_end_unknown"].mean())
    feasible = [row["population"] for row in populations if row["candidate_for_effect_sensitivity"]]
    critical = mismatches > 0 or len(severe) > 0
    review = max_smd >= float(thresholds["binary_smd_flag"]) or unknown_fraction >= float(thresholds["missingness_flag"])
    if critical:
        status = "SEQUENCING_AUDIT_IDENTIFIES_CRITICAL_ISSUE_DO_NOT_WRITE_PRIMARY_RESULTS"
        next_action = "Create a transparent Candidate V10 protocol amendment before any new primary effect analysis."
    elif review:
        status = "SEQUENCING_AUDIT_FLAGS_POTENTIAL_CONFOUNDING_RUN_PRESPECIFIED_SENSITIVITY_EFFECTS"
        next_action = "Keep Candidate V9 locked and unchanged; run a separate sequencing-sensitivity effect package in feasible populations, then decide V9 versus V10."
    else:
        status = "SEQUENCING_AUDIT_LOW_CONCERN_V9_PRIMARY_WITH_KEY_SENSITIVITY"
        next_action = "Retain Candidate V9 as primary and add one feasible prespecified sequencing sensitivity analysis."
    decision = {"status": status, "candidate_v9_modified": False, "candidate_v9_effect_reestimated": False,
                "max_abs_sequence_smd": max_smd, "flagged_sequence_variables": flagged,
                "severe_sequence_variables": severe, "assignment_timing_mismatches": mismatches,
                "pre180_chemo_end_unknown_fraction": unknown_fraction,
                "patients_with_negative_chemo_start_records": int((frame["chemo_records_negative_start"] > 0).sum()),
                "feasible_sensitivity_populations": feasible, "next_action": next_action,
                "interpretation_boundary": "Diagnostic only. No ever-chemotherapy variable is added to the locked adjustment set."}
    (output / "s24_chemo_sequencing_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main() -> int:
    root = root_dir(); cfg = load_config(root); output = out_dir(root, cfg)
    print("=" * 124)
    print("PAPER A STAGE 24 - CHEMOTHERAPY SEQUENCING AUDIT")
    print("=" * 124)
    print("Candidate V9 is read-only. No treatment-effect model is fitted.")
    selected = discover_v9_cohort(root, cfg, output)
    frame = build_registry(root, cfg, output, selected)
    decision = audit(frame, cfg, output)
    output_files = [path for path in output.glob("s24_*") if path.is_file() and "LOCAL_ONLY" not in path.name]
    write_csv(output / "s24_output_hashes.csv", [{"path": rel(root, path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sorted(output_files)])
    print("\nStage 24 decision")
    print(json.dumps(decision, indent=2))
    print("\nPASS: sequencing audit completed. Candidate V9 remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
