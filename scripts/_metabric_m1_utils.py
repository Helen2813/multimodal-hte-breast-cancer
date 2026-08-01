from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m1_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def out_dir(root: Path, config: dict) -> Path:
    path = (root / config["outputs_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root: Path, config: dict) -> Path:
    path = (root / config["raw_dir"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"METABRIC raw directory not found: {path}")
    return path


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def quick_fingerprint(path: Path, block_bytes: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with path.open("rb") as f:
        first = f.read(block_bytes)
        h.update(first)
        if size > block_bytes:
            f.seek(max(0, size - block_bytes))
            h.update(f.read(block_bytes))
    return h.hexdigest()


def read_first_noncomment_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                return line.rstrip("\r\n")
    return ""


def infer_delimiter(line: str) -> str:
    candidates = [("\t", line.count("\t")), (",", line.count(",")), (";", line.count(";"))]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates[0][1] > 0 else "\t"


def count_lines(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> int:
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            count += chunk.count(b"\n")
    return count


def read_cbio_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
        na_values=["", "NA", "N/A", "NaN", "nan", "[Not Available]", "Unknown"],
        keep_default_na=True,
    )


def norm_col(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


def choose_column(columns: Sequence[str], exact: Sequence[str], contains: Sequence[str] = (),
                  excludes: Sequence[str] = ()) -> str | None:
    scored: list[tuple[int, str]] = []
    exact_norm = [norm_col(x) for x in exact]
    contains_norm = [norm_col(x) for x in contains]
    excludes_norm = [norm_col(x) for x in excludes]
    for col in columns:
        n = norm_col(col)
        if any(token and token in n for token in excludes_norm):
            continue
        score = 0
        if n in exact_norm:
            score += 1000 - exact_norm.index(n)
        for token in contains_norm:
            if token and token in n:
                score += 40
        if any(n.startswith(token) for token in contains_norm if token):
            score += 20
        if score:
            scored.append((score, col))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    return scored[0][1]


def is_identifier_column(name: str) -> bool:
    n = norm_col(name)
    return n in {"PATIENT_ID", "SAMPLE_ID", "CASE_ID"} or n.endswith("_ID") or "BARCODE" in n


def safe_values(series: pd.Series, limit: int = 8) -> list[str]:
    vals = []
    for value in series.dropna().astype(str):
        value = value.strip()
        if value and value not in vals:
            vals.append(value[:120])
        if len(vals) >= limit:
            break
    return vals


def column_profile(df: pd.DataFrame, limit: int = 8) -> list[dict]:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nonmissing = int(s.notna().sum())
        unique = int(s.dropna().astype(str).nunique())
        rows.append({
            "column": col,
            "normalized_column": norm_col(col),
            "nonmissing": nonmissing,
            "missing": n - nonmissing,
            "missing_fraction": (n - nonmissing) / n if n else np.nan,
            "unique_nonmissing": unique,
            "examples": "[IDENTIFIER VALUES HIDDEN]" if is_identifier_column(col)
                        else " | ".join(safe_values(s, limit)),
        })
    return rows


def normalize_yes_no(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if not s:
            return np.nan
        if s in {"YES", "Y", "TRUE", "T", "1", "RECEIVED", "TREATED"}:
            return 1.0
        if s in {"NO", "N", "FALSE", "F", "0", "NOT RECEIVED", "UNTREATED"}:
            return 0.0
        if "YES" in s or "RECEIV" in s or "TREAT" in s and "NO" not in s:
            return 1.0
        if s.startswith("NO") or "NOT RECEIV" in s:
            return 0.0
        return np.nan
    return series.map(one)


def normalize_receptor(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if not s:
            return np.nan
        if s in {"POSITIVE", "POS", "+", "1", "YES"} or "POS" in s:
            return 1.0
        if s in {"NEGATIVE", "NEG", "-", "0", "NO"} or "NEG" in s:
            return 0.0
        return np.nan
    return series.map(one)


def normalize_event(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if not s:
            return np.nan
        if re.match(r"^1(?:\D|$)", s) or any(x in s for x in ("DECEASED", "DEAD", "DIED", "EVENT")):
            return 1.0
        if re.match(r"^0(?:\D|$)", s) or any(x in s for x in ("LIVING", "ALIVE", "CENSORED")):
            return 0.0
        if s in {"1", "TRUE", "YES"}:
            return 1.0
        if s in {"0", "FALSE", "NO"}:
            return 0.0
        return np.nan
    return series.map(one)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = ["empty"]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
        return
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> None:
    if not rows:
        print("<empty>")
        return
    show = list(rows if max_rows is None else rows[:max_rows])
    widths = {c: len(c) for c in columns}
    rendered = []
    for row in show:
        r = {c: str(row.get(c, "")) for c in columns}
        rendered.append(r)
        for c in columns:
            widths[c] = min(60, max(widths[c], len(r[c])))
    print("  ".join(c[:widths[c]].ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rendered:
        print("  ".join(row[c][:widths[c]].ljust(widths[c]) for c in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} additional rows written to CSV")


def find_timing_columns(columns: Sequence[str]) -> list[str]:
    treatment_tokens = ("HORMONE", "ENDOCRINE", "THERAP", "CHEMO", "TREATMENT")
    timing_tokens = ("DATE", "START", "INITIAT", "BEGIN", "DAY", "MONTH", "YEAR", "TIME")
    found = []
    for col in columns:
        n = norm_col(col)
        if any(t in n for t in treatment_tokens) and any(t in n for t in timing_tokens):
            found.append(col)
    return found


def sample_id_columns_from_matrix(path: Path, known_sample_ids: set[str]) -> dict:
    header = read_first_noncomment_line(path)
    delimiter = infer_delimiter(header)
    fields = next(csv.reader([header], delimiter=delimiter)) if header else []
    fields = [x.strip().strip('"') for x in fields]
    header_overlap = [x for x in fields if x in known_sample_ids]

    first_values = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")), delimiter=delimiter)
        try:
            next(reader)
        except StopIteration:
            pass
        for i, row in enumerate(reader):
            if row:
                first_values.append(row[0].strip().strip('"'))
            if i >= 49:
                break
    first_col_overlap = [x for x in first_values if x in known_sample_ids]

    if len(header_overlap) >= max(3, int(0.05 * max(1, len(known_sample_ids)))):
        orientation = "features_by_samples"
        sample_ids = set(header_overlap)
    elif len(first_col_overlap) >= 3:
        orientation = "samples_by_features"
        sample_ids = set(first_col_overlap)
    else:
        orientation = "unresolved"
        sample_ids = set(header_overlap) | set(first_col_overlap)

    return {
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "header_fields": len(fields),
        "orientation": orientation,
        "sample_ids_detected": sample_ids,
        "header_overlap_count": len(header_overlap),
        "first_column_overlap_count": len(first_col_overlap),
        "first_field": fields[0] if fields else "",
        "second_field": fields[1] if len(fields) > 1 else "",
    }
