from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m2_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def raw_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["raw_dir"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"METABRIC raw directory not found: {path}")
    return path


def out_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_col(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


def exact_col(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {norm_col(c): c for c in columns}
    for candidate in candidates:
        if norm_col(candidate) in lookup:
            return lookup[norm_col(candidate)]
    return None


def read_cbio(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
        na_values=["", "NA", "N/A", "NaN", "nan", "[Not Available]", "Unknown"],
        keep_default_na=True,
    )


def normalize_yes_no(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if s in {"YES", "Y", "TRUE", "T", "1"}:
            return 1.0
        if s in {"NO", "N", "FALSE", "F", "0"}:
            return 0.0
        return np.nan
    return series.map(one)


def normalize_receptor(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if "POS" in s or s in {"+", "1"}:
            return 1.0
        if "NEG" in s or s in {"-", "0"}:
            return 0.0
        return np.nan
    return series.map(one)


def normalize_event(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().upper()
        if s.startswith("1:") or s in {"1", "DECEASED", "DEAD"} or "DIED" in s:
            return 1.0
        if s.startswith("0:") or s in {"0", "LIVING", "ALIVE"}:
            return 0.0
        return np.nan
    return series.map(one)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_header(path: Path, delimiter: str) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            return [x.strip().strip('"') for x in next(csv.reader([line], delimiter=delimiter))]
    return []


def first_column_values(path: Path, delimiter: str) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")), delimiter=delimiter)
        try:
            next(reader)
        except StopIteration:
            return values
        for row in reader:
            if row:
                value = row[0].strip().strip('"')
                if value:
                    values.add(value)
    return values


def exact_matrix_sample_set(path: Path, known_samples: set[str], delimiter: str) -> tuple[str, set[str], dict]:
    header = read_header(path, delimiter)
    header_samples = set(header) & known_samples
    first_values = first_column_values(path, delimiter)
    first_samples = first_values & known_samples

    if len(header_samples) >= len(first_samples) and header_samples:
        orientation = "features_by_samples"
        samples = header_samples
    elif first_samples:
        orientation = "samples_by_features"
        samples = first_samples
    else:
        orientation = "unresolved"
        samples = set()

    return orientation, samples, {
        "header_fields": len(header),
        "header_sample_count": len(header_samples),
        "first_column_sample_count": len(first_samples),
        "first_field": header[0] if header else "",
        "second_field": header[1] if len(header) > 1 else "",
    }


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = list(fieldnames or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
        return
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
            widths[c] = min(55, max(widths[c], len(r[c])))
    print("  ".join(c[:widths[c]].ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rendered:
        print("  ".join(row[c][:widths[c]].ljust(widths[c]) for c in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} additional rows written to CSV")
