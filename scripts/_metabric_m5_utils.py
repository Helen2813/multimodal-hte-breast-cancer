from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m5_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def out_dir(root: Path, config: dict) -> Path:
    path = (root / config["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root: Path, config: dict) -> Path:
    path = (root / config["raw_dir"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing raw METABRIC directory: {path}")
    return path


def norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def exact_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {norm(column): column for column in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = list(fieldnames or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> None:
    if not rows:
        print("<empty>")
        return
    show = list(rows if max_rows is None else rows[:max_rows])
    widths = {column: len(column) for column in columns}
    rendered = []
    for row in show:
        item = {column: str(row.get(column, "")) for column in columns}
        rendered.append(item)
        for column in columns:
            widths[column] = min(60, max(widths[column], len(item[column])))
    print("  ".join(column[:widths[column]].ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rendered:
        print("  ".join(row[column][:widths[column]].ljust(widths[column]) for column in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} additional rows written to CSV")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_event(series: pd.Series) -> pd.Series:
    def one(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().upper()
        if text.startswith("1:") or text in {"1", "DECEASED", "DEAD"} or "DIED" in text:
            return 1.0
        if text.startswith("0:") or text in {"0", "LIVING", "ALIVE"}:
            return 0.0
        return np.nan
    return series.map(one)
