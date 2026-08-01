from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)
CPG_RE = re.compile(r"^cg\d{8,}$", re.IGNORECASE)


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m4_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def out_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["metabric_raw_dir"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing METABRIC raw directory: {path}")
    return path


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def canonical_ensembl(value: str) -> str:
    value = str(value).strip().upper()
    return value.split(".", 1)[0] if ENSEMBL_RE.match(value) else value


def strip_modality_prefix(column: str, modality: str) -> str:
    prefixes = {
        "rna": ["RNA_"],
        "cna": ["CNV_", "CNA_"],
        "mutations": ["MUT_"],
        "methylation": ["METH_"],
    }
    upper = str(column).upper()
    for prefix in prefixes.get(modality, []):
        if upper.startswith(prefix):
            return str(column)[len(prefix):]
    return str(column)


def quick_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        h.update(handle.read(block_size))
        if size > block_size:
            handle.seek(max(0, size - block_size))
            h.update(handle.read(block_size))
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def exact_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {norm(column): column for column in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    return None


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


def load_m3b_registry(root: Path, cfg: dict) -> pd.DataFrame:
    path = root / cfg["metabric_m3b_dir"] / "m15_tcga_feature_registry.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing M3B feature registry: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def selected_identifiers(registry: pd.DataFrame, modality: str) -> list[dict]:
    rows = []
    subset = registry[registry["modality"].str.lower() == modality.lower()].copy()
    for _, row in subset.iterrows():
        raw_identifier = strip_modality_prefix(row["column"], modality)
        identifier = canonical_ensembl(raw_identifier)
        if ENSEMBL_RE.match(identifier):
            identifier_type = "ensembl"
        elif CPG_RE.match(identifier):
            identifier_type = "cpg"
        elif identifier and not identifier.upper().endswith("_MISSING"):
            identifier_type = "gene_symbol"
        else:
            identifier_type = "sentinel_or_other"
        rows.append({
            "modality": modality,
            "tcga_column": row["column"],
            "raw_identifier": raw_identifier,
            "canonical_identifier": identifier,
            "identifier_type": identifier_type,
        })
    return rows


def numeric_diagnostics(frame: pd.DataFrame, id_columns: Sequence[str]) -> list[dict]:
    rows = []
    for column in frame.columns:
        if column in id_columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append({
            "feature": column,
            "n": len(series),
            "nonmissing": int(series.notna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique_nonmissing": int(series.dropna().nunique()),
            "mean": float(series.mean()) if series.notna().any() else np.nan,
            "sd": float(series.std(ddof=1)) if series.notna().sum() > 1 else np.nan,
            "minimum": float(series.min()) if series.notna().any() else np.nan,
            "maximum": float(series.max()) if series.notna().any() else np.nan,
        })
    return rows
