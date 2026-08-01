from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.I)
CPG_RE = re.compile(r"^cg\d{8,}$", re.I)
GENE_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{1,24}$")


def root() -> Path:
    return Path.cwd().resolve()


def load_cfg(project: Path) -> dict:
    return json.loads((project / "metabric_m3b_config.json").read_text(encoding="utf-8"))


def output_dir(project: Path, cfg: dict) -> Path:
    path = (project / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(project: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def quick_hash(path: Path, block: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as f:
        h.update(f.read(block))
        if size > block:
            f.seek(max(0, size - block))
            h.update(f.read(block))
    return h.hexdigest()


def row_count(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            count += chunk.count(b"\n")
    return max(0, count - 1)


def read_header(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix == ".tsv":
        return list(pd.read_csv(path, sep="\t", nrows=0).columns)
    for sep in ("\t", ","):
        try:
            cols = list(pd.read_csv(path, sep=sep, comment="#", nrows=0).columns)
            if len(cols) > 1:
                return cols
        except Exception:
            pass
    raise ValueError(f"Could not read header: {path}")


def strip_prefix(column: str, prefixes: Sequence[str]) -> tuple[str, str]:
    upper = column.upper()
    for prefix in prefixes:
        if upper.startswith(prefix.upper()):
            return prefix, column[len(prefix):]
    return "", column


def gene_like(value: str) -> bool:
    v = str(value).strip().upper()
    if not v or ENSEMBL_RE.match(v) or CPG_RE.match(v):
        return False
    if v in {"PATIENT_ID", "SAMPLE_ID", "CASE_ID", "ROW_ID", "UNNAMED_0"}:
        return False
    return bool(GENE_RE.match(v)) and any(ch.isalpha() for ch in v)


def id_type(value: str) -> str:
    v = str(value).strip()
    if ENSEMBL_RE.match(v):
        return "ensembl"
    if CPG_RE.match(v):
        return "cpg"
    if gene_like(v):
        return "gene_symbol"
    return "other"


def canonical_id(value: str) -> str:
    v = str(value).strip().upper()
    return v.split(".", 1)[0] if ENSEMBL_RE.match(v) else v


def classify_columns(columns: Sequence[str], prefix_map: dict) -> list[dict]:
    rows = []
    for index, column in enumerate(columns):
        modality = "other"
        prefix = ""
        identifier = str(column)
        for candidate, prefixes in prefix_map.items():
            used, stripped = strip_prefix(str(column), prefixes)
            if used:
                modality, prefix, identifier = candidate, used, stripped
                break
        rows.append({
            "column_index": index,
            "column": str(column),
            "modality": modality,
            "prefix": prefix,
            "identifier": identifier,
            "canonical_identifier": canonical_id(identifier),
            "identifier_type": id_type(identifier),
        })
    return rows


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        names = list(fields or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=names).writeheader()
        return
    if fields is None:
        names = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    else:
        names = list(fields)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def print_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> None:
    if not rows:
        print("<empty>")
        return
    shown = list(rows if max_rows is None else rows[:max_rows])
    widths = {c: len(c) for c in columns}
    rendered = []
    for row in shown:
        r = {c: str(row.get(c, "")) for c in columns}
        rendered.append(r)
        for c in columns:
            widths[c] = min(58, max(widths[c], len(r[c])))
    print("  ".join(c[:widths[c]].ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rendered:
        print("  ".join(row[c][:widths[c]].ljust(widths[c]) for c in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} additional rows written to CSV")


def excluded(project: Path, path: Path, tokens: Sequence[str]) -> bool:
    rp = rel(project, path).lower()
    return any(token.lower().replace("\\", "/") in rp for token in tokens)


def candidate_mapping_path(project: Path, path: Path, cfg: dict) -> bool:
    if path.suffix.lower() not in set(cfg["search_extensions"]):
        return False
    if excluded(project, path, cfg["excluded_path_tokens"]):
        return False
    if path.stat().st_size > float(cfg["maximum_mapping_candidate_size_mb"]) * 1024 * 1024:
        return False
    rp = rel(project, path).lower()
    return any(token.lower() in rp for token in cfg["mapping_path_tokens"])


def scan_tokens(path: Path, tokens: Sequence[str], byte_limit: int) -> list[str]:
    if not tokens:
        return []
    wanted = {token: token.encode("utf-8").upper() for token in tokens}
    found = set()
    read = 0
    with path.open("rb") as f:
        while read < byte_limit:
            chunk = f.read(min(1024 * 1024, byte_limit - read))
            if not chunk:
                break
            read += len(chunk)
            upper = chunk.upper()
            for token, encoded in wanted.items():
                if encoded in upper:
                    found.add(token)
            if len(found) == len(wanted):
                break
    return sorted(found)
