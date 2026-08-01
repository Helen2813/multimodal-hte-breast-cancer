from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


GENE_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{1,24}$")
ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m3_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def out_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(text).upper()).strip("_")


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


def quick_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with path.open("rb") as f:
        h.update(f.read(block_size))
        if size > block_size:
            f.seek(max(0, size - block_size))
            h.update(f.read(block_size))
    return h.hexdigest()


def read_text_prefix(path: Path, byte_limit: int) -> str:
    with path.open("rb") as f:
        raw = f.read(byte_limit)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def noncomment_lines(text: str, limit: int = 5) -> list[str]:
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def infer_delimiter(line: str) -> str:
    options = [("\t", line.count("\t")), (",", line.count(",")), (";", line.count(";"))]
    options.sort(key=lambda x: x[1], reverse=True)
    return options[0][0] if options and options[0][1] else "\t"


def parse_fields(line: str, delimiter: str) -> list[str]:
    try:
        return [x.strip().strip('"') for x in next(csv.reader([line], delimiter=delimiter))]
    except Exception:
        return [x.strip() for x in line.split(delimiter)]


def is_excluded(root: Path, path: Path, tokens: Sequence[str]) -> bool:
    r = rel(root, path).lower()
    return any(token.lower().replace("\\", "/") in r for token in tokens)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = list(fieldnames or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
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
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
            widths[c] = min(55, max(widths[c], len(r[c])))
    print("  ".join(c[:widths[c]].ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rendered:
        print("  ".join(row[c][:widths[c]].ljust(widths[c]) for c in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} additional rows written to CSV")


def gene_symbol_like(value: str) -> bool:
    v = str(value).strip().upper()
    if not v or ENSEMBL_RE.match(v):
        return False
    if v in {"PATIENT_ID", "SAMPLE_ID", "CASE_ID", "ROW_ID", "HUGO_SYMBOL", "ENTREZ_GENE_ID"}:
        return False
    return bool(GENE_SYMBOL_RE.match(v)) and any(ch.isalpha() for ch in v)


def ensembl_like(value: str) -> bool:
    return bool(ENSEMBL_RE.match(str(value).strip()))


def classify_identifiers(values: Sequence[str]) -> dict:
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return {"identifier_type": "none", "gene_like_count": 0, "ensembl_count": 0}
    gene_count = sum(gene_symbol_like(v) for v in vals)
    ensembl_count = sum(ensembl_like(v) for v in vals)
    if ensembl_count >= max(5, int(0.5 * len(vals))):
        id_type = "ensembl"
    elif gene_count >= max(5, int(0.5 * len(vals))):
        id_type = "gene_symbol"
    else:
        id_type = "mixed_or_unknown"
    return {
        "identifier_type": id_type,
        "gene_like_count": gene_count,
        "ensembl_count": ensembl_count,
    }


def modality_score(path: Path, header_fields: Sequence[str], modality: str) -> int:
    p = path.as_posix().lower()
    name = path.name.lower()
    fields_norm = [norm(x) for x in header_fields]
    joined = " ".join(fields_norm[:2000])
    score = 0

    name_tokens = {
        "clinical": ["clinical", "cohort", "compact", "baseline"],
        "rna": ["rna", "mrna", "expression", "transcript"],
        "cna": ["cna", "cnv", "copy_number", "copynumber"],
        "mutations": ["mutation", "mutations", "maf", "somatic"],
        "methylation": ["methyl", "cpg", "450k", "rrbs"],
        "annotation": ["annotation", "gene_map", "ensembl", "biomart"],
        "pathway_gmt": [".gmt", "pathway", "hallmark", "reactome", "geneset"],
    }
    for token in name_tokens.get(modality, []):
        if token in name or token in p:
            score += 30

    if modality == "clinical":
        for token in ("PATIENT_ID", "SAMPLE_ID", "OS_MONTHS", "OS_STATUS", "AGE", "STAGE", "TREATMENT"):
            if token in joined:
                score += 12
    elif modality == "rna":
        if sum(ensembl_like(x) for x in header_fields) >= 10:
            score += 100
        if sum(gene_symbol_like(x) for x in header_fields) >= 50:
            score += 80
        if any("RNA_ENSG" in x for x in fields_norm):
            score += 100
    elif modality == "cna":
        if any(token in joined for token in ("CNA", "CNV", "COPY_NUMBER")):
            score += 60
        if sum(gene_symbol_like(x) for x in header_fields) >= 50:
            score += 50
    elif modality == "mutations":
        if any(token in joined for token in ("HUGO_SYMBOL", "TUMOR_SAMPLE_BARCODE", "VARIANT_CLASSIFICATION")):
            score += 100
        if any("MUTATION" in x for x in fields_norm):
            score += 60
    elif modality == "methylation":
        if any(token in joined for token in ("METHYL", "CPG", "CG")):
            score += 60
    elif modality == "annotation":
        if any("ENSEMBL" in x for x in fields_norm) and any(token in joined for token in ("SYMBOL", "HUGO", "GENE_NAME")):
            score += 140
    elif modality == "pathway_gmt":
        if path.suffix.lower() == ".gmt":
            score += 200

    if "metabric" in p:
        score -= 1000
    if "stage22" in p or "publication_assets" in p:
        score -= 200
    return score
