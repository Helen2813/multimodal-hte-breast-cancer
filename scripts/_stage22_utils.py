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


def find_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "results").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing results/ and scripts/.")


def load_config(root: Path) -> dict[str, Any]:
    path = root / "stage22_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 22 config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    output = root / config["output_directory"]
    tables = output / "tables"
    figures = output / "figures"
    manuscript = output / "manuscript_snippets"
    audit = output / "audit"
    for path in (output, tables, figures, manuscript, audit):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "output": output,
        "tables": tables,
        "figures": figures,
        "manuscript": manuscript,
        "audit": audit,
    }


def first_existing(root: Path, candidates: Sequence[str], required: bool = True) -> Path | None:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    if required:
        raise FileNotFoundError("None of the required files exist:\n" + "\n".join(candidates))
    return None


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_one_row(path: Path) -> pd.Series:
    df = read_csv(path)
    if len(df) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, found {len(df)}")
    return df.iloc[0]


def pick_col(df: pd.DataFrame, candidates: Sequence[str], label: str, required: bool = True) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    if required:
        raise KeyError(f"Could not find {label}. Tried columns: {list(candidates)}. Available: {list(df.columns)}")
    return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite numeric value, found {value!r}")
    return result


def assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise RuntimeError(f"Non-finite value in check {label}: observed={observed}, expected={expected}")
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"Recomputation mismatch for {label}: observed={observed:.12g}, "
            f"expected={expected:.12g}, tolerance={tolerance}"
        )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    headers = [str(c) for c in df.columns]
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def tex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_value(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_align: str | None = None,
    digits: int = 2,
    footnote: str | None = None,
) -> str:
    align = column_align or ("l" + "r" * max(0, len(df.columns) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        values = [tex_escape(format_value(v, digits=digits)) for v in row.tolist()]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if footnote:
        lines.append(r"\par\vspace{2pt}{\footnotesize " + tex_escape(footnote) + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table_bundle(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    md_path: Path,
    caption: str,
    label: str,
    digits: int = 2,
    footnote: str | None = None,
) -> None:
    df.to_csv(csv_path, index=False)
    tex_path.write_text(
        dataframe_to_latex(df, caption=caption, label=label, digits=digits, footnote=footnote),
        encoding="utf-8",
    )
    md_path.write_text(markdown_table(df), encoding="utf-8")


def effect_quantiles(values: pd.Series) -> dict[str, float]:
    clean = numeric(values).dropna().astype(float)
    if clean.empty:
        raise RuntimeError("No finite bootstrap effects were found.")
    return {
        "q025": float(clean.quantile(0.025)),
        "q05": float(clean.quantile(0.05)),
        "q10": float(clean.quantile(0.10)),
        "q25": float(clean.quantile(0.25)),
        "q50": float(clean.quantile(0.50)),
        "q75": float(clean.quantile(0.75)),
        "q90": float(clean.quantile(0.90)),
        "q95": float(clean.quantile(0.95)),
        "q975": float(clean.quantile(0.975)),
    }


def compute_prefix_table(effects: pd.Series, point: float, prefixes: Sequence[int]) -> pd.DataFrame:
    clean = numeric(effects).dropna().astype(float).reset_index(drop=True)
    rows: list[dict[str, float | int]] = []
    for prefix in prefixes:
        if prefix > len(clean):
            continue
        sample = clean.iloc[:prefix]
        q025 = float(sample.quantile(0.025))
        q975 = float(sample.quantile(0.975))
        rows.append(
            {
                "prefix_repetitions": int(prefix),
                "mean_effect_days": float(sample.mean()),
                "median_effect_days": float(sample.median()),
                "sd_effect_days": float(sample.std(ddof=1)),
                "percentile_ci_low_days": q025,
                "percentile_ci_high_days": q975,
                "basic_ci_low_days": float(2 * point - q975),
                "basic_ci_high_days": float(2 * point - q025),
                "fraction_positive": float((sample > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def extract_manifest_entries(obj: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if isinstance(obj, dict):
        lowered = {str(k).lower(): k for k in obj.keys()}
        path_key = next((lowered[k] for k in ("path", "relative_path", "file", "file_path") if k in lowered), None)
        hash_key = next((lowered[k] for k in ("sha256", "expected_sha256", "file_sha256") if k in lowered), None)
        if path_key is not None and hash_key is not None:
            entries.append({"path": str(obj[path_key]), "sha256": str(obj[hash_key]).lower()})
        for value in obj.values():
            entries.extend(extract_manifest_entries(value))
    elif isinstance(obj, list):
        for value in obj:
            entries.extend(extract_manifest_entries(value))
    return entries


def verify_manifest(root: Path) -> pd.DataFrame:
    manifest = first_existing(
        root,
        ["data/derived/manifests/80_candidate_v9_protocol_lock_manifest.json"],
        required=False,
    )
    if manifest is None:
        return pd.DataFrame([{"path": "manifest", "exists": False, "match": False, "note": "manifest missing"}])
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    entries = extract_manifest_entries(obj)
    rows = []
    seen: set[str] = set()
    for entry in entries:
        rel = entry["path"].replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        path = root / rel
        exists = path.exists()
        observed = sha256(path) if exists and path.is_file() else ""
        rows.append(
            {
                "path": rel,
                "exists": bool(exists),
                "expected_sha256": entry["sha256"],
                "observed_sha256": observed,
                "match": bool(exists and observed.lower() == entry["sha256"].lower()),
            }
        )
    if not rows:
        rows.append(
            {
                "path": str(manifest.relative_to(root)),
                "exists": True,
                "expected_sha256": "",
                "observed_sha256": sha256(manifest),
                "match": True,
                "note": "No file entries recognized; manifest itself exists.",
            }
        )
    return pd.DataFrame(rows)


def locate_tex_sources(root: Path, generated_output: Path) -> list[Path]:
    preferred = [
        root / "sn-article_AAIL_working.tex",
        root / "paper_A_treatment_effects" / "sn-article_AAIL_working.tex",
    ]
    found = [p for p in preferred if p.exists()]
    if found:
        return found
    candidates = []
    for path in root.rglob("*.tex"):
        try:
            path.relative_to(generated_output)
            continue
        except ValueError:
            pass
        if any(part.startswith(".venv") for part in path.parts):
            continue
        if "results" in path.parts and "tables" in path.parts:
            continue
        if path.stat().st_size > 10_000:
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:5]


def stale_claim_audit(
    tex_sources: Sequence[Path],
    numeric_tokens: Sequence[str],
    claim_patterns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in tex_sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for line_no, line in enumerate(lines, start=1):
            lower = line.lower()
            for token in numeric_tokens:
                if token in line:
                    rows.append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "kind": "stale_numeric_token",
                            "pattern": token,
                            "context": line.strip()[:500],
                        }
                    )
            for pattern in claim_patterns:
                if pattern.lower() in lower:
                    rows.append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "kind": "potential_overclaim",
                            "pattern": pattern,
                            "context": line.strip()[:500],
                        }
                    )
    return pd.DataFrame(rows, columns=["file", "line", "kind", "pattern", "context"])


def print_frame(title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print("=" * 124)
    print(title)
    print("=" * 124)
    if df.empty:
        print("<empty>")
    else:
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"... {len(df) - max_rows} additional rows written to file")
