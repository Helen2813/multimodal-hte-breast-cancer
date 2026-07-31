#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _stage16_utils import project_root


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(data: object) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dataframe_console(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "<empty table>"
    view = df if max_rows is None else df.head(max_rows)
    with pd.option_context(
        "display.max_rows", None if max_rows is None else max_rows,
        "display.max_columns", None,
        "display.width", 320,
        "display.max_colwidth", 120,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        return view.to_string(index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(
            lambda value: ""
            if pd.isna(value)
            else (f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        )
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines)


def ensure_dirs(root: Path) -> None:
    for rel in (
        "results/tables",
        "results/logs",
        "data/derived/stage20",
        "data/derived/manifests",
        "paper_A_treatment_effects",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def load_stage20_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return load_json(root / "stage20_config.json")


def final_paths(root: Path) -> dict[str, Path]:
    return {
        "analysis_plan": root / "paper_A_treatment_effects/analysis_plan_FINAL.md",
        "primary_estimand": root / "paper_A_treatment_effects/primary_estimand_FINAL.json",
        "model_registry": root / "paper_A_treatment_effects/model_registry_FINAL.json",
        "bootstrap_registry": root / "paper_A_treatment_effects/bootstrap_registry_FINAL.json",
        "hash_manifest": root / "data/derived/manifests/80_candidate_v9_protocol_lock_manifest.json",
        "lock_sentinel": root / "paper_A_treatment_effects/PROTOCOL_LOCKED_CANDIDATE_V9.txt",
    }


def refuse_existing_lock(root: Path) -> None:
    existing = [str(path.relative_to(root)) for path in final_paths(root).values() if path.exists()]
    if existing:
        raise RuntimeError(
            "A Candidate V9 final protocol artifact already exists. This runner refuses to overwrite a lock.\n"
            + "\n".join(existing)
        )
