from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
RESULTS_DIR = PROJECT_ROOT / "results"

MODALITY_FOLDERS = {
    "clinical": "01_Clinical",
    "cnv": "02_CNV",
    "methylation": "03_Methylation",
    "mirna": "04_miRNA",
    "mutation": "05_Mutation",
    "protein": "06_proteins",
    "rna": "07_RNA",
}

MODALITY_PREFIXES = {
    "clinical": ("CLIN_",),
    "rna": ("RNA_",),
    "cnv": ("CNV_",),
    "mutation": ("MUT_", "MUTATION_"),
    "methylation": ("METH_", "METHYLATION_"),
    "mirna": ("MIRNA_", "miRNA_"),
    "protein": ("PROT_", "PROTEIN_"),
}

ID_CANDIDATES = (
    "patient_id", "patient", "case_submitter_id", "bcr_patient_barcode",
    "submitter_id", "case_id", "sample_id", "sample", "barcode",
)

TREATMENT_CANDIDATES = (
    "T", "T_hormone", "T_hormone_excl", "T_chemo",
    "T_targeted", "T_radiation",
)
OUTCOME_CANDIDATES = ("Y", "Y_died_5yr", "OS", "event", "status")
TIME_CANDIDATES = (
    "OS.time", "OS_time", "time", "survival_time", "days_to_event",
    "days_to_death", "days_to_last_follow_up",
)


def ensure_dirs() -> None:
    for path in (
        DERIVED_DIR / "audits",
        DERIVED_DIR / "cohorts",
        DERIVED_DIR / "manifests",
        RESULTS_DIR / "tables",
        RESULTS_DIR / "figures",
        RESULTS_DIR / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def find_first(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def find_ite_file() -> Optional[Path]:
    output_dir = PROCESSED_DIR / "output"
    return find_first([
        output_dir / "ite_ready_dataset_v2.csv",
        output_dir / "ite_ready_dataset.csv",
    ])


def find_merge_file(folder_name: str, preferred: str = "08_composite.csv") -> Optional[Path]:
    folder = PROCESSED_DIR / folder_name
    preferred_path = folder / preferred
    if preferred_path.exists():
        return preferred_path
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    return files[0] if files else None


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    lookup = {str(c).strip().lower(): str(c) for c in df.columns}
    for candidate in ID_CANDIDATES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if len(df.columns):
        first = str(df.columns[0])
        sample = df[first].dropna().astype(str).head(50)
        frac = sample.str.contains(
            r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}", regex=True
        ).mean() if len(sample) else 0.0
        if first.lower().startswith("unnamed") or frac >= 0.5:
            return first
    return None


_TCGA_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def normalize_patient_id(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace("_", "-")
    match = _TCGA_RE.search(text)
    return match.group(1).upper() if match else text


def add_normalized_patient_id(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    out = df.copy()
    out["patient_id_normalized"] = out[source_col].map(normalize_patient_id)
    return out


def exact_or_case_insensitive_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def parse_binary_status(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[numeric.notna()] = (numeric.loc[numeric.notna()] > 0.5).astype(float)
    text = series.astype(str).str.strip().str.lower()
    positive = {"positive", "pos", "yes", "true", "1", "1.0", "present", "detected"}
    negative = {"negative", "neg", "no", "false", "0", "0.0", "absent", "not detected"}
    out.loc[text.isin(positive)] = 1.0
    out.loc[text.isin(negative)] = 0.0
    return out


def safe_numeric_event(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[numeric.notna()] = (numeric.loc[numeric.notna()] > 0.5).astype(float)
        return out
    return parse_binary_status(series)


def modality_counts(columns: Iterable[str]) -> dict[str, int]:
    cols = [str(c) for c in columns]
    return {
        modality: sum(any(col.startswith(prefix) for prefix in prefixes) for col in cols)
        for modality, prefixes in MODALITY_PREFIXES.items()
    }


def treatment_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in TREATMENT_CANDIDATES if c in df.columns]


def outcome_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in OUTCOME_CANDIDATES if c in df.columns]


def time_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in TIME_CANDIDATES if c in df.columns]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_unique_ids(df: pd.DataFrame, label: str, report_dir: Path) -> None:
    duplicated = df[df["patient_id_normalized"].duplicated(keep=False)].copy()
    if not duplicated.empty:
        output = report_dir / f"duplicate_ids_{label}.csv"
        duplicated.to_csv(output, index=False)
        raise ValueError(
            f"{label}: duplicated normalized patient IDs found. Inspect {output}."
        )
