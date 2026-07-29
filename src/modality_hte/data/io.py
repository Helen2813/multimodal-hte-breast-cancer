from __future__ import annotations

from pathlib import Path

import pandas as pd


class UnsupportedTableFormat(ValueError):
    """Raised when a configured data file uses an unsupported extension."""


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, TSV, TXT, or Parquet tables with explicit format handling."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Data file not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(table_path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)

    raise UnsupportedTableFormat(
        f"Unsupported table format '{suffix}' for {table_path}. "
        "Supported extensions: .csv, .tsv, .txt, .parquet, .pq."
    )
