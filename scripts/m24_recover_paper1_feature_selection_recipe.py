from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from _metabric_m4_utils import (
    load_config, out_dir, print_table, project_root, quick_sha256,
    rel, write_csv
)


def infer_modality(path: Path) -> str:
    text = path.as_posix().lower()
    for modality, tokens in {
        "RNA": ["/07_rna/", "/rna/"],
        "CNV": ["/cnv/", "/cna/", "copy_number"],
        "Methylation": ["/03_methylation/", "/methylation/"],
        "Mutation": ["/05_mutation/", "/mutation/", "/mutations/"],
        "miRNA": ["/mirna/"],
        "Protein": ["/protein", "/proteins/"],
    }.items():
        if any(token in text for token in tokens):
            return modality
    return "Unknown"


def normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        re.sub(r"[^A-Z0-9]+", "_", str(column).upper()).strip("_"): column
        for column in frame.columns
    }


def possible_column(columns: dict[str, str], tokens: list[str]) -> str | None:
    for token in tokens:
        if token in columns:
            return columns[token]
    for normalized, original in columns.items():
        if any(token in normalized for token in tokens):
            return original
    return None


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    processed = (root / cfg["paper1_processed_root"]).resolve()

    print("=" * 124)
    print("METABRIC M4.24 - RECOVER PAPER-1 FEATURE-SELECTION RECIPE")
    print("=" * 124)

    summary_files = sorted(processed.rglob("summary_all_results.csv"))
    summary_rows = []
    reported_best_rows = []

    for path in summary_files:
        modality = infer_modality(path)
        try:
            frame = pd.read_csv(path, low_memory=False)
            columns = normalized_columns(frame)
            algorithm_col = possible_column(columns, ["ALGORITHM", "METHOD"])
            alpha_col = possible_column(columns, ["ALPHA", "SIGNIFICANCE"])
            cindex_col = possible_column(
                columns,
                ["C_INDEX", "CINDEX", "CONCORDANCE_INDEX", "CONCORDANCE"],
            )
            scenario_col = possible_column(
                columns,
                ["DATASET", "PANEL", "SCENARIO", "CONFIGURATION", "FILE"],
            )

            summary_rows.append({
                "modality": modality,
                "path": rel(root, path),
                "rows": len(frame),
                "columns": " | ".join(map(str, frame.columns)),
                "algorithm_column": algorithm_col or "",
                "alpha_column": alpha_col or "",
                "cindex_column": cindex_col or "",
                "scenario_column": scenario_col or "",
                "quick_sha256": quick_sha256(path),
            })

            if cindex_col:
                scores = pd.to_numeric(frame[cindex_col], errors="coerce")
                if scores.notna().any():
                    index = scores.idxmax()
                    row = frame.loc[index]
                    reported_best_rows.append({
                        "modality": modality,
                        "summary_path": rel(root, path),
                        "reported_best_row_index": int(index),
                        "reported_best_cindex": float(scores.loc[index]),
                        "algorithm": str(row[algorithm_col]) if algorithm_col else "",
                        "alpha": str(row[alpha_col]) if alpha_col else "",
                        "scenario": str(row[scenario_col]) if scenario_col else "",
                        "full_row_json": json.dumps(
                            {str(key): str(value) for key, value in row.to_dict().items()},
                            sort_keys=True,
                        ),
                        "status": "RECOVERED_FOR_DOCUMENTATION_NOT_REOPTIMIZED",
                    })
        except Exception as exc:
            summary_rows.append({
                "modality": modality,
                "path": rel(root, path),
                "rows": "",
                "columns": "",
                "algorithm_column": "",
                "alpha_column": "",
                "cindex_column": "",
                "scenario_column": "",
                "quick_sha256": quick_sha256(path),
                "error": f"{type(exc).__name__}: {exc}",
            })

    selected_list_rows = []
    list_patterns = [
        "*_genes.txt", "*_features.txt", "consensus*.txt", "*_metrics.json"
    ]
    seen = set()
    for pattern in list_patterns:
        for path in sorted(processed.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            selected_list_rows.append({
                "modality": infer_modality(path),
                "path": rel(root, path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "quick_sha256": quick_sha256(path),
            })

    candidate_rows = []
    for path in sorted(processed.rglob("*")):
        if not path.is_file() or "statistical_filtered" not in path.parts:
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".parquet"}:
            continue
        candidate_rows.append({
            "modality": infer_modality(path),
            "path": rel(root, path),
            "name": path.name,
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
            "quick_sha256": quick_sha256(path),
        })

    source_excerpt_rows = []
    source_extensions = {".py", ".ipynb", ".r", ".R", ".ps1"}
    source_roots = [root / "scripts", root / "01_Causal_feature_extraction"]
    tokens = cfg["paper1_recipe_tokens"]
    for base in source_roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in source_extensions:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            matches = []
            for line_number, line in enumerate(text.splitlines(), 1):
                if any(token.lower() in line.lower() for token in tokens):
                    matches.append({
                        "path": rel(root, path),
                        "line": line_number,
                        "excerpt": " ".join(line.strip().split())[:500],
                    })
                if len(matches) >= 25:
                    break
            source_excerpt_rows.extend(matches)

    write_csv(out / "m24_paper1_summary_files.csv", summary_rows)
    write_csv(out / "m24_paper1_reported_best_rows.csv", reported_best_rows)
    write_csv(out / "m24_paper1_selected_list_inventory.csv", selected_list_rows)
    write_csv(out / "m24_paper1_candidate_matrix_inventory.csv", candidate_rows)
    write_csv(out / "m24_paper1_source_excerpts.csv", source_excerpt_rows)

    shared = set(cfg["paper1_modalities_shared_with_metabric"])
    modalities_with_summary = {
        row["modality"] for row in summary_rows
        if not row.get("error")
    }
    modalities_with_lists = {row["modality"] for row in selected_list_rows}
    modalities_with_candidates = {row["modality"] for row in candidate_rows}
    recipe_status_rows = []
    for modality in sorted(shared):
        recipe_status_rows.append({
            "modality": modality,
            "summary_found": modality in modalities_with_summary,
            "selected_lists_found": modality in modalities_with_lists,
            "candidate_matrices_found": modality in modalities_with_candidates,
            "reported_best_row_recovered": any(
                row["modality"] == modality for row in reported_best_rows
            ),
        })
    write_csv(out / "m24_paper1_recipe_status.csv", recipe_status_rows)

    print("Paper-1 summary files")
    print_table(
        summary_rows,
        [
            "modality", "path", "rows", "algorithm_column",
            "alpha_column", "cindex_column", "scenario_column",
            "quick_sha256", "error"
        ],
        max_rows=50,
    )

    print("\nReported best rows from the historical summaries")
    print_table(
        reported_best_rows,
        [
            "modality", "summary_path", "reported_best_cindex",
            "algorithm", "alpha", "scenario", "status"
        ],
        max_rows=50,
    )

    print("\nShared-modality recipe recovery status")
    print_table(
        recipe_status_rows,
        [
            "modality", "summary_found", "selected_lists_found",
            "candidate_matrices_found", "reported_best_row_recovered"
        ],
    )

    print("\nSelected-list inventory")
    print_table(
        selected_list_rows,
        ["modality", "path", "name", "size_bytes", "quick_sha256"],
        max_rows=100,
    )

    print("\nCandidate matrix inventory")
    print_table(
        candidate_rows,
        ["modality", "path", "name", "size_mb", "quick_sha256"],
        max_rows=100,
    )

    print("\nPaper-1 source excerpts")
    print_table(
        source_excerpt_rows,
        ["path", "line", "excerpt"],
        max_rows=150,
    )

    print("\nPASS: historical Paper-1 recipe evidence recovered without rerunning selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
