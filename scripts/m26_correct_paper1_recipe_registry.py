from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _metabric_m5_utils import (
    load_config, out_dir, print_table, project_root, rel, sha256, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    processed = (root / cfg["paper1_processed_root"]).resolve()

    print("=" * 124)
    print("METABRIC M5.26 - CORRECTED PAPER-1 RECIPE REGISTRY")
    print("=" * 124)

    summary_rows = []
    grid_rows = []
    selected_list_rows = []
    candidate_rows = []

    for modality, folder_name in cfg["paper1_modalities"].items():
        base = processed / folder_name
        summary_path = base / "mb_results" / "summary_all_results.csv"
        if not summary_path.exists():
            summary_rows.append({
                "modality": modality,
                "summary_found": False,
                "summary_path": rel(root, summary_path),
                "rows": 0,
                "datasets": 0,
                "algorithms": 0,
                "alphas": 0,
                "selected_lists": 0,
                "candidate_matrices": 0,
                "complete": False,
            })
            continue

        frame = pd.read_csv(summary_path, low_memory=False)
        required = {"dataset", "algorithm", "alpha", "c_index"}
        lookup = {str(column).lower(): column for column in frame.columns}
        missing = required - set(lookup)
        if missing:
            raise RuntimeError(f"{summary_path} is missing columns: {sorted(missing)}")

        dataset_col = lookup["dataset"]
        algorithm_col = lookup["algorithm"]
        alpha_col = lookup["alpha"]
        cindex_col = lookup["c_index"]

        for index, row in frame.iterrows():
            grid_rows.append({
                "modality": modality,
                "row_index": int(index),
                "dataset": str(row[dataset_col]),
                "algorithm": str(row[algorithm_col]),
                "alpha": str(row[alpha_col]),
                "c_index": float(row[cindex_col]) if pd.notna(row[cindex_col]) else "",
                "source_summary": rel(root, summary_path),
            })

        list_files = sorted((base / "mb_results").rglob("*_genes.txt"))
        matrix_files = sorted((base / "statistical_filtered").glob("*.csv"))
        matrix_files = [
            path for path in matrix_files
            if path.name not in {"outcome.csv", "datasets_summary.csv"}
            and "statistics" not in path.name.lower()
            and "annotated" not in path.name.lower()
        ]

        for path in list_files:
            selected_list_rows.append({
                "modality": modality,
                "path": rel(root, path),
                "name": path.name,
                "parent_dataset": path.parent.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

        for path in matrix_files:
            candidate_rows.append({
                "modality": modality,
                "path": rel(root, path),
                "name": path.name,
                "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
                "sha256": sha256(path),
            })

        datasets = sorted(frame[dataset_col].astype(str).unique())
        algorithms = sorted(frame[algorithm_col].astype(str).unique())
        alphas = sorted(frame[alpha_col].astype(str).unique())
        complete = (
            len(frame) > 0
            and len(datasets) > 0
            and len(algorithms) > 0
            and len(alphas) > 0
            and len(list_files) > 0
            and len(matrix_files) > 0
        )
        summary_rows.append({
            "modality": modality,
            "summary_found": True,
            "summary_path": rel(root, summary_path),
            "rows": len(frame),
            "datasets": len(datasets),
            "dataset_values": " | ".join(datasets),
            "algorithms": len(algorithms),
            "algorithm_values": " | ".join(algorithms),
            "alphas": len(alphas),
            "alpha_values": " | ".join(alphas),
            "selected_lists": len(list_files),
            "candidate_matrices": len(matrix_files),
            "complete": complete,
            "summary_sha256": sha256(summary_path),
        })

    write_csv(out / "m26_paper1_recipe_summary.csv", summary_rows)
    write_csv(out / "m26_paper1_full_grid.csv", grid_rows)
    write_csv(out / "m26_paper1_selected_lists.csv", selected_list_rows)
    write_csv(out / "m26_paper1_candidate_matrices.csv", candidate_rows)

    recipe_complete = len(summary_rows) == 4 and all(bool(row["complete"]) for row in summary_rows)
    registry = {
        "status": "PAPER1_SHARED_MODALITY_RECIPE_RECOVERED" if recipe_complete else "PAPER1_RECIPE_INCOMPLETE",
        "modalities": summary_rows,
        "important_rule": (
            "Historical best rows are descriptive. METABRIC performance claims require nested selection "
            "inside training folds and may not use full-cohort best-row selection."
        ),
    }
    (out / "m26_paper1_recipe_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )

    print("Corrected Paper-1 recipe registry")
    print_table(
        summary_rows,
        [
            "modality", "summary_found", "rows", "datasets", "dataset_values",
            "algorithms", "algorithm_values", "alphas", "alpha_values",
            "selected_lists", "candidate_matrices", "complete"
        ],
    )

    print("\nExact algorithm/alpha/dataset grid")
    print_table(
        grid_rows,
        ["modality", "dataset", "algorithm", "alpha", "c_index"],
        max_rows=250,
    )

    print(f"\nRecipe status: {registry['status']}")
    print(f"Rule: {registry['important_rule']}")

    if not recipe_complete:
        raise RuntimeError("Paper-1 recipe remains incomplete after exact folder-based classification.")

    print("\nPASS: CNV is now classified correctly and all four shared-modality recipes are recovered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
