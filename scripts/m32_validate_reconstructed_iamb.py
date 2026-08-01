from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _metabric_m6_utils import (
    guess_id_column, iamb_select, jaccard, load_config, normalize_tcga_id,
    out_dir, overlap_coefficient, pad_selection, print_table, project_root,
    read_feature_list, standardize_matrix, transform_for_ci, write_csv
)


def load_candidate_and_outcome(candidate_path: Path, outcome_path: Path):
    candidates = pd.read_csv(candidate_path, low_memory=False)
    outcome = pd.read_csv(outcome_path, low_memory=False)
    candidate_id = guess_id_column(candidates)
    outcome_id = guess_id_column(outcome)

    candidates["__id"] = candidates[candidate_id].map(normalize_tcga_id)
    outcome["__id"] = outcome[outcome_id].map(normalize_tcga_id)
    time_col = next(
        (column for column in outcome.columns if str(column).lower() in {"os.time", "os_time", "time"}),
        None,
    )
    if time_col is None:
        raise RuntimeError(f"No OS.time column in {outcome_path}")

    merged = candidates.merge(
        outcome[["__id", time_col]],
        on="__id",
        how="inner",
        validate="one_to_one",
    )
    feature_columns = [
        column for column in candidates.columns
        if column not in {candidate_id, "__id"}
        and not str(column).lower().startswith("unnamed")
    ]
    x = merged[feature_columns].apply(pd.to_numeric, errors="coerce")
    x = x.loc[:, x.notna().sum(axis=0) > 0]
    feature_columns = list(x.columns)
    x_values = standardize_matrix(x.to_numpy(dtype=float))
    y = pd.to_numeric(merged[time_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    return x_values[valid], y[valid], feature_columns, int(valid.sum())


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["historical_engine_validation"]

    print("=" * 124)
    print("METABRIC M6.32 - RECONSTRUCTED IAMB VALIDATION AGAINST HISTORICAL TCGA OUTPUTS")
    print("=" * 124)
    print("No METABRIC outcome is used in this stage.")

    result_rows = []
    selection_rows = []
    for modality, spec in settings["configs"].items():
        candidate_path = root / spec["candidate"]
        outcome_path = root / spec["outcome"]
        selected_path = root / spec["selected"]
        for path in (candidate_path, outcome_path, selected_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing historical Paper-1 artifact: {path}")

        x_raw, y_raw, feature_names, n = load_candidate_and_outcome(candidate_path, outcome_path)
        historical = read_feature_list(selected_path)
        historical_norm = {str(value).strip() for value in historical}

        for engine in settings["engines"]:
            x, y = transform_for_ci(x_raw, y_raw, engine)
            selected_indices = iamb_select(
                x,
                y,
                float(spec["alpha"]),
                int(settings["maximum_selected_before_padding"]),
            )
            raw_selected = [feature_names[index] for index in selected_indices]
            padded_indices = pad_selection(
                selected_indices,
                x,
                y,
                int(settings["padding_target"]),
            )
            padded = [feature_names[index] for index in padded_indices]
            jac = jaccard(padded, historical_norm)
            overlap = overlap_coefficient(padded, historical_norm)

            result_rows.append({
                "modality": modality,
                "engine": engine,
                "n": n,
                "candidate_features": len(feature_names),
                "alpha": spec["alpha"],
                "raw_iamb_features": len(raw_selected),
                "padded_features": len(padded),
                "historical_features": len(historical_norm),
                "exact_overlap": len(set(padded) & historical_norm),
                "jaccard": jac,
                "overlap_coefficient": overlap,
                "candidate_path": spec["candidate"],
                "historical_selected_path": spec["selected"],
            })
            for rank, feature in enumerate(padded, 1):
                selection_rows.append({
                    "modality": modality,
                    "engine": engine,
                    "rank": rank,
                    "feature": feature,
                    "historical_selected": feature in historical_norm,
                })

            print(
                f"{modality:12s} {engine:24s} n={n:4d} "
                f"raw={len(raw_selected):3d} padded={len(padded):3d} "
                f"overlap={len(set(padded) & historical_norm):3d} "
                f"Jaccard={jac:.4f}"
            )

    write_csv(out / "m32_engine_validation_results.csv", result_rows)
    write_csv(out / "m32_engine_selected_features.csv", selection_rows)

    engine_summary = []
    for engine in settings["engines"]:
        rows = [row for row in result_rows if row["engine"] == engine]
        engine_summary.append({
            "engine": engine,
            "mean_jaccard": float(np.mean([row["jaccard"] for row in rows])),
            "minimum_jaccard": float(np.min([row["jaccard"] for row in rows])),
            "mean_overlap_coefficient": float(np.mean([row["overlap_coefficient"] for row in rows])),
            "modalities": len(rows),
        })
    engine_summary.sort(key=lambda row: (-row["mean_jaccard"], -row["minimum_jaccard"], row["engine"]))
    best = engine_summary[0]
    reproduced = (
        best["mean_jaccard"] >= float(settings["minimum_mean_jaccard_to_call_reproduced"])
        and best["minimum_jaccard"] >= float(settings["minimum_per_modality_jaccard"])
    )
    decision = {
        "selected_engine": best["engine"],
        "mean_jaccard": best["mean_jaccard"],
        "minimum_jaccard": best["minimum_jaccard"],
        "historical_engine_reproduced": reproduced,
        "status": (
            "HISTORICAL_IAMB_ENGINE_REPRODUCED"
            if reproduced else
            "RECONSTRUCTED_IAMB_ENGINE_NOT_BITWISE_REPRODUCED"
        ),
        "scientific_rule": (
            "The selected engine is chosen only by agreement with historical TCGA outputs. "
            "METABRIC outcomes are not used for engine choice."
        ),
    }
    write_csv(out / "m32_engine_summary.csv", engine_summary)
    (out / "m32_engine_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print("\nEngine summary")
    print_table(
        engine_summary,
        ["engine", "mean_jaccard", "minimum_jaccard", "mean_overlap_coefficient", "modalities"],
    )
    print("\nEngine decision")
    print(json.dumps(decision, indent=2))

    print("\nPASS: historical-engine validation completed. Track B status is explicitly labelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
