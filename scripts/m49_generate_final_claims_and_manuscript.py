from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _metabric_m9_utils import (
    claim_status,
    figure_dir,
    load_config,
    out_dir,
    print_table,
    project_root,
    sha256,
    write_csv,
)


def read_csv_rows(path):
    return pd.read_csv(path, dtype=str, low_memory=False).to_dict("records")


def get_summary(rows, analysis, modality, metric):
    return next(
        row
        for row in rows
        if row["analysis"] == analysis
        and row["modality"] == modality
        and row["metric"] == metric
    )


def main() -> int:
    root = project_root()
    config = load_config(root)
    output = out_dir(root, config)
    figures = figure_dir(root, config)

    print("=" * 124)
    print("METABRIC M9.49R - FINAL NUMERICAL CLAIM TABLES AND FIGURES ONLY")
    print("=" * 124)
    print("No manuscript prose, LaTeX manuscript, or Results/Discussion text is generated.")

    bootstrap = read_csv_rows(
        output / "m46_oof_patient_bootstrap_summary.csv"
    )
    stability = read_csv_rows(
        output / "m47_chance_adjusted_stability_summary.csv"
    )
    methylation = json.loads(
        (
            output / "m48_methylation_transport_summary.json"
        ).read_text(encoding="utf-8")
    )
    track_a = read_csv_rows(
        root / config["files"]["m7_track_a_deltas"]
    )
    protocol = json.loads(
        (output / "m45_m9_protocol.json").read_text(
            encoding="utf-8"
        )
    )

    claim_rows = []

    fixed_c = next(
        row
        for row in track_a
        if row["model_set"] == "clinical_rna_cna"
        and row["metric"] == "delta_c_index_vs_clinical"
    )
    fixed_auc = next(
        row
        for row in track_a
        if row["model_set"] == "clinical_rna_cna"
        and row["metric"] == "delta_auc_5y_vs_clinical"
    )
    claim_rows.append({
        "analysis": "Fixed TCGA RNA+CNA transport",
        "modality": "RNA+CNA",
        "delta_c_mean": float(fixed_c["mean"]),
        "delta_c_ci_low": float(fixed_c["ci_low"]),
        "delta_c_ci_high": float(fixed_c["ci_high"]),
        "delta_auc_5y_mean": float(fixed_auc["mean"]),
        "delta_auc_5y_ci_low": float(fixed_auc["ci_low"]),
        "delta_auc_5y_ci_high": float(fixed_auc["ci_high"]),
        "primary_claim_status": claim_status(
            float(fixed_c["ci_low"]),
            float(fixed_c["ci_high"]),
        ),
        "uncertainty_method": (
            "1000-repetition paired patient bootstrap"
        ),
    })

    combined_c = get_summary(
        bootstrap,
        "combined_reconstructed",
        "Multimodal",
        "delta_c_index",
    )
    combined_auc = get_summary(
        bootstrap,
        "combined_reconstructed",
        "Multimodal",
        "delta_auc_5y",
    )
    claim_rows.append({
        "analysis": "Reconstructed combined nested model",
        "modality": "Multimodal",
        "delta_c_mean": float(combined_c["mean"]),
        "delta_c_ci_low": float(combined_c["ci_low"]),
        "delta_c_ci_high": float(combined_c["ci_high"]),
        "delta_auc_5y_mean": float(combined_auc["mean"]),
        "delta_auc_5y_ci_low": float(combined_auc["ci_low"]),
        "delta_auc_5y_ci_high": float(combined_auc["ci_high"]),
        "primary_claim_status": claim_status(
            float(combined_c["ci_low"]),
            float(combined_c["ci_high"]),
        ),
        "uncertainty_method": (
            "2000-repetition paired patient bootstrap of locked "
            "repeated OOF predictions"
        ),
    })

    for modality in ("RNA", "CNV", "Methylation", "Mutation"):
        c_row = get_summary(
            bootstrap,
            "modality_specific",
            modality,
            "delta_c_index",
        )
        auc_row = get_summary(
            bootstrap,
            "modality_specific",
            modality,
            "delta_auc_5y",
        )
        claim_rows.append({
            "analysis": "Modality-specific nested model",
            "modality": modality,
            "delta_c_mean": float(c_row["mean"]),
            "delta_c_ci_low": float(c_row["ci_low"]),
            "delta_c_ci_high": float(c_row["ci_high"]),
            "delta_auc_5y_mean": float(auc_row["mean"]),
            "delta_auc_5y_ci_low": float(auc_row["ci_low"]),
            "delta_auc_5y_ci_high": float(auc_row["ci_high"]),
            "primary_claim_status": claim_status(
                float(c_row["ci_low"]),
                float(c_row["ci_high"]),
            ),
            "uncertainty_method": (
                "2000-repetition paired patient bootstrap of locked "
                "repeated OOF predictions"
            ),
        })

    write_csv(
        output / "m49_final_claim_table.csv",
        claim_rows,
    )

    labels = [
        f"{row['analysis']}: {row['modality']}"
        for row in claim_rows
    ]
    means = np.asarray(
        [row["delta_c_mean"] for row in claim_rows]
    )
    lows = np.asarray(
        [row["delta_c_ci_low"] for row in claim_rows]
    )
    highs = np.asarray(
        [row["delta_c_ci_high"] for row in claim_rows]
    )
    positions = np.arange(len(claim_rows))

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        means,
        positions,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        capsize=4,
    )
    plt.axvline(0.0, linewidth=1)
    plt.yticks(positions, labels)
    plt.xlabel("Delta Harrell C-index versus clinical-only")
    plt.title(
        "Cross-cohort and nested-validation incremental performance"
    )
    plt.tight_layout()
    plt.savefig(
        figures / "m49_incremental_c_index_forest.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    stability_labels = [
        row["modality"] for row in stability
    ]
    raw_values = [
        float(row["mean_raw_jaccard"])
        for row in stability
    ]
    adjusted_values = [
        float(row["mean_chance_adjusted_overlap"])
        for row in stability
    ]
    x = np.arange(len(stability_labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2,
        raw_values,
        width,
        label="Raw Jaccard",
    )
    plt.bar(
        x + width / 2,
        adjusted_values,
        width,
        label="Chance-adjusted overlap",
    )
    plt.xticks(x, stability_labels)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Stability")
    plt.title(
        "Raw and chance-adjusted feature-selection stability"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures / "m49_chance_adjusted_stability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    report = {
        "metabric_m9_decision": (
            "M9_FINAL_NUMERICAL_INFERENCE_COMPLETE"
        ),
        "protocol_id": protocol["protocol_id"],
        "claim_table_rows": len(claim_rows),
        "methylation_transport_status": methylation["status"],
        "generated_outputs": [
            "m49_final_claim_table.csv",
            "m49_incremental_c_index_forest.png",
            "m49_chance_adjusted_stability.png",
        ],
        "manuscript_text_generated": False,
        "manuscript_text_policy": (
            "Results, Discussion, title selection, and manuscript wording "
            "will be written manually after reviewing the numerical outputs."
        ),
    }
    (
        output / "m49_m9_numerical_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    output_files = [
        output / "m45_m9_protocol.json",
        output / "m46_oof_patient_bootstrap_summary.csv",
        output / "m47_chance_adjusted_stability_summary.csv",
        output / "m48_methylation_transport_summary.json",
        output / "m49_final_claim_table.csv",
        output / "m49_m9_numerical_report.json",
        figures / "m49_incremental_c_index_forest.png",
        figures / "m49_chance_adjusted_stability.png",
    ]
    write_csv(
        output / "m49_output_hashes.csv",
        [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        ],
    )

    print("Final numerical claim table")
    print_table(
        claim_rows,
        [
            "analysis",
            "modality",
            "delta_c_mean",
            "delta_c_ci_low",
            "delta_c_ci_high",
            "delta_auc_5y_mean",
            "delta_auc_5y_ci_low",
            "delta_auc_5y_ci_high",
            "primary_claim_status",
        ],
    )

    print("\nNumerical report")
    print(json.dumps(report, indent=2))

    print(
        "\nPASS: M9 numerical analysis completed. "
        "No manuscript text was generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
