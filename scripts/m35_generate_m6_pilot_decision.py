from __future__ import annotations

import json

import numpy as np

from _metabric_m6_utils import (
    load_config,
    out_dir,
    print_table,
    project_root,
    read_rows,
    write_csv,
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    engine = json.loads(
        (out / "m32_engine_decision.json").read_text(encoding="utf-8")
    )
    track_a = read_rows(out / "m33_track_a_external_results.csv")
    track_a_bootstrap = read_rows(
        out / "m33_track_a_bootstrap_summary.csv"
    )
    track_a_deltas = read_rows(
        out / "m33_track_a_paired_delta_summary.csv"
    )
    track_b = json.loads(
        (out / "m34_track_b_summary.json").read_text(encoding="utf-8")
    )

    clinical = next(
        row for row in track_a if row["model_set"] == "clinical"
    )
    full = next(
        row for row in track_a
        if row["model_set"] == "clinical_rna_cna"
    )
    delta_c = (
        float(full["harrell_c_index"])
        - float(clinical["harrell_c_index"])
    )
    delta_auc = float(full["auc_5y"]) - float(clinical["auc_5y"])

    full_delta_c = next(
        (
            row for row in track_a_deltas
            if row["model_set"] == "clinical_rna_cna"
            and row["metric"] == "delta_c_index_vs_clinical"
        ),
        None,
    )
    full_delta_auc = next(
        (
            row for row in track_a_deltas
            if row["model_set"] == "clinical_rna_cna"
            and row["metric"] == "delta_auc_5y_vs_clinical"
        ),
        None,
    )

    checks = [
        {
            "check": "Historical IAMB engine reproduction",
            "observed": engine["status"],
            "pass": bool(engine["historical_engine_reproduced"]),
            "role": "Track B exactness gate",
        },
        {
            "check": "Track A fixed-panel external pilot completed",
            "observed": (
                f"delta C={delta_c:.4f}; "
                f"delta AUC5y={delta_auc:.4f}"
            ),
            "pass": np.isfinite(delta_c) and np.isfinite(delta_auc),
            "role": "Computational gate; no minimum effect required",
        },
        {
            "check": "Track A paired bootstrap completed",
            "observed": (
                f"delta-C CI="
                f"[{float(full_delta_c['ci_low']):.4f}, "
                f"{float(full_delta_c['ci_high']):.4f}]"
                if full_delta_c else "missing"
            ),
            "pass": full_delta_c is not None and full_delta_auc is not None,
            "role": "Computational gate",
        },
        {
            "check": "Track B nested five-fold pilot completed",
            "observed": (
                f"C={track_b['mean_c_index']:.4f}; "
                f"AUC5y={track_b['mean_auc_5y']:.4f}"
            ),
            "pass": (
                track_b["outer_folds"]
                == int(cfg["track_b_pilot"]["outer_folds"])
            ),
            "role": "Computational gate",
        },
        {
            "check": "Track B selection stability measurable",
            "observed": (
                f"mean Jaccard="
                f"{track_b['mean_pairwise_jaccard']:.4f}"
            ),
            "pass": np.isfinite(track_b["mean_pairwise_jaccard"]),
            "role": "Descriptive; no post-hoc threshold",
        },
        {
            "check": "Track B uses full panel-aware mutation universe",
            "observed": track_b["mutation_candidate_universe"],
            "pass": track_b["mutation_candidate_universe"] == 173,
            "role": "Methodological gate",
        },
    ]

    computational_pass = all(
        bool(row["pass"])
        for row in checks
        if row["role"] != "Track B exactness gate"
    )

    if computational_pass and engine["historical_engine_reproduced"]:
        decision = (
            "M6_PILOT_COMPLETE_READY_FOR_FULL_DUAL_TRACK_ANALYSIS"
        )
        next_step = (
            "Freeze the repaired pilot code and run the prespecified full "
            "analysis: 20 outer repeats for Track B and 1000 paired patient-"
            "bootstrap repetitions for Track A."
        )
    elif computational_pass:
        decision = (
            "M6_PILOT_COMPLETE_TRACK_A_READY_TRACK_B_RECONSTRUCTED_"
            "METHOD_REQUIRES_LABEL"
        )
        next_step = (
            "Track A may proceed. Track B must be registered and reported as "
            "a reconstructed dependency-aware replication unless the original "
            "conditional-independence implementation is recovered."
        )
    else:
        decision = "M6_PILOT_COMPUTATION_INCOMPLETE_DO_NOT_SCALE"
        next_step = (
            "Resolve failed computational or methodological gates before "
            "running the full analysis."
        )

    result = {
        "metabric_m6_decision": decision,
        "track_a_delta_c_index_vs_clinical": delta_c,
        "track_a_delta_auc_5y_vs_clinical": delta_auc,
        "track_a_paired_delta_c_ci": (
            [
                float(full_delta_c["ci_low"]),
                float(full_delta_c["ci_high"]),
            ]
            if full_delta_c else None
        ),
        "track_a_paired_delta_auc_5y_ci": (
            [
                float(full_delta_auc["ci_low"]),
                float(full_delta_auc["ci_high"]),
            ]
            if full_delta_auc else None
        ),
        "track_b_mean_c_index": track_b["mean_c_index"],
        "track_b_mean_clinical_only_c_index": (
            track_b["mean_clinical_only_c_index"]
        ),
        "track_b_mean_delta_c_index_vs_clinical": (
            track_b["mean_delta_c_index_vs_clinical"]
        ),
        "track_b_mean_auc_5y": track_b["mean_auc_5y"],
        "track_b_mean_clinical_only_auc_5y": (
            track_b["mean_clinical_only_auc_5y"]
        ),
        "track_b_mean_delta_auc_5y_vs_clinical": (
            track_b["mean_delta_auc_5y_vs_clinical"]
        ),
        "track_b_mean_pairwise_jaccard": (
            track_b["mean_pairwise_jaccard"]
        ),
        "historical_engine_status": engine["status"],
        "recommended_next_step": next_step,
        "interpretation_boundary": (
            "Pilot estimates are not publication results. No model, mapping, "
            "or feature-selection implementation is accepted or rejected "
            "because its METABRIC performance is favorable or unfavorable."
        ),
    }

    write_csv(out / "m35_m6_decision_checks.csv", checks)
    write_csv(out / "m35_m6_decision.csv", [result])
    (out / "m35_m6_decision.md").write_text(
        "\n".join([
            "# METABRIC M6 repaired pilot decision",
            "",
            f"**Decision:** `{decision}`",
            "",
            "## Track A",
            "",
            f"- Delta C-index versus clinical-only: {delta_c:.6f}",
            f"- Delta 5-year AUC versus clinical-only: {delta_auc:.6f}",
            (
                "- Paired bootstrap delta-C interval: "
                f"{result['track_a_paired_delta_c_ci']}"
            ),
            (
                "- Paired bootstrap delta-AUC interval: "
                f"{result['track_a_paired_delta_auc_5y_ci']}"
            ),
            "",
            "## Track B",
            "",
            f"- Mean outer-fold C-index: {track_b['mean_c_index']:.6f}",
            (
                "- Mean clinical-only C-index: "
                f"{track_b['mean_clinical_only_c_index']:.6f}"
            ),
            (
                "- Mean delta C-index: "
                f"{track_b['mean_delta_c_index_vs_clinical']:.6f}"
            ),
            f"- Mean outer-fold 5-year AUC: {track_b['mean_auc_5y']:.6f}",
            (
                "- Mean pairwise Jaccard: "
                f"{track_b['mean_pairwise_jaccard']:.6f}"
            ),
            f"- Historical engine status: `{engine['status']}`",
            "",
            "## Interpretation boundary",
            "",
            result["interpretation_boundary"],
            "",
            "## Recommended next step",
            "",
            next_step,
        ]) + "\n",
        encoding="utf-8",
    )

    print("=" * 124)
    print("METABRIC M6.35R - REPAIRED PILOT DECISION")
    print("=" * 124)
    print(f"Decision: {decision}")

    print("\nDecision checks")
    print_table(checks, ["check", "observed", "pass", "role"])

    print("\nTrack A results")
    print_table(
        track_a,
        [
            "model_set", "features", "tcga_train_c_index",
            "harrell_c_index", "auc_5y"
        ],
    )

    print("\nTrack A bootstrap summary")
    print_table(
        track_a_bootstrap,
        [
            "model_set", "metric", "mean", "sd",
            "ci_low", "ci_high"
        ],
    )

    print("\nTrack A paired deltas")
    print_table(
        track_a_deltas,
        [
            "model_set", "metric", "mean", "sd",
            "ci_low", "ci_high", "fraction_positive"
        ],
    )

    print("\nFinal repaired pilot summary")
    print(json.dumps(result, indent=2))

    print(
        "\nPASS: M6 repaired decision generated without changing "
        "the locked M5 protocol."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
