from __future__ import annotations

import json

from _metabric_m7_utils import (
    load_config,
    out_dir,
    print_table,
    project_root,
    read_rows,
    sha256,
    write_csv,
)


def find_row(rows, model_set, metric):
    return next(
        row
        for row in rows
        if row["model_set"] == model_set
        and row["metric"] == metric
    )


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M7.39 - FULL CORE ANALYSIS REPORT")
    print("=" * 124)

    protocol = json.loads(
        (out / "m36_m7_full_core_protocol.json")
        .read_text(encoding="utf-8")
    )
    track_a = read_rows(
        out / "m37_track_a_full_results.csv"
    )
    track_a_deltas = read_rows(
        out / "m37_track_a_paired_delta_summary.csv"
    )
    prefix = read_rows(
        out / "m37_pilot_prefix_verification.csv"
    )
    track_b = json.loads(
        (out / "m38_track_b_full_summary.json")
        .read_text(encoding="utf-8")
    )
    repeat_summary = read_rows(
        out / "m38_repeat_level_summary.csv"
    )

    delta_c = find_row(
        track_a_deltas,
        "clinical_rna_cna",
        "delta_c_index_vs_clinical",
    )
    delta_auc = find_row(
        track_a_deltas,
        "clinical_rna_cna",
        "delta_auc_5y_vs_clinical",
    )

    checks = [
        {
            "check": "M7 protocol locked",
            "observed": protocol["protocol_id"],
            "pass": (
                protocol["status"]
                == "METABRIC_M7_FULL_CORE_ANALYSIS_LOCKED"
            ),
        },
        {
            "check": "Track A 1000 paired bootstraps",
            "observed": delta_c["repetitions"],
            "pass": int(float(delta_c["repetitions"])) == 1000,
        },
        {
            "check": "Track A pilot prefix reproduced",
            "observed": prefix[0]["maximum_absolute_difference"],
            "pass": str(prefix[0]["pass"]).lower() == "true",
        },
        {
            "check": "Track B 20x5 folds completed",
            "observed": track_b["completed_fold_fits"],
            "pass": int(track_b["completed_fold_fits"]) == 100,
        },
        {
            "check": "Track B label retained",
            "observed": track_b["historical_engine_status"],
            "pass": (
                track_b["historical_engine_status"]
                == cfg["track_b"]["historical_engine_status"]
            ),
        },
    ]
    if not all(bool(row["pass"]) for row in checks):
        decision = "M7_FULL_CORE_ANALYSIS_INCOMPLETE"
    else:
        decision = (
            "M7_FULL_CORE_ANALYSIS_COMPLETE_"
            "TRACK_B_RECONSTRUCTED"
        )

    track_a_interpretation = {
        "full_panel_delta_c_mean": float(delta_c["mean"]),
        "full_panel_delta_c_ci": [
            float(delta_c["ci_low"]),
            float(delta_c["ci_high"]),
        ],
        "full_panel_delta_auc_5y_mean": float(delta_auc["mean"]),
        "full_panel_delta_auc_5y_ci": [
            float(delta_auc["ci_low"]),
            float(delta_auc["ci_high"]),
        ],
        "incremental_discrimination_claim_allowed": (
            float(delta_c["ci_low"]) > 0
            or float(delta_auc["ci_low"]) > 0
        ),
        "wording": (
            "Report the fixed-panel result as external transportability "
            "evidence, not as proof of incremental prognostic benefit."
        ),
    }

    report = {
        "metabric_m7_decision": decision,
        "protocol_id": protocol["protocol_id"],
        "track_a": track_a_interpretation,
        "track_b": {
            "status": track_b["status"],
            "mean_repeat_delta_c_index": track_b[
                "mean_repeat_delta_c_index"
            ],
            "sd_repeat_delta_c_index": track_b[
                "sd_repeat_delta_c_index"
            ],
            "mean_repeat_delta_auc_5y": track_b[
                "mean_repeat_delta_auc_5y"
            ],
            "sd_repeat_delta_auc_5y": track_b[
                "sd_repeat_delta_auc_5y"
            ],
            "mean_within_repeat_jaccard": track_b[
                "mean_within_repeat_jaccard"
            ],
            "mean_between_repeat_stable_set_jaccard": track_b[
                "mean_between_repeat_stable_set_jaccard"
            ],
            "historical_engine_status": track_b[
                "historical_engine_status"
            ],
            "wording": (
                "Report as a leakage-controlled reconstructed "
                "dependency-aware replication. Do not describe it as "
                "an exact rerun of the published IAMB engine."
            ),
        },
        "next_step": (
            "Run M8 modality-specific RNA, CNV, methylation, and mutation "
            "replications, followed by gene/pathway concordance and "
            "publication-asset generation."
        ),
        "scientific_boundary": (
            "Performance direction does not determine inclusion. "
            "Negative or null transport is a valid external-validation result."
        ),
    }
    (out / "m39_m7_full_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_csv(out / "m39_m7_checks.csv", checks)
    write_csv(out / "m39_m7_decision.csv", [{
        "metabric_m7_decision": decision,
        "protocol_id": protocol["protocol_id"],
        "track_a_delta_c_mean": delta_c["mean"],
        "track_a_delta_c_ci_low": delta_c["ci_low"],
        "track_a_delta_c_ci_high": delta_c["ci_high"],
        "track_a_delta_auc_5y_mean": delta_auc["mean"],
        "track_a_delta_auc_5y_ci_low": delta_auc["ci_low"],
        "track_a_delta_auc_5y_ci_high": delta_auc["ci_high"],
        "track_b_mean_repeat_delta_c": track_b[
            "mean_repeat_delta_c_index"
        ],
        "track_b_mean_repeat_delta_auc_5y": track_b[
            "mean_repeat_delta_auc_5y"
        ],
        "track_b_mean_within_repeat_jaccard": track_b[
            "mean_within_repeat_jaccard"
        ],
        "track_b_engine_status": track_b[
            "historical_engine_status"
        ],
        "next_step": report["next_step"],
    }])

    inventory_paths = [
        out / "m36_m7_full_core_protocol.json",
        out / "m37_track_a_full_results.csv",
        out / "m37_track_a_bootstrap_summary.csv",
        out / "m37_track_a_paired_delta_summary.csv",
        out / "m38_repeat_level_oof_results.csv",
        out / "m38_repeat_level_summary.csv",
        out / "m38_selection_frequency.csv",
        out / "m38_modality_composition.csv",
        out / "m38_track_b_full_summary.json",
        out / "m39_m7_full_report.json",
    ]
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in inventory_paths
    ]
    write_csv(out / "m39_m7_output_hashes.csv", inventory)

    print(f"Decision: {decision}")
    print("\nCompletion checks")
    print_table(checks, ["check", "observed", "pass"])
    print("\nTrack A full results")
    print_table(
        track_a,
        [
            "model_set",
            "harrell_c_index",
            "uno_c_10y",
            "ipcw_auc_5y",
            "ipcw_auc_10y",
            "integrated_brier_1_to_10y",
            "calibration_slope",
        ],
    )
    print("\nTrack A paired deltas")
    print_table(
        track_a_deltas,
        [
            "model_set",
            "metric",
            "mean",
            "ci_low",
            "ci_high",
            "fraction_positive",
        ],
    )
    print("\nTrack B repeated-split summary")
    print_table(
        repeat_summary,
        [
            "metric",
            "mean",
            "sd",
            "q025",
            "q975",
            "fraction_positive",
        ],
    )
    print("\nFinal report")
    print(json.dumps(report, indent=2))

    if decision == "M7_FULL_CORE_ANALYSIS_INCOMPLETE":
        raise RuntimeError("M7 full core analysis incomplete.")

    print("\nPASS: M7 full core analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
