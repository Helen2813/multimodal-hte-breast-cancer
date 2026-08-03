from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage33b_summary_repair_utils import (
    design_gates,
    empirical_anchor_checks,
    json_safe,
    load_json,
    project_root,
    restore_effect_regime_from_scenario_id,
    summarize_checkpoint,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    repair_config = load_json(
        root / "stage33b_null_summary_repair_config.json"
    )
    source = repair_config["source"]
    output = repair_config["output"]
    expected = repair_config["expected"]

    print("=" * 128)
    print("STAGE 128 - REPAIR STAGE 33 SUMMARY WITH NULL SCENARIOS")
    print("=" * 128)

    audit_manifest = load_json(
        root / output["audit_manifest"]
    )
    if (
        audit_manifest["status"]
        != "STAGE33B_NULL_SUMMARY_REPAIR_AUDITED"
    ):
        raise RuntimeError("Stage 33B audit is not complete.")

    stage33_config = load_json(root / source["stage33_config"])
    checkpoint_raw = pd.read_csv(
        root / source["checkpoint"],
        low_memory=False,
        keep_default_na=False,
    )
    checkpoint, _ = restore_effect_regime_from_scenario_id(
        checkpoint_raw
    )
    summary = summarize_checkpoint(checkpoint)

    if len(summary) != expected["summary_row_count"]:
        raise RuntimeError(
            f"Expected {expected['summary_row_count']} summary rows, "
            f"found {len(summary)}."
        )

    null_summary = summary[
        summary["effect_regime"] == "null"
    ].copy()
    benefit_summary = summary[
        summary["effect_regime"]
        == "empirically_calibrated_benefit"
    ].copy()

    if len(null_summary) != expected["null_summary_rows"]:
        raise RuntimeError(
            f"Expected {expected['null_summary_rows']} null rows, "
            f"found {len(null_summary)}."
        )
    if len(benefit_summary) != expected["benefit_summary_rows"]:
        raise RuntimeError(
            f"Expected {expected['benefit_summary_rows']} benefit rows, "
            f"found {len(benefit_summary)}."
        )

    write_csv(summary, root / output["scenario_summary"])
    write_csv(null_summary, root / output["null_summary"])

    naive = summary[
        summary["method"] == "naive_full"
    ][
        [
            "scenario_id",
            "sample_size",
            "sequencing_level",
            "sequencing_strength",
            "effect_regime",
            "mean_primary_truth_days",
            "mean_secondary_truth_days",
            "mean_estimate_days",
            "bias_days",
            "mean_target_drift_days",
            "mean_residual_omitted_sequence_bias_days",
            "mean_weighted_chemo_smd",
            "primary_if_coverage",
            "positive_ci_exclusion_rate",
            "negative_ci_exclusion_rate",
        ]
    ].copy()
    write_csv(naive, root / output["naive_decomposition"])

    anchor_checks = empirical_anchor_checks(
        summary,
        stage33_config,
    )
    write_csv(
        anchor_checks,
        root / output["empirical_anchor_check"],
    )

    gates = design_gates(summary, stage33_config)
    write_csv(gates, root / output["design_gates"])

    original_summary = pd.read_csv(
        root / source["original_scenario_summary"],
        low_memory=False,
        keep_default_na=False,
    )
    comparison = pd.DataFrame([
        {
            "version": "original_stage126",
            "summary_rows": len(original_summary),
            "null_rows": int(
                (
                    original_summary["effect_regime"]
                    == "null"
                ).sum()
            ),
            "benefit_rows": int(
                (
                    original_summary["effect_regime"]
                    == "empirically_calibrated_benefit"
                ).sum()
            ),
        },
        {
            "version": "repaired_stage33b",
            "summary_rows": len(summary),
            "null_rows": len(null_summary),
            "benefit_rows": len(benefit_summary),
        },
    ])
    write_csv(
        comparison,
        root / output["original_vs_repaired"],
    )

    adjusted = gates[
        gates["method"].isin(
            ["adjusted_full", "sequencing_aware"]
        )
    ]
    repaired_ready = bool(
        gates["success_gate"].all()
        and gates["coverage_gate"].all()
        and gates["included_balance_gate"].all()
        and adjusted["bias_gate"].fillna(False).all()
        and anchor_checks["pass"].all()
    )

    null_naive = null_summary[
        null_summary["method"] == "naive_full"
    ].copy()
    empirical_null_naive = null_naive[
        null_naive["sequencing_level"] == "empirical"
    ]

    final = {
        "status": (
            "STAGE33B_REPAIRED_READY_FOR_CONFIRMATORY_LOCK"
            if repaired_ready
            else "STAGE33B_REPAIRED_REQUIRES_REVIEW"
        ),
        "repair_id": audit_manifest["repair_id"],
        "stage33_simulation_id": audit_manifest[
            "stage33_simulation_id"
        ],
        "repaired_ready_for_confirmatory_lock": repaired_ready,
        "checkpoint_rows_reused": len(checkpoint),
        "simulation_rows_rerun": 0,
        "original_summary_rows": len(original_summary),
        "repaired_summary_rows": len(summary),
        "restored_null_summary_rows": len(null_summary),
        "all_empirical_anchor_checks_passed": bool(
            anchor_checks["pass"].all()
        ),
        "null_scenario_diagnostics": {
            "maximum_absolute_naive_mean_estimate_days": float(
                null_naive["mean_estimate_days"].abs().max()
            ),
            "maximum_naive_positive_ci_exclusion_rate": float(
                null_naive[
                    "positive_ci_exclusion_rate"
                ].max()
            ),
            "empirical_sequence_naive_rows": json_safe(
                empirical_null_naive.to_dict("records")
            ),
        },
        "scenario_summary": json_safe(
            summary.to_dict("records")
        ),
        "design_gates": json_safe(
            gates.to_dict("records")
        ),
        "empirical_anchor_checks": json_safe(
            anchor_checks.to_dict("records")
        ),
        "repair_reason": repair_config["repair_reason"],
        "boundary": repair_config["boundary"],
        "next_action": (
            "Review the restored null scenarios and then lock the "
            "confirmatory simulation only if the repaired gates pass."
            if repaired_ready
            else "Review the restored null-scenario failures before "
            "any confirmatory simulation."
        ),
    }
    write_json(final, root / output["final_json"])

    figure_dir = root / output["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    for n in sorted(null_summary["sample_size"].unique()):
        panel = null_summary[
            null_summary["sample_size"] == n
        ]
        plt.figure(figsize=(8.2, 5.0))
        for method in expected["methods"]:
            method_frame = panel[
                panel["method"] == method
            ].sort_values("sequencing_strength")
            plt.plot(
                method_frame["sequencing_strength"],
                method_frame["mean_estimate_days"],
                marker="o",
                label=method,
            )
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Sequencing strength")
        plt.ylabel("Mean estimated RMST contrast under true null (days)")
        plt.title(f"Restored true-null scenarios, n={n}")
        plt.legend()
        plt.tight_layout()
        base = root / output["null_figure"]
        path = base.with_name(
            base.stem + f"_n{int(n)}" + base.suffix
        )
        plt.savefig(path, dpi=220)
        plt.close()

    for effect in summary["effect_regime"].unique():
        for n in sorted(summary["sample_size"].unique()):
            panel = summary[
                (summary["effect_regime"] == effect)
                & (summary["sample_size"] == n)
            ]
            plt.figure(figsize=(8.2, 5.0))
            for method in expected["methods"]:
                method_frame = panel[
                    panel["method"] == method
                ].sort_values("sequencing_strength")
                plt.plot(
                    method_frame["sequencing_strength"],
                    method_frame["bias_days"],
                    marker="o",
                    label=method,
                )
            plt.axhline(0.0, linewidth=1.0)
            plt.xlabel("Sequencing strength")
            plt.ylabel("Mean bias relative to method-specific truth (days)")
            plt.title(f"Repaired pilot bias: {effect}, n={n}")
            plt.legend()
            plt.tight_layout()
            base = root / output["bias_figure"]
            path = base.with_name(
                base.stem
                + f"_{effect}_n{int(n)}"
                + base.suffix
            )
            plt.savefig(path, dpi=220)
            plt.close()

    print("Original versus repaired summary")
    print(comparison.to_string(index=False))
    print("\nRestored null scenarios")
    print(
        null_summary[
            [
                "scenario_id",
                "method",
                "mean_primary_truth_days",
                "mean_estimate_days",
                "bias_days",
                "primary_if_coverage",
                "positive_ci_exclusion_rate",
                "mean_weighted_chemo_smd",
            ]
        ].to_string(index=False)
    )
    print("\nRepaired design gates")
    print(gates.to_string(index=False))
    print("\nFinal repaired decision")
    print(json.dumps(json_safe(final), indent=2))
    print(
        "\nPASS: Stage 33 summaries repaired without rerunning "
        "any simulation repetition."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
