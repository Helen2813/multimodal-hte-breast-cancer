from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage31_simulation_utils import (
    load_json,
    project_root,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage31_sequencing_simulation_config.json"
    )
    output = config["outputs"]
    gates = config["pilot_gates"]

    print("=" * 128)
    print("STAGE 120 - SUMMARIZE SEQUENCING SIMULATION PILOT")
    print("=" * 128)

    checkpoint = pd.read_csv(
        root / output["checkpoint"],
        low_memory=False,
    )
    checkpoint["success_bool"] = (
        checkpoint["success"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    group_columns = [
        "scenario_id",
        "sample_size",
        "sequencing_strength_name",
        "sequencing_strength",
        "method",
    ]
    rows = []
    for keys, group in checkpoint.groupby(group_columns):
        (
            scenario_id,
            sample_size,
            strength_name,
            strength_value,
            method,
        ) = keys
        successful = group[group["success_bool"]].copy()
        total = len(group)

        row = {
            "scenario_id": scenario_id,
            "sample_size": int(sample_size),
            "sequencing_strength_name": strength_name,
            "sequencing_strength": float(strength_value),
            "method": method,
            "attempted_repetitions": total,
            "successful_repetitions": len(successful),
            "success_fraction": (
                len(successful) / total if total else np.nan
            ),
        }

        if len(successful):
            for column in (
                "truth_days",
                "estimate_days",
                "bias_days",
                "squared_error",
                "diagnostic_if_se_days",
                "covered",
                "max_abs_weighted_smd",
                "ato_ess_fraction_treated",
                "ato_ess_fraction_control",
                "events",
                "treated_events",
                "control_events",
                "chemo_prevalence",
                "treated_fraction_full",
                "treated_fraction_no_chemo",
            ):
                successful[column] = pd.to_numeric(
                    successful[column],
                    errors="raise",
                )

            row.update({
                "mean_truth_days": float(
                    successful["truth_days"].mean()
                ),
                "mean_estimate_days": float(
                    successful["estimate_days"].mean()
                ),
                "bias_days": float(
                    successful["bias_days"].mean()
                ),
                "empirical_sd_days": float(
                    successful["estimate_days"].std(ddof=1)
                ),
                "rmse_days": float(
                    math.sqrt(
                        successful["squared_error"].mean()
                    )
                ),
                "mean_if_se_days": float(
                    successful[
                        "diagnostic_if_se_days"
                    ].mean()
                ),
                "if_coverage": float(
                    successful["covered"].mean()
                ),
                "mean_max_abs_weighted_smd": float(
                    successful[
                        "max_abs_weighted_smd"
                    ].mean()
                ),
                "mean_ato_ess_fraction_treated": float(
                    successful[
                        "ato_ess_fraction_treated"
                    ].mean()
                ),
                "mean_ato_ess_fraction_control": float(
                    successful[
                        "ato_ess_fraction_control"
                    ].mean()
                ),
                "mean_events": float(
                    successful["events"].mean()
                ),
                "mean_treated_events": float(
                    successful["treated_events"].mean()
                ),
                "mean_control_events": float(
                    successful["control_events"].mean()
                ),
                "mean_chemo_prevalence": float(
                    successful["chemo_prevalence"].mean()
                ),
                "mean_treated_fraction_full": float(
                    successful[
                        "treated_fraction_full"
                    ].mean()
                ),
                "mean_treated_fraction_no_chemo": float(
                    successful[
                        "treated_fraction_no_chemo"
                    ].mean()
                ),
            })
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["sample_size", "sequencing_strength", "method"]
    ).reset_index(drop=True)
    write_csv(summary, root / output["scenario_summary"])

    design_rows = []
    for _, row in summary.iterrows():
        adjusted_method = row["method"] in {
            "adjusted_full",
            "sequencing_aware",
        }
        design_rows.append({
            "scenario_id": row["scenario_id"],
            "method": row["method"],
            "success_gate": bool(
                row["success_fraction"]
                >= float(gates["minimum_success_fraction"])
            ),
            "bias_gate_applicable": adjusted_method,
            "bias_gate": (
                bool(
                    abs(row["bias_days"])
                    <= float(
                        gates[
                            "maximum_absolute_bias_days_for_adjusted_estimators"
                        ]
                    )
                )
                if adjusted_method
                and pd.notna(row.get("bias_days"))
                else None
            ),
            "coverage_gate_applicable": adjusted_method,
            "coverage_gate": (
                bool(
                    row["if_coverage"]
                    >= float(
                        gates[
                            "minimum_if_coverage_for_adjusted_estimators"
                        ]
                    )
                )
                if adjusted_method
                and pd.notna(row.get("if_coverage"))
                else None
            ),
            "balance_gate": (
                bool(
                    row["mean_max_abs_weighted_smd"]
                    <= float(
                        gates[
                            "maximum_mean_abs_weighted_smd"
                        ]
                    )
                )
                if pd.notna(
                    row.get(
                        "mean_max_abs_weighted_smd"
                    )
                )
                else False
            ),
        })
    design = pd.DataFrame(design_rows)
    write_csv(design, root / output["design_summary"])

    adjusted = design[
        design["method"].isin(
            ["adjusted_full", "sequencing_aware"]
        )
    ]
    pilot_ready = bool(
        design["success_gate"].all()
        and design["balance_gate"].all()
        and adjusted["bias_gate"].fillna(False).all()
        and adjusted[
            "coverage_gate"
        ].fillna(False).all()
    )

    final = {
        "status": (
            "STAGE31_SIMULATION_PILOT_READY_FOR_FULL_RUN"
            if pilot_ready
            else "STAGE31_SIMULATION_PILOT_REQUIRES_REVIEW"
        ),
        "simulation_id": load_json(
            root / output["manifest"]
        )["simulation_id"],
        "pilot_ready_for_full_run": pilot_ready,
        "scenario_summary": summary.to_dict("records"),
        "design_gates": design.to_dict("records"),
        "interpretation_boundary": config["boundary"],
        "next_action": (
            "Lock and run a larger confirmatory simulation."
            if pilot_ready
            else "Review failures, calibration, and estimator behaviour "
            "before any confirmatory simulation."
        ),
    }
    write_json(final, root / output["final_json"])

    figure_dir = root / output["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "naive_full",
        "adjusted_full",
        "sequencing_aware",
    ]
    for sample_size in sorted(summary["sample_size"].unique()):
        subset = summary[
            summary["sample_size"] == sample_size
        ].copy()
        plt.figure(figsize=(8.0, 5.0))
        for method in methods:
            method_frame = subset[
                subset["method"] == method
            ].sort_values("sequencing_strength")
            plt.plot(
                method_frame["sequencing_strength"],
                method_frame["bias_days"],
                marker="o",
                label=method,
            )
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Sequencing strength")
        plt.ylabel("Mean bias in ATO RMST estimate (days)")
        plt.title(
            f"Simulation pilot bias, n={sample_size}"
        )
        plt.legend()
        plt.tight_layout()
        path = root / output["bias_figure"]
        path = path.with_name(
            path.stem + f"_n{sample_size}" + path.suffix
        )
        plt.savefig(path, dpi=220)
        plt.close()

        plt.figure(figsize=(8.0, 5.0))
        for method in methods:
            method_frame = subset[
                subset["method"] == method
            ].sort_values("sequencing_strength")
            plt.plot(
                method_frame["sequencing_strength"],
                method_frame["if_coverage"],
                marker="o",
                label=method,
            )
        plt.axhline(0.95, linewidth=1.0)
        plt.xlabel("Sequencing strength")
        plt.ylabel("Empirical 95% IF-interval coverage")
        plt.title(
            f"Simulation pilot coverage, n={sample_size}"
        )
        plt.legend()
        plt.tight_layout()
        path = root / output["coverage_figure"]
        path = path.with_name(
            path.stem + f"_n{sample_size}" + path.suffix
        )
        plt.savefig(path, dpi=220)
        plt.close()

    print("Scenario summary")
    print(summary.to_string(index=False))
    print("\nPilot design gates")
    print(design.to_string(index=False))
    print("\nFinal pilot decision")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 31 simulation pilot summarized. "
        "No manuscript prose was generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
