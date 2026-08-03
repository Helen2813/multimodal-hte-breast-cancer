from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage33_simulation_v2_utils import (
    load_json,
    project_root,
    write_csv,
    write_json,
)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="raise")


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage33_sequence_simulation_pilot_v2_config.json"
    )
    output = config["outputs"]
    gates = config["pilot_gates"]
    manifest = load_json(root / output["manifest"])

    print("=" * 128)
    print("STAGE 126 - SUMMARIZE REVISED SIMULATION PILOT")
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
        "sequencing_level",
        "sequencing_strength",
        "effect_regime",
        "treatment_log_hazard_effect",
        "method",
    ]
    rows = []
    for keys, group in checkpoint.groupby(group_columns):
        successful = group[group["success_bool"]].copy()
        (
            scenario_id,
            sample_size,
            sequence_level,
            sequence_strength,
            effect_regime,
            treatment_effect,
            method,
        ) = keys
        row = {
            "scenario_id": scenario_id,
            "sample_size": int(sample_size),
            "sequencing_level": sequence_level,
            "sequencing_strength": float(sequence_strength),
            "effect_regime": effect_regime,
            "treatment_log_hazard_effect": float(
                treatment_effect
            ),
            "method": method,
            "attempted_repetitions": len(group),
            "successful_repetitions": len(successful),
            "success_fraction": (
                len(successful) / len(group)
                if len(group)
                else np.nan
            ),
        }

        if len(successful):
            for column in (
                "primary_truth_days",
                "estimate_days",
                "primary_bias_days",
                "primary_squared_error",
                "diagnostic_if_se_days",
                "primary_covered",
                "included_covariate_max_abs_weighted_smd",
                "unweighted_chemo_smd",
                "weighted_chemo_smd",
                "ato_ess_fraction_treated",
                "ato_ess_fraction_control",
                "n_analysis",
                "treated",
                "control",
                "events",
                "treated_events",
                "control_events",
                "full_treated_fraction",
                "full_chemo_fraction_treated",
                "full_chemo_fraction_control",
                "strict_population_n",
                "strict_treated_n",
                "strict_control_n",
                "strict_treated_events",
                "strict_control_events",
                "secondary_truth_days",
                "secondary_bias_days",
                "target_drift_days",
                "residual_omitted_sequence_bias_days",
            ):
                successful[column] = pd.to_numeric(
                    successful[column],
                    errors="coerce",
                )

            row.update({
                "mean_primary_truth_days": float(
                    successful["primary_truth_days"].mean()
                ),
                "mean_estimate_days": float(
                    successful["estimate_days"].mean()
                ),
                "bias_days": float(
                    successful["primary_bias_days"].mean()
                ),
                "empirical_sd_days": float(
                    successful["estimate_days"].std(ddof=1)
                ),
                "rmse_days": float(
                    math.sqrt(
                        successful[
                            "primary_squared_error"
                        ].mean()
                    )
                ),
                "mean_if_se_days": float(
                    successful[
                        "diagnostic_if_se_days"
                    ].mean()
                ),
                "primary_if_coverage": float(
                    successful["primary_covered"].mean()
                ),
                "mean_included_covariate_max_abs_weighted_smd": float(
                    successful[
                        "included_covariate_max_abs_weighted_smd"
                    ].mean()
                ),
                "mean_unweighted_chemo_smd": float(
                    successful["unweighted_chemo_smd"].mean()
                ),
                "mean_weighted_chemo_smd": float(
                    successful["weighted_chemo_smd"].mean()
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
                "mean_n_analysis": float(
                    successful["n_analysis"].mean()
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
                "mean_full_treated_fraction": float(
                    successful[
                        "full_treated_fraction"
                    ].mean()
                ),
                "mean_full_chemo_fraction_treated": float(
                    successful[
                        "full_chemo_fraction_treated"
                    ].mean()
                ),
                "mean_full_chemo_fraction_control": float(
                    successful[
                        "full_chemo_fraction_control"
                    ].mean()
                ),
                "mean_strict_population_n": float(
                    successful[
                        "strict_population_n"
                    ].mean()
                ),
                "mean_strict_treated_n": float(
                    successful["strict_treated_n"].mean()
                ),
                "mean_strict_control_n": float(
                    successful["strict_control_n"].mean()
                ),
                "mean_strict_treated_events": float(
                    successful[
                        "strict_treated_events"
                    ].mean()
                ),
                "mean_strict_control_events": float(
                    successful[
                        "strict_control_events"
                    ].mean()
                ),
                "mean_secondary_truth_days": float(
                    successful[
                        "secondary_truth_days"
                    ].mean()
                )
                if successful[
                    "secondary_truth_days"
                ].notna().any()
                else np.nan,
                "mean_secondary_bias_days": float(
                    successful[
                        "secondary_bias_days"
                    ].mean()
                )
                if successful[
                    "secondary_bias_days"
                ].notna().any()
                else np.nan,
                "mean_target_drift_days": float(
                    successful["target_drift_days"].mean()
                )
                if successful[
                    "target_drift_days"
                ].notna().any()
                else np.nan,
                "mean_residual_omitted_sequence_bias_days": float(
                    successful[
                        "residual_omitted_sequence_bias_days"
                    ].mean()
                )
                if successful[
                    "residual_omitted_sequence_bias_days"
                ].notna().any()
                else np.nan,
            })
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        [
            "sample_size",
            "effect_regime",
            "sequencing_strength",
            "method",
        ]
    ).reset_index(drop=True)
    write_csv(summary, root / output["scenario_summary"])

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
        ]
    ].copy()
    write_csv(naive, root / output["naive_decomposition"])

    anchor = summary[
        (summary["sample_size"] == 559)
        & (summary["sequencing_level"] == "empirical")
        & (
            summary["effect_regime"]
            == "empirically_calibrated_benefit"
        )
        & (summary["method"] == "sequencing_aware")
    ].copy()
    if len(anchor) != 1:
        raise RuntimeError(
            "Expected one empirical-anchor sequencing-aware row."
        )
    anchor_row = anchor.iloc[0]
    expected = config["expected"]
    tolerances = gates["empirical_anchor_tolerances"]

    anchor_checks = pd.DataFrame([
        {
            "metric": "full_treated_fraction",
            "observed": anchor_row[
                "mean_full_treated_fraction"
            ],
            "target": (
                expected["full_treated_anchor"]
                / expected["full_cohort_n_anchor"]
            ),
            "tolerance": tolerances[
                "full_treated_fraction"
            ],
        },
        {
            "metric": "chemo_fraction_treated",
            "observed": anchor_row[
                "mean_full_chemo_fraction_treated"
            ],
            "target": expected[
                "chemo_fraction_treated_anchor"
            ],
            "tolerance": tolerances[
                "chemo_fraction_treated"
            ],
        },
        {
            "metric": "chemo_fraction_control",
            "observed": anchor_row[
                "mean_full_chemo_fraction_control"
            ],
            "target": expected[
                "chemo_fraction_control_anchor"
            ],
            "tolerance": tolerances[
                "chemo_fraction_control"
            ],
        },
        {
            "metric": "strict_population_n",
            "observed": anchor_row[
                "mean_strict_population_n"
            ],
            "target": expected[
                "strict_sequence_eligible_n_anchor"
            ],
            "tolerance": tolerances[
                "strict_population_n"
            ],
        },
        {
            "metric": "strict_treated_n",
            "observed": anchor_row[
                "mean_strict_treated_n"
            ],
            "target": expected["strict_treated_anchor"],
            "tolerance": tolerances["strict_treated_n"],
        },
        {
            "metric": "strict_control_n",
            "observed": anchor_row[
                "mean_strict_control_n"
            ],
            "target": expected["strict_control_anchor"],
            "tolerance": tolerances["strict_control_n"],
        },
        {
            "metric": "strict_treated_events",
            "observed": anchor_row[
                "mean_strict_treated_events"
            ],
            "target": expected[
                "strict_treated_events_anchor"
            ],
            "tolerance": tolerances[
                "strict_treated_events"
            ],
        },
        {
            "metric": "strict_control_events",
            "observed": anchor_row[
                "mean_strict_control_events"
            ],
            "target": expected[
                "strict_control_events_anchor"
            ],
            "tolerance": tolerances[
                "strict_control_events"
            ],
        },
    ])
    anchor_checks["absolute_difference"] = np.abs(
        anchor_checks["observed"] - anchor_checks["target"]
    )
    anchor_checks["pass"] = (
        anchor_checks["absolute_difference"]
        <= anchor_checks["tolerance"]
    )
    write_csv(
        anchor_checks,
        root / output["empirical_anchor_check"],
    )

    gate_rows = []
    for _, row in summary.iterrows():
        method = row["method"]
        bias_limit = (
            gates["maximum_absolute_bias_adjusted_days"]
            if method == "adjusted_full"
            else gates[
                "maximum_absolute_bias_sequencing_aware_days"
            ]
            if method == "sequencing_aware"
            else None
        )
        gate_rows.append({
            "scenario_id": row["scenario_id"],
            "method": method,
            "success_gate": bool(
                row["success_fraction"]
                >= gates["minimum_success_fraction"]
            ),
            "bias_gate_applicable": bias_limit is not None,
            "bias_gate": (
                bool(abs(row["bias_days"]) <= bias_limit)
                if bias_limit is not None
                else None
            ),
            "coverage_gate": bool(
                row["primary_if_coverage"]
                >= gates["minimum_primary_if_coverage"]
            ),
            "included_balance_gate": bool(
                row[
                    "mean_included_covariate_max_abs_weighted_smd"
                ]
                <= gates[
                    "maximum_included_covariate_weighted_smd"
                ]
            ),
        })
    design_gates = pd.DataFrame(gate_rows)
    write_csv(design_gates, root / output["design_gates"])

    adjusted_rows = design_gates[
        design_gates["method"].isin(
            ["adjusted_full", "sequencing_aware"]
        )
    ]
    pilot_ready = bool(
        design_gates["success_gate"].all()
        and design_gates["coverage_gate"].all()
        and design_gates["included_balance_gate"].all()
        and adjusted_rows["bias_gate"].fillna(False).all()
        and anchor_checks["pass"].all()
    )

    final = {
        "status": (
            "STAGE33_SIMULATION_PILOT_V2_READY_FOR_CONFIRMATORY_LOCK"
            if pilot_ready
            else "STAGE33_SIMULATION_PILOT_V2_REQUIRES_REVIEW"
        ),
        "simulation_id": manifest["simulation_id"],
        "pilot_ready_for_confirmatory_lock": pilot_ready,
        "all_empirical_anchor_checks_passed": bool(
            anchor_checks["pass"].all()
        ),
        "maximum_naive_total_bias_days": float(
            naive["bias_days"].abs().max()
        ),
        "maximum_naive_target_drift_days": float(
            naive["mean_target_drift_days"].abs().max()
        ),
        "maximum_naive_residual_omitted_sequence_bias_days": float(
            naive[
                "mean_residual_omitted_sequence_bias_days"
            ].abs().max()
        ),
        "scenario_summary": summary.to_dict("records"),
        "naive_bias_decomposition": naive.to_dict("records"),
        "empirical_anchor_checks": (
            anchor_checks.to_dict("records")
        ),
        "design_gates": design_gates.to_dict("records"),
        "next_action": (
            "Lock a confirmatory simulation with at least 500 "
            "repetitions per scenario."
            if pilot_ready
            else "Review failed gates before any confirmatory run."
        ),
        "boundary": config["boundary"],
    }
    write_json(final, root / output["final_json"])

    figure_dir = root / output["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    for effect_regime in summary["effect_regime"].unique():
        subset = summary[
            summary["effect_regime"] == effect_regime
        ]
        for sample_size in sorted(
            subset["sample_size"].unique()
        ):
            panel = subset[
                subset["sample_size"] == sample_size
            ]
            plt.figure(figsize=(8.2, 5.0))
            for method in [
                "naive_full",
                "adjusted_full",
                "sequencing_aware",
            ]:
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
            plt.ylabel("Mean bias relative to method's primary truth (days)")
            plt.title(
                f"Revised pilot bias: {effect_regime}, n={sample_size}"
            )
            plt.legend()
            plt.tight_layout()
            base = root / output["bias_figure"]
            path = base.with_name(
                base.stem
                + f"_{effect_regime}_n{sample_size}"
                + base.suffix
            )
            plt.savefig(path, dpi=220)
            plt.close()

    plt.figure(figsize=(9.0, 5.2))
    naive_plot = summary[
        summary["method"] == "naive_full"
    ].copy()
    for effect_regime in naive_plot["effect_regime"].unique():
        for sample_size in sorted(
            naive_plot["sample_size"].unique()
        ):
            panel = naive_plot[
                (naive_plot["effect_regime"] == effect_regime)
                & (naive_plot["sample_size"] == sample_size)
            ].sort_values("sequencing_strength")
            plt.plot(
                panel["sequencing_strength"],
                panel["mean_weighted_chemo_smd"],
                marker="o",
                label=f"{effect_regime}, n={sample_size}",
            )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Sequencing strength")
    plt.ylabel("Overlap-weighted SMD for omitted chemotherapy")
    plt.title("Residual sequencing imbalance in the naive full-cohort estimator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        root / output["omitted_balance_figure"],
        dpi=220,
    )
    plt.close()

    null_plot = summary[
        summary["effect_regime"] == "null"
    ].copy()
    plt.figure(figsize=(9.0, 5.2))
    for method in [
        "naive_full",
        "adjusted_full",
        "sequencing_aware",
    ]:
        panel = null_plot[
            (null_plot["method"] == method)
            & (null_plot["sample_size"] == 559)
        ].sort_values("sequencing_strength")
        plt.plot(
            panel["sequencing_strength"],
            panel["mean_estimate_days"],
            marker="o",
            label=method,
        )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Sequencing strength")
    plt.ylabel("Mean estimated RMST contrast under true null (days)")
    plt.title("Null-effect behaviour at the empirical sample-size scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / output["null_figure"], dpi=220)
    plt.close()

    print("Scenario summary")
    print(summary.to_string(index=False))
    print("\nNaive bias decomposition")
    print(naive.to_string(index=False))
    print("\nEmpirical anchor checks")
    print(anchor_checks.to_string(index=False))
    print("\nFinal decision")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 33 revised simulation pilot summarized. "
        "No manuscript prose was generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
