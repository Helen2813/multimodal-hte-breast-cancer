from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage33_simulation_v2_utils import load_json, write_csv


def bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            safe_json(data),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    root = Path.cwd().resolve()
    config = load_json(root / "stage34_confirmatory_simulation_config.json")
    output = config["outputs"]
    manifest = load_json(root / output["manifest"])
    gates_cfg = config["confirmatory_gates"]

    print("=" * 128)
    print("STAGE 132 - SUMMARIZE INDEPENDENT CONFIRMATORY SIMULATION")
    print("=" * 128)

    checkpoint = pd.read_csv(
        root / output["checkpoint"],
        low_memory=False,
        keep_default_na=False,
    )
    checkpoint["success_bool"] = bool_series(checkpoint["success"])

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
    for keys, group in checkpoint.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        successful = group[group["success_bool"]].copy()
        (
            scenario_id,
            sample_size,
            sequencing_level,
            sequencing_strength,
            effect_regime,
            treatment_effect,
            method,
        ) = keys
        row = {
            "scenario_id": scenario_id,
            "sample_size": int(float(sample_size)),
            "sequencing_level": sequencing_level,
            "sequencing_strength": float(sequencing_strength),
            "effect_regime": effect_regime,
            "treatment_log_hazard_effect": float(treatment_effect),
            "method": method,
            "attempted_repetitions": len(group),
            "successful_repetitions": len(successful),
            "success_fraction": len(successful) / len(group),
        }

        for column in [
            "primary_truth_days",
            "estimate_days",
            "primary_bias_days",
            "primary_squared_error",
            "diagnostic_if_se_days",
            "primary_ci_low_days",
            "primary_ci_high_days",
            "included_covariate_max_abs_weighted_smd",
            "unweighted_chemo_smd",
            "weighted_chemo_smd",
            "ato_ess_fraction_treated",
            "ato_ess_fraction_control",
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
        ]:
            successful[column] = pd.to_numeric(
                successful[column], errors="coerce"
            )

        covered = bool_series(successful["primary_covered"])
        positive_exclusion = successful["primary_ci_low_days"] > 0
        negative_exclusion = successful["primary_ci_high_days"] < 0
        repetitions = len(successful)

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
            "bias_mcse_days": float(
                successful["primary_bias_days"].std(ddof=1)
                / math.sqrt(repetitions)
            ),
            "empirical_sd_days": float(
                successful["estimate_days"].std(ddof=1)
            ),
            "rmse_days": float(
                math.sqrt(successful["primary_squared_error"].mean())
            ),
            "mean_if_se_days": float(
                successful["diagnostic_if_se_days"].mean()
            ),
            "primary_if_coverage": float(covered.mean()),
            "coverage_mcse": float(
                math.sqrt(
                    covered.mean()
                    * (1.0 - covered.mean())
                    / repetitions
                )
            ),
            "positive_ci_exclusion_rate": float(
                positive_exclusion.mean()
            ),
            "positive_exclusion_mcse": float(
                math.sqrt(
                    positive_exclusion.mean()
                    * (1.0 - positive_exclusion.mean())
                    / repetitions
                )
            ),
            "negative_ci_exclusion_rate": float(
                negative_exclusion.mean()
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
                successful["ato_ess_fraction_treated"].mean()
            ),
            "mean_ato_ess_fraction_control": float(
                successful["ato_ess_fraction_control"].mean()
            ),
            "mean_full_treated_fraction": float(
                successful["full_treated_fraction"].mean()
            ),
            "mean_full_chemo_fraction_treated": float(
                successful["full_chemo_fraction_treated"].mean()
            ),
            "mean_full_chemo_fraction_control": float(
                successful["full_chemo_fraction_control"].mean()
            ),
            "mean_strict_population_n": float(
                successful["strict_population_n"].mean()
            ),
            "mean_strict_treated_n": float(
                successful["strict_treated_n"].mean()
            ),
            "mean_strict_control_n": float(
                successful["strict_control_n"].mean()
            ),
            "mean_strict_treated_events": float(
                successful["strict_treated_events"].mean()
            ),
            "mean_strict_control_events": float(
                successful["strict_control_events"].mean()
            ),
            "mean_secondary_truth_days": float(
                successful["secondary_truth_days"].mean()
            )
            if successful["secondary_truth_days"].notna().any()
            else np.nan,
            "mean_secondary_bias_days": float(
                successful["secondary_bias_days"].mean()
            )
            if successful["secondary_bias_days"].notna().any()
            else np.nan,
            "mean_target_drift_days": float(
                successful["target_drift_days"].mean()
            )
            if successful["target_drift_days"].notna().any()
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
    write_csv(summary, root / output["summary"])

    naive = summary[summary["method"] == "naive_full"].copy()
    write_csv(
        naive[
            [
                "scenario_id",
                "sample_size",
                "sequencing_level",
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
            ]
        ],
        root / output["naive_decomposition"],
    )

    anchor = summary[
        (summary["sample_size"] == 559)
        & (summary["sequencing_level"] == "empirical")
        & (summary["effect_regime"] == "observed_risk_benefit")
        & (summary["method"] == "sequencing_aware")
    ].iloc[0]
    tol = gates_cfg["empirical_anchor_tolerances"]
    target = {
        "full_treated_fraction": 194 / 559,
        "chemo_fraction_treated": 51 / 194,
        "chemo_fraction_control": 221 / 365,
        "strict_population_n": 271.0,
        "strict_treated_n": 138.0,
        "strict_control_n": 133.0,
        "strict_treated_events": 9.0,
        "strict_control_events": 27.0,
    }
    observed = {
        "full_treated_fraction": anchor[
            "mean_full_treated_fraction"
        ],
        "chemo_fraction_treated": anchor[
            "mean_full_chemo_fraction_treated"
        ],
        "chemo_fraction_control": anchor[
            "mean_full_chemo_fraction_control"
        ],
        "strict_population_n": anchor["mean_strict_population_n"],
        "strict_treated_n": anchor["mean_strict_treated_n"],
        "strict_control_n": anchor["mean_strict_control_n"],
        "strict_treated_events": anchor[
            "mean_strict_treated_events"
        ],
        "strict_control_events": anchor[
            "mean_strict_control_events"
        ],
    }
    anchor_rows = []
    for metric, target_value in target.items():
        difference = abs(float(observed[metric]) - target_value)
        anchor_rows.append({
            "metric": metric,
            "observed": float(observed[metric]),
            "target": target_value,
            "tolerance": float(tol[metric]),
            "absolute_difference": difference,
            "pass": difference <= float(tol[metric]),
        })
    anchor_checks = pd.DataFrame(anchor_rows)
    write_csv(anchor_checks, root / output["anchor_checks"])

    gate_rows = []
    for _, row in summary.iterrows():
        valid = row["method"] in gates_cfg["valid_methods"]
        gate_rows.append({
            "scenario_id": row["scenario_id"],
            "method": row["method"],
            "success_gate": (
                row["success_fraction"]
                >= gates_cfg["minimum_success_fraction"]
            ),
            "valid_method": valid,
            "bias_gate": (
                abs(row["bias_days"])
                <= gates_cfg[
                    "maximum_absolute_bias_valid_methods_days"
                ]
            )
            if valid
            else None,
            "coverage_gate": (
                row["primary_if_coverage"]
                >= gates_cfg[
                    "minimum_if_coverage_valid_methods"
                ]
            )
            if valid
            else None,
            "balance_gate": (
                row[
                    "mean_included_covariate_max_abs_weighted_smd"
                ]
                <= gates_cfg[
                    "maximum_included_covariate_weighted_smd"
                ]
            ),
        })
    gates = pd.DataFrame(gate_rows)
    write_csv(gates, root / output["gates"])

    valid_rows = gates[gates["valid_method"]]
    valid_pass = bool(
        valid_rows["success_gate"].all()
        and valid_rows["bias_gate"].fillna(False).all()
        and valid_rows["coverage_gate"].fillna(False).all()
        and valid_rows["balance_gate"].all()
    )

    naive_no_null = naive[
        (naive["sequencing_level"] == "none")
        & (naive["effect_regime"] == "true_zero")
    ]
    naive_emp_null = naive[
        (naive["sequencing_level"] == "empirical")
        & (naive["effect_regime"] == "true_zero")
    ]
    naive_emp_benefit = naive[
        (naive["sequencing_level"] == "empirical")
        & (naive["effect_regime"] == "observed_risk_benefit")
    ]
    mechanism_checks = {
        "naive_no_sequence_true_zero_near_zero": bool(
            naive_no_null["mean_estimate_days"].abs().max()
            <= gates_cfg[
                "naive_no_sequence_true_zero_max_abs_estimate_days"
            ]
        ),
        "naive_empirical_true_zero_spurious_positive": bool(
            naive_emp_null["mean_estimate_days"].min()
            >= gates_cfg[
                "naive_empirical_true_zero_min_positive_estimate_days"
            ]
        ),
        "naive_empirical_residual_chemo_imbalance": bool(
            naive[
                naive["sequencing_level"] == "empirical"
            ]["mean_weighted_chemo_smd"].abs().min()
            >= gates_cfg[
                "naive_empirical_min_abs_weighted_chemo_smd"
            ]
        ),
        "naive_empirical_benefit_positive_bias": bool(
            naive_emp_benefit["bias_days"].min()
            >= gates_cfg[
                "naive_empirical_benefit_min_positive_bias_days"
            ]
        ),
    }
    mechanism_pass = bool(all(mechanism_checks.values()))
    anchors_pass = bool(anchor_checks["pass"].all())
    final_pass = bool(valid_pass and mechanism_pass and anchors_pass)

    final = {
        "status": (
            "STAGE34_CONFIRMATORY_SIMULATION_COMPLETE"
            if final_pass
            else "STAGE34_CONFIRMATORY_SIMULATION_REQUIRES_REVIEW"
        ),
        "simulation_id": manifest["simulation_id"],
        "confirmatory_result_passed_locked_checks": final_pass,
        "valid_method_checks_passed": valid_pass,
        "naive_mechanism_checks_passed": mechanism_pass,
        "empirical_anchor_checks_passed": anchors_pass,
        "naive_mechanism_checks": mechanism_checks,
        "scenario_summary": summary.to_dict("records"),
        "anchor_checks": anchor_checks.to_dict("records"),
        "gates": gates.to_dict("records"),
        "boundary": config["boundary"],
    }
    write_json(final, root / output["final_json"])

    figure_dir = root / output["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    for n in sorted(summary["sample_size"].unique()):
        panel = summary[
            (summary["sample_size"] == n)
            & (summary["effect_regime"] == "true_zero")
        ]
        plt.figure(figsize=(8.2, 5.0))
        for method in config["methods"].keys():
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
        plt.ylabel("Mean RMST contrast under true zero (days)")
        plt.title(f"Confirmatory true-zero scenarios, n={n}")
        plt.legend()
        plt.tight_layout()
        base = root / output["null_figure"]
        plt.savefig(
            base.with_name(base.stem + f"_n{int(n)}" + base.suffix),
            dpi=220,
        )
        plt.close()

    for effect in summary["effect_regime"].unique():
        for n in sorted(summary["sample_size"].unique()):
            panel = summary[
                (summary["sample_size"] == n)
                & (summary["effect_regime"] == effect)
            ]
            plt.figure(figsize=(8.2, 5.0))
            for method in config["methods"].keys():
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
            plt.title(f"Confirmatory bias: {effect}, n={n}")
            plt.legend()
            plt.tight_layout()
            base = root / output["bias_figure"]
            plt.savefig(
                base.with_name(
                    base.stem + f"_{effect}_n{int(n)}" + base.suffix
                ),
                dpi=220,
            )
            plt.close()

    plt.figure(figsize=(9.0, 5.2))
    for effect in naive["effect_regime"].unique():
        for n in sorted(naive["sample_size"].unique()):
            panel = naive[
                (naive["sample_size"] == n)
                & (naive["effect_regime"] == effect)
            ].sort_values("sequencing_strength")
            plt.plot(
                panel["sequencing_strength"],
                panel["mean_weighted_chemo_smd"],
                marker="o",
                label=f"{effect}, n={n}",
            )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Sequencing strength")
    plt.ylabel("Overlap-weighted SMD for omitted chemotherapy")
    plt.title("Confirmatory omitted-sequencing imbalance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / output["smd_figure"], dpi=220)
    plt.close()

    print("Confirmatory scenario summary")
    print(summary.to_string(index=False))
    print("\nEmpirical anchor checks")
    print(anchor_checks.to_string(index=False))
    print("\nFinal decision")
    print(json.dumps(safe_json(final), indent=2))
    print(
        "\nPASS: Stage 34 confirmatory simulation summarized. "
        "No manuscript prose was generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
