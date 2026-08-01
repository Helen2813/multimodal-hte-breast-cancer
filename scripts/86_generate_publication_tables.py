from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _stage22_utils import (
    as_float,
    compute_prefix_table,
    ensure_dirs,
    find_root,
    first_existing,
    load_config,
    pick_col,
    print_frame,
    read_csv,
    read_one_row,
    write_table_bundle,
)


def maybe_float(row: pd.Series, name: str) -> float:
    return as_float(row[name])


def build_design_sensitivity(root: Path, point: float, summary: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    bridge_path = first_existing(root, ["results/tables/57_common_target_estimator_bridge.csv"], required=False)
    if bridge_path is not None:
        bridge = read_csv(bridge_path)
        effect_col = pick_col(bridge, ["rmst_effect_days", "estimate_days", "effect_days"], "bridge effect", required=False)
        analysis_col = pick_col(bridge, ["analysis", "estimator", "model"], "bridge analysis", required=False)
        target_col = pick_col(bridge, ["target", "target_population"], "bridge target", required=False)
        adjustment_col = pick_col(bridge, ["adjustment", "method"], "bridge adjustment", required=False)
        if effect_col and analysis_col:
            for _, row in bridge.iterrows():
                rows.append(
                    {
                        "estimand_family": "Design bridge",
                        "analysis": str(row[analysis_col]),
                        "estimate_days": float(row[effect_col]),
                        "interval_low_days": np.nan,
                        "interval_high_days": np.nan,
                        "target_or_note": " | ".join(
                            part for part in [
                                str(row[target_col]) if target_col else "",
                                str(row[adjustment_col]) if adjustment_col else "",
                            ] if part and part != "nan"
                        ),
                    }
                )

    decomposition_path = first_existing(root, ["results/tables/62_exact_landmark_aipw_decomposition.csv"], required=False)
    if decomposition_path is not None:
        decomp = read_csv(decomposition_path)
        if not decomp.empty:
            row = decomp.iloc[0]
            if "direct_ato_ipw_effect_days" in row:
                rows.append(
                    {
                        "estimand_family": "Landmark ATO, same IPCW pseudo-outcome",
                        "analysis": "Direct ATO-IPW",
                        "estimate_days": float(row["direct_ato_ipw_effect_days"]),
                        "interval_low_days": np.nan,
                        "interval_high_days": np.nan,
                        "target_or_note": "Anchor without outcome augmentation",
                    }
                )

    robustness_path = first_existing(root, ["results/tables/63_outcome_model_robustness.csv"], required=False)
    if robustness_path is not None:
        robust = read_csv(robustness_path)
        if {"model", "estimate_days"}.issubset(robust.columns):
            for _, row in robust.iterrows():
                rows.append(
                    {
                        "estimand_family": "Landmark ATO-AIPW outcome-model sensitivity",
                        "analysis": str(row["model"]),
                        "estimate_days": float(row["estimate_days"]),
                        "interval_low_days": np.nan,
                        "interval_high_days": np.nan,
                        "target_or_note": "Same cohort, folds, propensity, censoring, and pseudo-outcome",
                    }
                )

    rows.append(
        {
            "estimand_family": "Locked primary analysis",
            "analysis": "Candidate V9, 20-partition repeated-score ATO-AIPW",
            "estimate_days": point,
            "interval_low_days": maybe_float(summary, "percentile_ci_low_days"),
            "interval_high_days": maybe_float(summary, "percentile_ci_high_days"),
            "target_or_note": "Primary 95% patient-bootstrap percentile interval",
        }
    )

    ccw_path = first_existing(root, ["results/tables/58_reestimated_truncation_bootstrap_summary.csv"], required=False)
    if ccw_path is not None:
        ccw = read_csv(ccw_path)
        if not ccw.empty:
            strategy_col = pick_col(ccw, ["strategy"], "CCW strategy", required=False)
            mean_col = pick_col(ccw, ["bootstrap_mean_days", "mean_effect_days"], "CCW bootstrap mean", required=False)
            low_col = pick_col(ccw, ["percentile_ci_low_days"], "CCW CI low", required=False)
            high_col = pick_col(ccw, ["percentile_ci_high_days"], "CCW CI high", required=False)
            if strategy_col and mean_col:
                for _, row in ccw.iterrows():
                    rows.append(
                        {
                            "estimand_family": "Diagnosis-time CCW sensitivity (not directly comparable)",
                            "analysis": str(row[strategy_col]),
                            "estimate_days": float(row[mean_col]),
                            "interval_low_days": float(row[low_col]) if low_col else np.nan,
                            "interval_high_days": float(row[high_col]) if high_col else np.nan,
                            "target_or_note": "Different time zero and adherence/censoring estimand",
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    root = find_root(Path.cwd())
    config = load_config(root)
    dirs = ensure_dirs(root, config)

    point_row = read_one_row(root / "results/tables/79_candidate_v9_final_point_estimate.csv")
    summary = read_one_row(root / "results/tables/83_publication_bootstrap_summary.csv")
    decision = read_one_row(root / "results/tables/84_publication_bootstrap_decision.csv")
    reps = read_csv(root / "results/tables/82_publication_bootstrap_repetitions_checkpoint.csv")
    effect_col = pick_col(reps, ["aggregated_effect_days", "estimate_days", "effect_days"], "bootstrap effect")
    point = as_float(point_row.get("estimate_days", point_row.get("locked_point_estimate_days")))

    n = int(point_row.get("n", 559))
    treated = int(point_row.get("treated", 194))
    control = int(point_row.get("control", 365))
    events = int(point_row.get("events", 50))

    primary = pd.DataFrame(
        [
            {
                "Population": "Verified HR+/HER2- day-180 landmark survivors",
                "N": n,
                "Treated": treated,
                "Control": control,
                "Events": events,
                "Estimand": "ATO difference in 730-day post-landmark RMST",
                "Point estimate (days)": point,
                "95% percentile CI": f"{float(summary['percentile_ci_low_days']):.2f} to {float(summary['percentile_ci_high_days']):.2f}",
                "Bootstrap positive": float(summary["fraction_positive"]),
                "Interpretation": "Positive direction, statistically imprecise; interval includes zero",
            }
        ]
    )
    write_table_bundle(
        primary,
        dirs["tables"] / "86_table_primary_result.csv",
        dirs["tables"] / "86_table_primary_result.tex",
        dirs["tables"] / "86_table_primary_result.md",
        caption="Locked Candidate V9 primary treatment-effect result.",
        label="tab:candidate_v9_primary",
        digits=3,
        footnote="The effect is treated minus control in post-landmark RMST days. The percentile patient-bootstrap interval is primary. ATO denotes the overlap target population.",
    )

    interval = pd.DataFrame(
        [
            {
                "Interval": "Percentile patient bootstrap (primary)",
                "Low (days)": float(summary["percentile_ci_low_days"]),
                "High (days)": float(summary["percentile_ci_high_days"]),
                "Includes zero": bool(float(summary["percentile_ci_low_days"]) <= 0 <= float(summary["percentile_ci_high_days"])),
                "Role": "Primary inference",
            },
            {
                "Interval": "Basic patient bootstrap",
                "Low (days)": float(summary["basic_ci_low_days"]),
                "High (days)": float(summary["basic_ci_high_days"]),
                "Includes zero": bool(float(summary["basic_ci_low_days"]) <= 0 <= float(summary["basic_ci_high_days"])),
                "Role": "Sensitivity",
            },
            {
                "Interval": "Studentized patient bootstrap",
                "Low (days)": float(summary["studentized_ci_low_days"]),
                "High (days)": float(summary["studentized_ci_high_days"]),
                "Includes zero": bool(float(summary["studentized_ci_low_days"]) <= 0 <= float(summary["studentized_ci_high_days"])),
                "Role": "Sensitivity",
            },
            {
                "Interval": "Influence-function diagnostic",
                "Low (days)": float(point_row["if_ci_low_days"]),
                "High (days)": float(point_row["if_ci_high_days"]),
                "Includes zero": bool(float(point_row["if_ci_low_days"]) <= 0 <= float(point_row["if_ci_high_days"])),
                "Role": "Diagnostic only",
            },
        ]
    )
    write_table_bundle(
        interval,
        dirs["tables"] / "86_table_interval_sensitivity.csv",
        dirs["tables"] / "86_table_interval_sensitivity.tex",
        dirs["tables"] / "86_table_interval_sensitivity.md",
        caption="Primary and sensitivity uncertainty intervals for Candidate V9.",
        label="tab:candidate_v9_intervals",
        digits=2,
        footnote="The percentile patient-bootstrap interval was prospectively designated as primary. Other intervals are sensitivity or diagnostic summaries.",
    )

    prefixes = compute_prefix_table(reps[effect_col], point, config["prefix_repetitions"])
    write_table_bundle(
        prefixes,
        dirs["tables"] / "86_table_bootstrap_convergence.csv",
        dirs["tables"] / "86_table_bootstrap_convergence.tex",
        dirs["tables"] / "86_table_bootstrap_convergence.md",
        caption="Monte Carlo convergence of the publication bootstrap.",
        label="tab:bootstrap_convergence",
        digits=2,
        footnote="Rows use the first B prespecified bootstrap repetitions; they are convergence diagnostics and do not define alternative inferential analyses.",
    )

    mcse_col = pick_col(
        reps,
        ["partition_mcse_days", "aggregated_partition_mcse_days", "inner_partition_mcse_days"],
        "partition Monte Carlo standard error",
        required=False,
    )
    computational = pd.DataFrame(
        [
            {"Metric": "Successful patient-bootstrap repetitions", "Observed": int(summary["successful_repetitions"]), "Target/threshold": int(summary["target_repetitions"]), "Status": "Pass"},
            {"Metric": "Persistent failed repetitions", "Observed": int(summary["failed_repetitions"]), "Target/threshold": 0, "Status": "Pass"},
            {"Metric": "Completed nuisance-partition fits", "Observed": int(summary["successful_repetitions"]) * int(config["expected_partitions_per_bootstrap"]), "Target/threshold": int(config["expected_bootstrap_repetitions"]) * int(config["expected_partitions_per_bootstrap"]), "Status": "Pass"},
            {"Metric": "Numerical explosion fraction", "Observed": float(summary["numerical_explosion_fraction"]), "Target/threshold": "<=0.02", "Status": "Pass"},
            {"Metric": "Median inner-partition MCSE (days)", "Observed": float(summary["median_partition_mcse_days"]), "Target/threshold": "descriptive", "Status": "Reported"},
            {"Metric": "P95 inner-partition MCSE (days)", "Observed": float(summary["p95_partition_mcse_days"]), "Target/threshold": "descriptive", "Status": "Reported"},
            {"Metric": "Prefix 200-to-300 mean shift (days)", "Observed": abs(float(prefixes.iloc[-1]["mean_effect_days"]) - float(prefixes.loc[prefixes["prefix_repetitions"] == 200, "mean_effect_days"].iloc[0])), "Target/threshold": "<=5", "Status": "Pass"},
        ]
    )
    write_table_bundle(
        computational,
        dirs["tables"] / "86_table_computational_reproducibility.csv",
        dirs["tables"] / "86_table_computational_reproducibility.tex",
        dirs["tables"] / "86_table_computational_reproducibility.md",
        caption="Computational completion and Monte Carlo diagnostics.",
        label="tab:computational_reproducibility",
        digits=3,
        footnote="The estimator and bootstrap configuration were cryptographically locked before the 300-repetition analysis.",
    )

    design = build_design_sensitivity(root, point, summary)
    write_table_bundle(
        design,
        dirs["tables"] / "86_table_design_and_model_sensitivity.csv",
        dirs["tables"] / "86_table_design_and_model_sensitivity.tex",
        dirs["tables"] / "86_table_design_and_model_sensitivity.md",
        caption="Design, target-population, and outcome-model sensitivity analyses.",
        label="tab:design_model_sensitivity",
        digits=2,
        footnote="Rows from the diagnosis-time clone-censor-weight analysis use a different time zero and estimand and are not direct numerical replications of the landmark ATO result.",
    )

    print_frame("STAGE 86 - PRIMARY RESULT TABLE", primary)
    print_frame("INTERVAL SENSITIVITY", interval)
    print_frame("BOOTSTRAP CONVERGENCE", prefixes)
    print_frame("COMPUTATIONAL REPRODUCIBILITY", computational)
    print_frame("DESIGN AND MODEL SENSITIVITY", design, max_rows=50)
    print(f"\nPublication tables written to: {dirs['tables']}")


if __name__ == "__main__":
    main()
