#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage18_utils import (
    dataframe_console,
    ensure_stage18_dirs,
    finite_quantile,
    load_stage18_config,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def safe_corr(frame: pd.DataFrame, x: str, y: str) -> float:
    data = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return float("nan")
    return float(data[x].corr(data[y]))


def main() -> int:
    root = project_root()
    ensure_stage18_dirs(root)
    cfg = load_stage18_config(root)
    bcfg = cfg["bootstrap_pilot"]
    tables = root / "results/tables"

    repetitions = read_csv(tables / "72_bootstrap_pilot_repetitions_checkpoint.csv")
    errors_path = tables / "72_bootstrap_pilot_errors.csv"
    errors = read_csv(errors_path) if errors_path.exists() else pd.DataFrame()
    stage17 = read_csv(tables / "70_stage17_decision.csv").iloc[0]

    effect = pd.to_numeric(repetitions["aggregated_effect_days"], errors="coerce")
    repetitions = repetitions.loc[np.isfinite(effect)].copy()
    effect = pd.to_numeric(repetitions["aggregated_effect_days"], errors="raise").to_numpy(float)
    target = int(bcfg["n_repetitions"])
    successful = len(repetitions)
    original = float(stage17["aggregated_30_repeat_effect_days"])

    if successful == 0:
        raise RuntimeError("No successful bootstrap repetitions to summarize.")

    q025, q975 = np.quantile(effect, [0.025, 0.975])
    summary = pd.DataFrame([
        {
            "target_repetitions": target,
            "successful_repetitions": successful,
            "failed_repetitions": int(len(errors)),
            "success_rate": float(successful / target),
            "stage17_original_repeated_crossfit_effect_days": original,
            "bootstrap_mean_days": float(np.mean(effect)),
            "bootstrap_median_days": float(np.median(effect)),
            "bootstrap_sd_days": float(np.std(effect, ddof=1)) if successful > 1 else float("nan"),
            "bootstrap_bias_vs_stage17_days": float(np.mean(effect) - original),
            "pilot_percentile_ci_low_days": float(q025),
            "pilot_percentile_ci_high_days": float(q975),
            "pilot_basic_ci_low_days": float(2 * original - q975),
            "pilot_basic_ci_high_days": float(2 * original - q025),
            "fraction_positive": float(np.mean(effect > 0)),
            "minimum_effect_days": float(np.min(effect)),
            "maximum_effect_days": float(np.max(effect)),
            "median_partition_sd_days": float(repetitions["partition_sd_effect_days"].median()),
            "p95_partition_sd_days": finite_quantile(repetitions["partition_sd_effect_days"], 0.95),
            "median_unique_original_patient_fraction": float(repetitions["unique_original_patient_fraction"].median()),
            "maximum_patient_multiplicity": int(repetitions["maximum_patient_multiplicity"].max()),
            "minimum_control_events": int(repetitions["control_events"].min()),
            "minimum_treated_events": int(repetitions["treated_events"].min()),
            "maximum_absolute_effect_days": float(np.max(np.abs(effect))),
        }
    ])

    prefix_rows = []
    for k in (10, 20, 30):
        if successful < k:
            continue
        vals = effect[:k]
        prefix_rows.append({
            "prefix_repetitions": k,
            "mean_effect_days": float(np.mean(vals)),
            "median_effect_days": float(np.median(vals)),
            "sd_effect_days": float(np.std(vals, ddof=1)),
            "percentile_ci_low_days": float(np.quantile(vals, 0.025)),
            "percentile_ci_high_days": float(np.quantile(vals, 0.975)),
            "fraction_positive": float(np.mean(vals > 0)),
        })
    prefix = pd.DataFrame(prefix_rows)

    quantiles = pd.DataFrame([
        {"quantile": q, "effect_days": float(np.quantile(effect, q))}
        for q in (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
    ])

    correlation_vars = [
        "unique_original_patient_fraction",
        "maximum_patient_multiplicity",
        "treated",
        "control",
        "events",
        "treated_events",
        "control_events",
        "partition_sd_effect_days",
        "minimum_G_min_raw",
        "minimum_propensity_p01",
        "maximum_propensity_p99",
        "maximum_pseudo_max",
    ]
    correlations = pd.DataFrame([
        {
            "diagnostic": name,
            "correlation_with_effect": safe_corr(repetitions, "aggregated_effect_days", name),
        }
        for name in correlation_vars
    ])

    distribution = repetitions[[
        "bootstrap_repetition", "aggregated_effect_days", "aggregated_if_se_days",
        "partition_sd_effect_days", "unique_original_patients", "unique_original_patient_fraction",
        "maximum_patient_multiplicity", "treated", "control", "events", "treated_events",
        "control_events", "minimum_G_min_raw", "minimum_propensity_p01",
        "maximum_propensity_p99", "maximum_pseudo_max"
    ]].sort_values("bootstrap_repetition")

    write_csv(summary, tables / "73_bootstrap_pilot_summary.csv")
    write_csv(prefix, tables / "73_bootstrap_prefix_stability.csv")
    write_csv(quantiles, tables / "73_bootstrap_effect_quantiles.csv")
    write_csv(correlations, tables / "73_bootstrap_composition_correlations.csv")
    write_csv(distribution, tables / "73_bootstrap_pilot_distribution.csv")

    report = f"""# Stage 18 bootstrap pilot summary

The intervals below are pilot intervals based on {successful} repetitions. They are not publication
confidence intervals and must not be reported as final inference.

Stage 17 repeated-cross-fit estimate: {original:.3f} days  
Bootstrap pilot mean: {float(np.mean(effect)):.3f} days  
Bootstrap pilot SD: {float(np.std(effect, ddof=1)):.3f} days  
Pilot percentile interval: [{float(q025):.3f}, {float(q975):.3f}] days  
Fraction positive: {float(np.mean(effect > 0)):.3f}
"""
    write_text(report, tables / "73_bootstrap_pilot_summary.md")

    print("=" * 124)
    print("STAGE 73 - BOOTSTRAP PILOT SUMMARY")
    print("=" * 124)
    print("Pilot summary")
    print(dataframe_console(summary))
    print("\nPrefix stability")
    print(dataframe_console(prefix))
    print("\nEffect quantiles")
    print(dataframe_console(quantiles))
    print("\nComposition and nuisance correlations")
    print(dataframe_console(correlations))
    print("\nAll bootstrap repetition estimates")
    print(dataframe_console(distribution))
    print("\nImportant: the 30-repetition interval is diagnostic, not publication inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
