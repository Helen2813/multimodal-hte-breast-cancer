#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage21_utils import (
    dataframe_console,
    empty_or_read,
    ensure_stage21_dirs,
    finite_quantile,
    load_stage21_config,
    markdown_table,
    project_root,
    verify_locked_files,
    write_csv,
    write_text,
)


def interval_summary(values: np.ndarray, theta_hat: float, se_hat: float) -> dict[str, float]:
    q025 = float(np.quantile(values, 0.025))
    q975 = float(np.quantile(values, 0.975))
    return {
        "percentile_ci_low_days": q025,
        "percentile_ci_high_days": q975,
        "basic_ci_low_days": float(2 * theta_hat - q975),
        "basic_ci_high_days": float(2 * theta_hat - q025),
    }


def main() -> int:
    root = project_root()
    ensure_stage21_dirs(root)
    verify_locked_files(root)
    cfg = load_stage21_config(root)
    tables = root / "results/tables"

    reps = empty_or_read(tables / "82_publication_bootstrap_repetitions_checkpoint.csv")
    errors = empty_or_read(tables / "82_publication_bootstrap_errors.csv")
    point = pd.read_csv(tables / "79_candidate_v9_final_point_estimate.csv", low_memory=False)
    if len(point) != 1:
        raise RuntimeError("Expected exactly one locked Candidate V9 point-estimate row.")
    theta_hat = float(point.iloc[0]["estimate_days"])
    se_hat = float(point.iloc[0]["if_se_days"])
    target = int(cfg["full_bootstrap"]["n_repetitions"])

    if reps.empty:
        raise RuntimeError("No successful Stage 21 bootstrap repetitions are available.")
    reps = reps.sort_values("bootstrap_repetition").reset_index(drop=True)
    effects = pd.to_numeric(reps["aggregated_effect_days"], errors="coerce").to_numpy(float)
    ses = pd.to_numeric(reps["aggregated_if_se_days"], errors="coerce").to_numpy(float)
    finite = np.isfinite(effects)
    effects = effects[finite]
    ses = ses[finite]
    successful = len(effects)
    failed = max(target - successful, 0)

    primary = interval_summary(effects, theta_hat, se_hat)
    valid_t = np.isfinite(ses) & (ses > 0)
    tvals = (effects[valid_t] - theta_hat) / ses[valid_t]
    if len(tvals) >= max(30, int(0.8 * successful)):
        t025 = float(np.quantile(tvals, 0.025))
        t975 = float(np.quantile(tvals, 0.975))
        student_low = float(theta_hat - t975 * se_hat)
        student_high = float(theta_hat - t025 * se_hat)
    else:
        t025 = t975 = student_low = student_high = float("nan")

    threshold = float(cfg["decision_thresholds"]["numerical_explosion_abs_effect_days"])
    summary = pd.DataFrame([{
        "target_repetitions": target,
        "successful_repetitions": successful,
        "failed_repetitions": failed,
        "success_rate": successful / target,
        "locked_point_estimate_days": theta_hat,
        "locked_point_if_se_days": se_hat,
        "bootstrap_mean_days": float(np.mean(effects)),
        "bootstrap_median_days": float(np.median(effects)),
        "bootstrap_sd_days": float(np.std(effects, ddof=1)),
        "bootstrap_bias_days": float(np.mean(effects) - theta_hat),
        **primary,
        "studentized_ci_low_days": student_low,
        "studentized_ci_high_days": student_high,
        "studentized_t_q025": t025,
        "studentized_t_q975": t975,
        "fraction_positive": float(np.mean(effects > 0)),
        "minimum_effect_days": float(np.min(effects)),
        "maximum_effect_days": float(np.max(effects)),
        "median_partition_mcse_days": float(pd.to_numeric(reps.loc[finite, "partition_mcse_effect_days"], errors="coerce").median()),
        "p95_partition_mcse_days": finite_quantile(pd.to_numeric(reps.loc[finite, "partition_mcse_effect_days"], errors="coerce"), 0.95),
        "numerical_explosion_fraction": float(np.mean(np.abs(effects) > threshold)),
        "errors_recorded": len(errors),
    }])

    prefix_rows: list[dict] = []
    for prefix in cfg["inference"]["prefixes"]:
        prefix = int(prefix)
        subset = effects[: min(prefix, successful)]
        if len(subset) == 0:
            continue
        ints = interval_summary(subset, theta_hat, se_hat)
        prefix_rows.append({
            "prefix_repetitions": len(subset),
            "mean_effect_days": float(np.mean(subset)),
            "median_effect_days": float(np.median(subset)),
            "sd_effect_days": float(np.std(subset, ddof=1)) if len(subset) > 1 else float("nan"),
            **ints,
            "fraction_positive": float(np.mean(subset > 0)),
        })
    prefixes = pd.DataFrame(prefix_rows).drop_duplicates("prefix_repetitions", keep="last")

    if not prefixes.empty and 200 in set(prefixes["prefix_repetitions"]) and 300 in set(prefixes["prefix_repetitions"]):
        p200 = prefixes[prefixes["prefix_repetitions"] == 200].iloc[0]
        p300 = prefixes[prefixes["prefix_repetitions"] == 300].iloc[0]
        convergence = pd.DataFrame([{
            "absolute_prefix_200_vs_300_mean_shift_days": abs(float(p200["mean_effect_days"]) - float(p300["mean_effect_days"])),
            "absolute_prefix_200_vs_300_lower_endpoint_shift_days": abs(float(p200["percentile_ci_low_days"]) - float(p300["percentile_ci_low_days"])),
            "absolute_prefix_200_vs_300_upper_endpoint_shift_days": abs(float(p200["percentile_ci_high_days"]) - float(p300["percentile_ci_high_days"])),
        }])
    else:
        convergence = pd.DataFrame([{
            "absolute_prefix_200_vs_300_mean_shift_days": float("nan"),
            "absolute_prefix_200_vs_300_lower_endpoint_shift_days": float("nan"),
            "absolute_prefix_200_vs_300_upper_endpoint_shift_days": float("nan"),
        }])

    diagnostics = [
        "unique_original_patient_fraction", "maximum_patient_multiplicity", "treated", "control",
        "events", "treated_events", "control_events", "partition_mcse_effect_days",
        "minimum_G_min_raw", "minimum_propensity_p01", "maximum_propensity_p99",
        "maximum_pseudo_max",
    ]
    corr_rows: list[dict] = []
    work = reps.loc[finite].copy()
    for col in diagnostics:
        if col in work.columns:
            x = pd.to_numeric(work[col], errors="coerce")
            y = pd.to_numeric(work["aggregated_effect_days"], errors="coerce")
            mask = x.notna() & y.notna()
            corr = float(x[mask].corr(y[mask])) if int(mask.sum()) >= 3 else float("nan")
            corr_rows.append({"diagnostic": col, "correlation_with_effect": corr})
    correlations = pd.DataFrame(corr_rows)

    quantiles = pd.DataFrame({
        "quantile": [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975],
        "effect_days": [float(np.quantile(effects, q)) for q in [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]],
    })

    write_csv(summary, tables / "83_publication_bootstrap_summary.csv")
    write_csv(prefixes, tables / "83_publication_bootstrap_prefix_stability.csv")
    write_csv(convergence, tables / "83_publication_bootstrap_convergence.csv")
    write_csv(correlations, tables / "83_publication_bootstrap_diagnostic_correlations.csv")
    write_csv(quantiles, tables / "83_publication_bootstrap_effect_quantiles.csv")

    report = f"""# Candidate V9 publication-bootstrap summary

## Primary summary

{markdown_table(summary)}

## Prefix stability

{markdown_table(prefixes)}

## 200-to-300 convergence

{markdown_table(convergence)}

## Effect quantiles

{markdown_table(quantiles)}

## Composition and nuisance correlations

{markdown_table(correlations)}

The percentile interval is the locked primary interval. Basic and studentized intervals are prespecified sensitivities. Interpretation remains conditional on consistency, conditional exchangeability, positivity, conditional independent censoring, and source validity.
"""
    write_text(report, tables / "83_publication_bootstrap_summary.md")

    print("=" * 128)
    print("STAGE 83 - PUBLICATION BOOTSTRAP SUMMARY")
    print("=" * 128)
    print("Primary summary")
    print(dataframe_console(summary))
    print("\nPrefix stability")
    print(dataframe_console(prefixes))
    print("\n200-to-300 convergence")
    print(dataframe_console(convergence))
    print("\nEffect quantiles")
    print(dataframe_console(quantiles))
    print("\nComposition and nuisance correlations")
    print(dataframe_console(correlations))
    print("\nAll repetition estimates")
    print(dataframe_console(reps[[
        "bootstrap_repetition", "aggregated_effect_days", "aggregated_if_se_days",
        "partition_mcse_effect_days", "events", "treated_events", "control_events",
        "minimum_G_min_raw", "minimum_propensity_p01", "maximum_propensity_p99",
        "maximum_pseudo_max"
    ]], max_rows=300))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
