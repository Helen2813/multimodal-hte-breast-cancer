from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage22_utils import (
    as_float,
    compute_prefix_table,
    ensure_dirs,
    find_root,
    load_config,
    pick_col,
    print_frame,
    read_csv,
    read_one_row,
)


def save_both(fig: plt.Figure, base: Path, dpi: int) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_root(Path.cwd())
    config = load_config(root)
    dirs = ensure_dirs(root, config)
    dpi = int(config.get("figure_dpi", 300))

    reps = read_csv(root / "results/tables/82_publication_bootstrap_repetitions_checkpoint.csv")
    summary = read_one_row(root / "results/tables/83_publication_bootstrap_summary.csv")
    point_row = read_one_row(root / "results/tables/79_candidate_v9_final_point_estimate.csv")
    point = as_float(point_row.get("estimate_days", point_row.get("locked_point_estimate_days")))
    effect_col = pick_col(reps, ["aggregated_effect_days", "estimate_days", "effect_days"], "bootstrap effect")
    rep_col = pick_col(reps, ["bootstrap_repetition", "repetition", "rep"], "bootstrap repetition")
    effects = pd.to_numeric(reps[effect_col], errors="coerce").dropna().astype(float)
    low = float(summary["percentile_ci_low_days"])
    high = float(summary["percentile_ci_high_days"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(effects, bins="auto", edgecolor="black", linewidth=0.6)
    ax.axvline(0, linestyle=":", linewidth=1.5, label="Null (0 days)")
    ax.axvline(point, linestyle="-", linewidth=1.8, label=f"Locked point estimate ({point:.1f})")
    ax.axvline(low, linestyle="--", linewidth=1.4, label=f"Percentile 95% CI ({low:.1f}, {high:.1f})")
    ax.axvline(high, linestyle="--", linewidth=1.4)
    ax.set_xlabel("Bootstrap ATO RMST contrast (days)")
    ax.set_ylabel("Bootstrap repetitions")
    ax.set_title("Candidate V9 patient-bootstrap distribution")
    ax.legend(frameon=False)
    save_both(fig, dirs["figures"] / "87_bootstrap_distribution", dpi)

    sorted_effects = np.sort(effects.to_numpy())
    ecdf = np.arange(1, len(sorted_effects) + 1) / len(sorted_effects)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.step(sorted_effects, ecdf, where="post")
    ax.axvline(0, linestyle=":", linewidth=1.5)
    ax.axvline(point, linestyle="-", linewidth=1.5)
    ax.axhline(1 - float(summary["fraction_positive"]), linestyle="--", linewidth=1.0)
    ax.set_xlabel("Bootstrap ATO RMST contrast (days)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Empirical distribution of the locked estimator")
    save_both(fig, dirs["figures"] / "87_bootstrap_ecdf", dpi)

    prefixes = compute_prefix_table(reps[effect_col], point, config["prefix_repetitions"])
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = prefixes["prefix_repetitions"].to_numpy(dtype=float)
    mean = prefixes["mean_effect_days"].to_numpy(dtype=float)
    lo = prefixes["percentile_ci_low_days"].to_numpy(dtype=float)
    hi = prefixes["percentile_ci_high_days"].to_numpy(dtype=float)
    ax.plot(x, mean, marker="o", label="Bootstrap mean")
    ax.plot(x, lo, marker="o", linestyle="--", label="Percentile lower endpoint")
    ax.plot(x, hi, marker="o", linestyle="--", label="Percentile upper endpoint")
    ax.axhline(point, linestyle=":", linewidth=1.5, label="Locked point estimate")
    ax.axhline(0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Bootstrap repetitions included")
    ax.set_ylabel("RMST contrast (days)")
    ax.set_title("Publication-bootstrap prefix convergence")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    save_both(fig, dirs["figures"] / "87_bootstrap_prefix_convergence", dpi)

    mcse_col = pick_col(
        reps,
        ["partition_mcse_days", "aggregated_partition_mcse_days", "inner_partition_mcse_days"],
        "partition MCSE",
        required=False,
    )
    if mcse_col is not None:
        mcse = pd.to_numeric(reps[mcse_col], errors="coerce").dropna().astype(float)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.hist(mcse, bins="auto", edgecolor="black", linewidth=0.6)
        ax.axvline(float(summary["median_partition_mcse_days"]), linestyle="-", linewidth=1.5, label="Median")
        ax.axvline(float(summary["p95_partition_mcse_days"]), linestyle="--", linewidth=1.5, label="95th percentile")
        ax.set_xlabel("Inner repeated-cross-fit Monte Carlo SE (days)")
        ax.set_ylabel("Bootstrap repetitions")
        ax.set_title("Residual nuisance-partition Monte Carlo uncertainty")
        ax.legend(frameon=False)
        save_both(fig, dirs["figures"] / "87_inner_partition_mcse", dpi)

    design_path = dirs["tables"] / "86_table_design_and_model_sensitivity.csv"
    if design_path.exists():
        design = read_csv(design_path)
        plot_df = design.loc[
            design["estimand_family"].isin(
                [
                    "Landmark ATO, same IPCW pseudo-outcome",
                    "Landmark ATO-AIPW outcome-model sensitivity",
                    "Locked primary analysis",
                ]
            )
        ].copy()
        if not plot_df.empty:
            plot_df = plot_df.reset_index(drop=True)
            y = np.arange(len(plot_df))
            fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.55 * len(plot_df) + 1.5)))
            ax.scatter(plot_df["estimate_days"], y, zorder=3)
            primary_mask = plot_df["analysis"].str.contains("Candidate V9", regex=False)
            for idx in np.where(primary_mask.to_numpy())[0]:
                lo_val = plot_df.loc[idx, "interval_low_days"]
                hi_val = plot_df.loc[idx, "interval_high_days"]
                if pd.notna(lo_val) and pd.notna(hi_val):
                    ax.hlines(y[idx], lo_val, hi_val, linewidth=2)
            ax.axvline(0, linestyle=":", linewidth=1.2)
            ax.set_yticks(y)
            ax.set_yticklabels(plot_df["analysis"])
            ax.invert_yaxis()
            ax.set_xlabel("Estimated 730-day post-landmark RMST contrast (days)")
            ax.set_title("Landmark ATO estimator and outcome-model sensitivity")
            save_both(fig, dirs["figures"] / "87_landmark_sensitivity_forest", dpi)

    generated = sorted(p.name for p in dirs["figures"].glob("87_*"))
    report = pd.DataFrame({"generated_figure": generated})
    print_frame("STAGE 87 - GENERATED PUBLICATION FIGURES", report, max_rows=50)
    print(f"\nFigures written to: {dirs['figures']}")


if __name__ == "__main__":
    main()
