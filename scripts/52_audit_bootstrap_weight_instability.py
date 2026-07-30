#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage14_utils import (
    checkpoint_table,
    detect_checkpoint_weight_columns,
    ensure_dirs,
    load_config,
    markdown_table,
    project_root,
    select_estimate_column,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    gate = cfg["bootstrap_weight_gate"]
    tables = root / "results/tables"

    path, df = checkpoint_table(root, "ccw")
    estimate_col = select_estimate_column(df)
    weight_cols = detect_checkpoint_weight_columns(df)
    work = pd.DataFrame(
        {
            "estimate_days": pd.to_numeric(df[estimate_col], errors="coerce"),
            "weight_max": pd.to_numeric(df[weight_cols["max"]], errors="coerce")
            if weight_cols["max"]
            else np.nan,
            "weight_p99": pd.to_numeric(df[weight_cols["p99"]], errors="coerce")
            if weight_cols["p99"]
            else np.nan,
        }
    ).dropna(subset=["estimate_days"])

    threshold = float(gate["max_weight_warning"])
    work["weight_max_gt_warning"] = work["weight_max"] > threshold
    finite_weight = work["weight_max"].notna()
    fraction_extreme = float(work.loc[finite_weight, "weight_max_gt_warning"].mean()) if finite_weight.any() else float("nan")
    maximum_weight = float(work["weight_max"].max()) if finite_weight.any() else float("nan")
    correlation = (
        float(work[["estimate_days", "weight_max"]].corr().iloc[0, 1])
        if finite_weight.sum() >= 3
        else float("nan")
    )

    groups = []
    if finite_weight.any():
        for label, mask in (
            (f"weight_max_le_{threshold:g}", work["weight_max"] <= threshold),
            (f"weight_max_gt_{threshold:g}", work["weight_max"] > threshold),
        ):
            values = work.loc[mask, "estimate_days"].dropna()
            groups.append(
                {
                    "group": label,
                    "repetitions": len(values),
                    "mean_estimate_days": values.mean() if len(values) else np.nan,
                    "median_estimate_days": values.median() if len(values) else np.nan,
                    "sd_estimate_days": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "fraction_positive": (values > 0).mean() if len(values) else np.nan,
                }
            )
    grouped = pd.DataFrame(groups)

    warning = (
        (np.isfinite(fraction_extreme) and fraction_extreme > float(gate["fraction_repetitions_warning"]))
        or (np.isfinite(maximum_weight) and maximum_weight > float(gate["maximum_weight_critical"]))
    )
    status = (
        "BOOTSTRAP_WEIGHT_INSTABILITY_REQUIRES_TRUNCATION_SENSITIVITY"
        if warning
        else "BOOTSTRAP_WEIGHT_BEHAVIOR_ACCEPTABLE"
    )
    summary = pd.DataFrame(
        [
            {
                "checkpoint": str(path.relative_to(root)),
                "successful_repetitions": len(work),
                "estimate_column": estimate_col,
                "weight_max_column": weight_cols["max"] or "",
                "weight_p99_column": weight_cols["p99"] or "",
                "median_weight_p99": work["weight_p99"].median(),
                "median_weight_max": work["weight_max"].median(),
                "maximum_weight_across_repetitions": maximum_weight,
                "fraction_repetitions_weight_max_gt10": fraction_extreme,
                "correlation_estimate_vs_weight_max": correlation,
                "weight_stability_status": status,
            }
        ]
    )

    write_csv(work, tables / "52_ccw_bootstrap_estimate_weight_pairs.csv")
    write_csv(grouped, tables / "52_ccw_bootstrap_weight_strata.csv")
    write_csv(summary, tables / "52_ccw_bootstrap_weight_audit.csv")

    # Default matplotlib styling is deliberately retained.
    try:
        import matplotlib.pyplot as plt

        plot = work.dropna(subset=["weight_max"])
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.scatter(plot["weight_max"], plot["estimate_days"], alpha=0.8)
            ax.axvline(threshold, linestyle="--", linewidth=1)
            ax.axhline(0.0, linestyle=":", linewidth=1)
            ax.set_xlabel("Maximum clone weight in bootstrap repetition")
            ax.set_ylabel("CCW RMST difference (days)")
            ax.set_title("CCW bootstrap estimate versus maximum clone weight")
            fig.tight_layout()
            fig.savefig(root / "results/figures/52_ccw_estimate_vs_weight_max.png", dpi=220)
            plt.close(fig)
    except Exception as exc:
        print(f"Plot warning: {type(exc).__name__}: {exc}")

    write_text(
        f"""# Stage 14 bootstrap-weight audit

**Status:** `{status}`

## Summary

{markdown_table(summary)}

## Estimates stratified by maximum weight

{markdown_table(grouped)}

A good full-data 99th-percentile weight does not guarantee stable re-estimated weights in bootstrap
samples. This audit therefore uses repetition-level maximum weights, not only the original-sample
weight summary.
""",
        tables / "52_ccw_bootstrap_weight_audit.md",
    )

    print("=" * 116)
    print("STAGE 52 — CCW BOOTSTRAP WEIGHT INSTABILITY")
    print("=" * 116)
    print(summary.to_string(index=False))
    print("\nWeight strata")
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
