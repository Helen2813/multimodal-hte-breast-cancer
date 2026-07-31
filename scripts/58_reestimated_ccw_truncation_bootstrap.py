#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage15_utils import (
    ensure_dirs,
    load_config,
    project_root,
    read_csv,
    run_stage43_with_cap,
    select_estimate_column,
    write_csv,
)


def find_checkpoint(root, strategy_name):
    candidates = sorted((root / "results/tables").glob(f"58_ccw_{strategy_name}*CHECKPOINT.csv"))
    if not candidates:
        raise FileNotFoundError(f"No generated checkpoint for {strategy_name}")
    return candidates[0]


def replicate_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        name = str(col).lower()
        if not any(token in name for token in ("rep", "iteration", "bootstrap_id")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == len(df) and values.nunique() == len(df):
            return str(col)
    return None


def checkpoint_summary(df: pd.DataFrame, name: str) -> dict:
    estimate_col = select_estimate_column(df)
    values = pd.to_numeric(df[estimate_col], errors="coerce").dropna()
    return {
        "strategy": name,
        "successful_reps": len(values),
        "bootstrap_mean_days": values.mean(),
        "bootstrap_median_days": values.median(),
        "bootstrap_sd_days": values.std(ddof=1),
        "percentile_ci_low_days": values.quantile(0.025),
        "percentile_ci_high_days": values.quantile(0.975),
        "fraction_positive": (values > 0).mean(),
    }


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    boot = cfg["truncation_bootstrap"]
    target = int(boot["target_reps"])
    tables = root / "results/tables"

    for spec in boot["strategies"]:
        print("-" * 116)
        print(f"Running re-estimated CCW bootstrap: {spec['name']}, target={target}")
        run_stage43_with_cap(root, spec, target)

    original = read_csv(tables / "43_ccw_bootstrap_CHECKPOINT.csv")
    datasets = {"original_uncapped": original}
    for spec in boot["strategies"]:
        datasets[spec["name"]] = read_csv(find_checkpoint(root, spec["name"]))

    summaries = pd.DataFrame([checkpoint_summary(df, name) for name, df in datasets.items()])
    write_csv(summaries, tables / "58_reestimated_truncation_bootstrap_summary.csv")

    original_est_col = select_estimate_column(original)
    original_rep = replicate_column(original)
    original_values = pd.to_numeric(original[original_est_col], errors="coerce").reset_index(drop=True)
    paired_rows = []
    for name, df in datasets.items():
        if name == "original_uncapped":
            continue
        est_col = select_estimate_column(df)
        rep_col = replicate_column(df)
        if original_rep and rep_col:
            left = original[[original_rep, original_est_col]].rename(columns={original_rep: "replicate", original_est_col: "original_estimate"})
            right = df[[rep_col, est_col]].rename(columns={rep_col: "replicate", est_col: "truncated_estimate"})
            pair = left.merge(right, on="replicate", how="inner")
        else:
            n = min(len(original), len(df))
            pair = pd.DataFrame({
                "replicate": np.arange(1, n + 1),
                "original_estimate": original_values.iloc[:n].to_numpy(),
                "truncated_estimate": pd.to_numeric(df[est_col], errors="coerce").iloc[:n].to_numpy(),
            })
        pair["strategy"] = name
        pair["difference_days"] = pair["truncated_estimate"] - pair["original_estimate"]
        pair["same_sign"] = np.sign(pair["truncated_estimate"]) == np.sign(pair["original_estimate"])
        paired_rows.append(pair)

    paired = pd.concat(paired_rows, ignore_index=True)
    write_csv(paired, tables / "58_reestimated_truncation_paired_repetitions.csv")
    paired_summary = (
        paired.groupby("strategy", as_index=False)
        .agg(
            paired_repetitions=("replicate", "count"),
            mean_shift_days=("difference_days", "mean"),
            median_shift_days=("difference_days", "median"),
            sd_shift_days=("difference_days", "std"),
            sign_agreement=("same_sign", "mean"),
        )
    )
    paired_summary["truncation_status"] = np.where(
        (paired_summary["sign_agreement"] >= float(boot["direction_stability_threshold"]))
        & (paired_summary["mean_shift_days"].abs() <= float(boot["mean_shift_warning_days"])),
        "DIRECTION_ROBUST_TO_REESTIMATED_TRUNCATION",
        "TRUNCATION_SENSITIVE",
    )
    write_csv(paired_summary, tables / "58_reestimated_truncation_paired_summary.csv")

    print("=" * 116)
    print("STAGE 58 — RE-ESTIMATED CCW TRUNCATION BOOTSTRAP")
    print("=" * 116)
    print(summaries.to_string(index=False))
    print("\nPaired comparisons")
    print(paired_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
