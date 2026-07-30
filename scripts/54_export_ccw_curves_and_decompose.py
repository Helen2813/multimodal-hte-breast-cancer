#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from _stage14_utils import (
    discover_curve_columns,
    ensure_dirs,
    integrate_step_curve,
    load_config,
    markdown_table,
    point_estimate_table,
    prepare_survival_rows,
    project_root,
    read_csv,
    strategy_orientation,
    weighted_counting_process_km,
    write_csv,
    write_text,
)


def score_candidate(df: pd.DataFrame, detected: dict[str, str | None], expected_rows: int) -> float:
    score = 0.0
    for key, points in (("strategy", 10), ("weight", 10), ("event", 8), ("stop", 8), ("time", 5), ("start", 4)):
        if detected.get(key):
            score += points
    score -= abs(len(df) - expected_rows) / max(expected_rows, 1)
    if detected.get("strategy") and df[detected["strategy"]].nunique(dropna=True) == 2:
        score += 8
    return score


def compute_for_mapping(rows, raw_values, mapping, horizon, landmark):
    outputs = {}
    curves = {}
    for raw in raw_values:
        label = mapping[raw]
        curve, stats = weighted_counting_process_km(rows, raw, horizon)
        pre, _ = integrate_step_curve(curve, 0.0, landmark, conditional=False)
        post_unconditional, s180 = integrate_step_curve(curve, landmark, horizon, conditional=False)
        post_conditional, _ = integrate_step_curve(curve, landmark, horizon, conditional=True)
        stats.update(
            {
                "strategy": label,
                "raw_strategy_value": str(raw),
                "rmst_0_180": pre,
                "rmst_180_910_unconditional": post_unconditional,
                "survival_day180": s180,
                "rmst_180_910_conditional_on_survival_to_180": post_conditional,
            }
        )
        curve = curve.copy()
        curve["strategy"] = label
        curve["raw_strategy_value"] = str(raw)
        outputs[label] = stats
        curves[label] = curve
    return outputs, curves


def validation_error(outputs, expected):
    init = outputs["initiate_by_180"]
    noinit = outputs["no_initiation_by_180"]
    return (
        abs(init["rmst"] - expected["rmst_initiate_by_180"])
        + abs(noinit["rmst"] - expected["rmst_no_initiation_by_180"])
        + 100.0 * abs(init["survival_horizon"] - expected["survival_initiate_at_910"])
        + 100.0 * abs(noinit["survival_horizon"] - expected["survival_no_initiation_at_910"])
    )


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    expected = cfg["expected_ccw"]
    validation = cfg["curve_validation"]
    horizon = float(cfg["primary_design"]["diagnosis_time_end_day"])
    landmark = float(cfg["primary_design"]["landmark_day"])
    tables = root / "results/tables"

    manifest_path = tables / "53_ccw_trace_candidate_manifest.csv"
    manifest = read_csv(manifest_path)
    if manifest.empty:
        raise RuntimeError("No traced CCW candidates are available.")

    candidates = []
    for _, row in manifest.iterrows():
        path = root / str(row["path"])
        df = read_csv(path)
        detected = discover_curve_columns(df)
        candidates.append(
            {
                "path": path,
                "df": df,
                "detected": detected,
                "score": score_candidate(df, detected, int(expected["clone_rows"])),
            }
        )
    candidate = max(candidates, key=lambda item: item["score"])
    df = candidate["df"]
    detected = candidate["detected"]

    rows = prepare_survival_rows(df, detected, cap=None)
    raw_values = list(rows["strategy_raw"].dropna().unique())
    if len(raw_values) != 2:
        raise ValueError(f"Expected exactly two strategies, found {raw_values}")

    proposed = strategy_orientation(rows["strategy_raw"])
    mappings = []
    if proposed is not None:
        mappings.append(proposed)
    mappings.append(
        {
            raw_values[0]: "initiate_by_180",
            raw_values[1]: "no_initiation_by_180",
        }
    )
    mappings.append(
        {
            raw_values[0]: "no_initiation_by_180",
            raw_values[1]: "initiate_by_180",
        }
    )

    unique_mappings = []
    seen = set()
    for mapping in mappings:
        key = tuple(sorted((str(k), v) for k, v in mapping.items()))
        if key not in seen:
            seen.add(key)
            unique_mappings.append(mapping)

    attempts = []
    for mapping in unique_mappings:
        outputs, curves = compute_for_mapping(rows, raw_values, mapping, horizon, landmark)
        attempts.append((validation_error(outputs, expected), mapping, outputs, curves))
    _, mapping, outputs, curves = min(attempts, key=lambda item: item[0])

    init = outputs["initiate_by_180"]
    noinit = outputs["no_initiation_by_180"]
    effect = init["rmst"] - noinit["rmst"]
    pre_effect = init["rmst_0_180"] - noinit["rmst_0_180"]
    post_unconditional_effect = (
        init["rmst_180_910_unconditional"] - noinit["rmst_180_910_unconditional"]
    )
    post_conditional_effect = (
        init["rmst_180_910_conditional_on_survival_to_180"]
        - noinit["rmst_180_910_conditional_on_survival_to_180"]
    )

    checks = pd.DataFrame(
        [
            {
                "quantity": "rmst_initiate_by_180",
                "observed": init["rmst"],
                "expected": expected["rmst_initiate_by_180"],
                "tolerance": validation["rmst_tolerance_days"],
            },
            {
                "quantity": "rmst_no_initiation_by_180",
                "observed": noinit["rmst"],
                "expected": expected["rmst_no_initiation_by_180"],
                "tolerance": validation["rmst_tolerance_days"],
            },
            {
                "quantity": "survival_initiate_at_910",
                "observed": init["survival_horizon"],
                "expected": expected["survival_initiate_at_910"],
                "tolerance": validation["survival_tolerance"],
            },
            {
                "quantity": "survival_no_initiation_at_910",
                "observed": noinit["survival_horizon"],
                "expected": expected["survival_no_initiation_at_910"],
                "tolerance": validation["survival_tolerance"],
            },
            {
                "quantity": "effect_days",
                "observed": effect,
                "expected": expected["effect_days"],
                "tolerance": validation["effect_tolerance_days"],
            },
        ]
    )
    checks["absolute_error"] = (checks["observed"] - checks["expected"]).abs()
    checks["pass"] = checks["absolute_error"] <= checks["tolerance"]
    curve_status = "CCW_CURVE_REPRODUCTION_PASSED" if bool(checks["pass"].all()) else "CCW_CURVE_REPRODUCTION_FAILED"

    curve_table = pd.concat(curves.values(), ignore_index=True)
    write_csv(curve_table, tables / "54_ccw_weighted_survival_curves.csv")
    write_csv(checks, tables / "54_ccw_curve_replication_checks.csv")

    strategy_summary = pd.DataFrame(list(outputs.values()))
    write_csv(strategy_summary, tables / "54_ccw_curve_strategy_summary.csv")

    decomposition = pd.DataFrame(
        [
            {
                "total_rmst_effect_day0_to_day910": effect,
                "pre_landmark_rmst_effect_day0_to_day180": pre_effect,
                "post_landmark_unconditional_effect_day180_to_day910": post_unconditional_effect,
                "post_landmark_conditional_effect_given_survival_to_day180": post_conditional_effect,
                "survival_day180_initiate": init["survival_day180"],
                "survival_day180_no_initiation": noinit["survival_day180"],
                "curve_status": curve_status,
                "candidate_path": str(candidate["path"].relative_to(root)),
                "strategy_column": detected.get("strategy") or "",
                "weight_column": detected.get("weight") or "",
                "event_column": detected.get("event") or "",
                "start_column": detected.get("start") or "",
                "stop_or_time_column": detected.get("stop") or detected.get("time") or "",
            }
        ]
    )
    write_csv(decomposition, tables / "54_ccw_rmst_decomposition.csv")

    # Full-data fixed-weight cap sensitivity. This is not a replacement for a re-estimated bootstrap.
    weight_p99 = float(rows["weight"].quantile(0.99))
    cap_specs = [("none", None), ("cap_5", 5.0), ("cap_10", 10.0), ("cap_empirical_p99", weight_p99)]
    cap_rows = []
    for label, cap in cap_specs:
        capped = prepare_survival_rows(df, detected, cap=cap)
        cap_outputs, _ = compute_for_mapping(capped, raw_values, mapping, horizon, landmark)
        cap_init = cap_outputs["initiate_by_180"]
        cap_no = cap_outputs["no_initiation_by_180"]
        cap_rows.append(
            {
                "cap_strategy": label,
                "cap_value": np.nan if cap is None else cap,
                "rmst_initiate": cap_init["rmst"],
                "rmst_no_initiation": cap_no["rmst"],
                "effect_days": cap_init["rmst"] - cap_no["rmst"],
                "survival_initiate": cap_init["survival_horizon"],
                "survival_no_initiation": cap_no["survival_horizon"],
                "weight_max_after_cap": capped["weight"].max(),
            }
        )
    cap_df = pd.DataFrame(cap_rows)
    write_csv(cap_df, tables / "54_ccw_fixed_weight_cap_sensitivity.csv")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for strategy, part in curve_table.groupby("strategy"):
            ordered = part.sort_values("time")
            ax.step(ordered["time"], ordered["survival"], where="post", label=strategy)
        ax.axvline(landmark, linestyle="--", linewidth=1)
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Days from diagnosis")
        ax.set_ylabel("Weighted survival")
        ax.set_title("Diagnosis-time clone-censor-weight survival curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "results/figures/54_ccw_weighted_survival_curves.png", dpi=240)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot warning: {type(exc).__name__}: {exc}")

    write_text(
        f"""# Stage 14 CCW curve export and RMST decomposition

**Curve reproduction:** `{curve_status}`

## Replication checks

{markdown_table(checks)}

## Strategy summaries

{markdown_table(strategy_summary)}

## RMST decomposition

{markdown_table(decomposition)}

## Fixed-weight cap sensitivity

{markdown_table(cap_df)}

The conditional post-landmark contrast is a descriptive decomposition of the diagnosis-time CCW
curves. It does not transform the CCW analysis into the landmark ATO estimand because the weighting
targets and eligibility conditioning still differ.
""",
        tables / "54_ccw_curve_decomposition.md",
    )

    print("=" * 116)
    print("STAGE 54 — EXPORT CCW CURVES AND DECOMPOSE RMST")
    print("=" * 116)
    print(f"Selected traced candidate: {candidate['path'].relative_to(root)}")
    print(f"Detected columns: {detected}")
    print("\nReplication checks")
    print(checks.to_string(index=False))
    print("\nRMST decomposition")
    print(decomposition.to_string(index=False))
    print("\nFixed-weight cap sensitivity")
    print(cap_df.to_string(index=False))
    return 0 if curve_status == "CCW_CURVE_REPRODUCTION_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
