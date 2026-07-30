#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage13_utils import (
    conditional_rmst_from_curve,
    ensure_output_dirs,
    find_ccw_curve_table,
    find_point_rows,
    load_config,
    markdown_table,
    numeric,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_output_dirs(root)
    cfg = load_config(root)
    tables = root / "results" / "tables"
    landmark, ccw, _, _ = find_point_rows(root)

    lm_effect = numeric(landmark.get("estimate_days"))
    ccw_effect = numeric(ccw.get("estimate_days"))
    horizon = float(cfg["primary_design"]["post_landmark_horizon_days"])
    landmark_day = float(cfg["primary_design"]["landmark_day"])
    diagnosis_end = float(cfg["primary_design"]["diagnosis_time_horizon_days"])

    estimands = pd.DataFrame(
        [
            {
                "analysis": "primary_landmark",
                "eligibility_time": "day 180 after diagnosis",
                "time_zero_day": int(landmark_day),
                "analysis_end_day_from_diagnosis": int(landmark_day + horizon),
                "followup_horizon_days": int(horizon),
                "population": "alive and uncensored at day 180; non-ambiguous initiation timing",
                "n": int(round(numeric(landmark.get("n"), cfg["expected_counts"]["landmark_n"]))),
                "strategy_1": "hormone therapy initiated by day 180",
                "strategy_0": "not initiated by day 180",
                "target_population": "ATO overlap population within the landmark-eligible cohort",
                "conditioning": "conditions on survival and observation through day 180",
                "rmst_difference_days": lm_effect,
            },
            {
                "analysis": "diagnosis_time_ccw",
                "eligibility_time": "diagnosis",
                "time_zero_day": 0,
                "analysis_end_day_from_diagnosis": int(diagnosis_end),
                "followup_horizon_days": int(diagnosis_end),
                "population": "verified source cohort excluding ambiguous initiation timing",
                "n": int(round(numeric(ccw.get("ccw_eligible_n"), cfg["expected_counts"]["ccw_eligible_n"]))),
                "strategy_1": "initiate hormone therapy by day 180",
                "strategy_0": "do not initiate hormone therapy by day 180",
                "target_population": "diagnosis-time dynamic-strategy population under clone/adherence weighting",
                "conditioning": "includes events and censoring before day 180",
                "rmst_difference_days": ccw_effect,
            },
        ]
    )

    comparison = pd.DataFrame(
        [
            {"dimension": "treatment window", "same": True, "detail": "Both distinguish initiation by day 180."},
            {"dimension": "time zero", "same": False, "detail": "Landmark begins at day 180; CCW begins at diagnosis."},
            {"dimension": "eligibility population", "same": False, "detail": "Landmark conditions on being alive/observed at day 180."},
            {"dimension": "follow-up scale", "same": False, "detail": "Landmark integrates days 180-910; CCW integrates days 0-910."},
            {"dimension": "target weighting", "same": False, "detail": "Landmark targets an ATO overlap population; CCW uses clone/adherence weights."},
            {"dimension": "direct numerical comparability", "same": False, "detail": "The two point estimates must not be treated as interchangeable estimators of one parameter."},
        ]
    )

    curve_path, cols = find_ccw_curve_table(root)
    curve_status = "CCW_CURVE_OUTPUT_NOT_FOUND"
    conditional_rows = []
    if curve_path is not None:
        curve = read_csv(curve_path)
        for strategy, part in curve.groupby(cols["strategy"]):
            rmst, s180 = conditional_rmst_from_curve(
                pd.to_numeric(part[cols["time"]], errors="coerce").to_numpy(),
                pd.to_numeric(part[cols["survival"]], errors="coerce").to_numpy(),
                landmark_day,
                diagnosis_end,
            )
            conditional_rows.append(
                {
                    "strategy": strategy,
                    "survival_at_day180": s180,
                    "conditional_rmst_day180_to_day910": rmst,
                    "source_curve": str(curve_path.relative_to(root)),
                }
            )
        curve_status = "CCW_CONDITIONAL_RMST_COMPUTED"
        write_csv(pd.DataFrame(conditional_rows), tables / "46_ccw_conditional_rmst.csv")

    sign_status = (
        "OPPOSITE_DIRECTIONS"
        if np.isfinite(lm_effect) and np.isfinite(ccw_effect) and lm_effect * ccw_effect < 0
        else "SAME_DIRECTION_OR_ZERO"
    )
    harmonization_status = (
        "NOT_DIRECTLY_COMPARABLE_REQUIRES_SEPARATE_ESTIMAND_REPORTING"
        if not bool(comparison["same"].all())
        else "DIRECTLY_COMPARABLE"
    )

    write_csv(estimands, tables / "46_estimand_map.csv")
    write_csv(comparison, tables / "46_estimand_comparison.csv")
    summary = pd.DataFrame(
        [
            {
                "landmark_effect_days": lm_effect,
                "ccw_effect_days": ccw_effect,
                "direction_status": sign_status,
                "direct_comparability": harmonization_status,
                "ccw_curve_status": curve_status,
            }
        ]
    )
    write_csv(summary, tables / "46_estimand_harmonization_summary.csv")

    report = f"""# Stage 13 estimand harmonization

## Estimand map

{markdown_table(estimands)}

## Compatibility audit

{markdown_table(comparison)}

## Interpretation

- Direction status: `{sign_status}`.
- Direct-comparability status: `{harmonization_status}`.
- CCW curve export: `{curve_status}`.

The sign disagreement is not, by itself, proof that one estimator is wrong. The analyses start at
different time origins, use different eligibility populations, and target differently weighted
populations. They must be reported as **design-sensitive estimands**, not pooled or described as
two implementations of a single treatment effect.

A weighted CCW survival-curve export is needed to decompose the diagnosis-time result into the
pre-landmark and post-landmark portions. Stage 13 computes a conditional day-180-to-day-910
RMST only when a suitable curve table is already present.
"""
    write_text(report, tables / "46_estimand_harmonization.md")

    print("=" * 112)
    print("STAGE 46 — ESTIMAND AND TARGET-POPULATION AUDIT")
    print("=" * 112)
    print(estimands.to_string(index=False))
    print("\nCompatibility")
    print(comparison.to_string(index=False))
    print(f"\nDirection: {sign_status}")
    print(f"Comparability: {harmonization_status}")
    print(f"CCW curve status: {curve_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
