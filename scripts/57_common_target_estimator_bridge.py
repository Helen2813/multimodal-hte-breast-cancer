#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage14_utils import discover_curve_columns, prepare_survival_rows
from _stage15_utils import (
    add_clone_id,
    choose_landmark_candidate,
    conditional_strategy_effect,
    detect_binary_column,
    detect_propensity_column,
    detect_time_column,
    ensure_dirs,
    load_config,
    markdown_table,
    normalize_id_series,
    project_root,
    read_csv,
    weighted_landmark_km,
    write_csv,
    write_text,
)


def normalize_preserving_missing(series: pd.Series) -> pd.Series:
    missing = series.isna()
    out = normalize_id_series(series).mask(missing)
    return out.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NA": np.nan})


def landmark_id_column(df: pd.DataFrame) -> str:
    exact = (
        "patient_id_normalized",
        "patient_id_norm",
        "normalized_patient_id",
        "patient_id",
        "cases.submitter_id",
        "submitter_id",
        "case_id",
    )
    lower = {str(col).lower(): str(col) for col in df.columns}
    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]
    candidates = [
        str(col)
        for col in df.columns
        if any(token in str(col).lower() for token in ("patient_id", "submitter_id", "case_id"))
        and not any(
            token in str(col).lower()
            for token in ("rna", "cnv", "protein", "methyl", "mirna", "mutation")
        )
    ]
    if not candidates:
        raise RuntimeError("No named patient-ID column was found in the landmark candidate.")
    return candidates[0]


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    design = cfg["design"]
    tables = root / "results/tables"

    mapped_path = root / "data/derived/stage15/60_ccw_long_with_patient_id.csv"
    if not mapped_path.exists():
        raise FileNotFoundError(
            f"Exact mapped clone table not found: {mapped_path}. Run Stage 60 first."
        )
    long_df = read_csv(mapped_path)
    if "__stage15_patient_id" not in long_df.columns:
        raise KeyError("__stage15_patient_id is missing from the mapped clone table.")

    cohort_df, cohort_path = choose_landmark_candidate(
        root, int(design["expected_landmark_n"])
    )
    cohort_id = landmark_id_column(cohort_df)
    treatment_col = detect_binary_column(
        cohort_df,
        ("analysis_treatment", "treatment", "treated"),
        ("event", "censor"),
    )
    event_col = detect_binary_column(
        cohort_df,
        ("analysis_event", "event", "death"),
        ("treatment", "censor"),
    )
    time_col = detect_time_column(cohort_df)
    propensity_col = detect_propensity_column(
        cohort_df, int(cfg["bridge"]["minimum_propensity_unique_values"])
    )

    cohort = cohort_df.copy()
    cohort["patient_id_norm"] = normalize_preserving_missing(cohort[cohort_id])
    if cohort["patient_id_norm"].isna().any():
        raise RuntimeError("Landmark patient ID contains missing values.")
    if cohort["patient_id_norm"].duplicated().any():
        raise RuntimeError("Landmark patient ID is not unique.")

    cohort["treatment_stage15"] = pd.to_numeric(
        cohort[treatment_col], errors="coerce"
    )
    cohort["propensity_stage15"] = pd.to_numeric(
        cohort[propensity_col], errors="coerce"
    )
    cohort["ato_weight_stage15"] = np.where(
        cohort["treatment_stage15"] == 1,
        1.0 - cohort["propensity_stage15"],
        cohort["propensity_stage15"],
    )

    detected = discover_curve_columns(long_df)
    rows = prepare_survival_rows(long_df, detected, cap=None)
    rows["patient_id_norm"] = normalize_preserving_missing(
        long_df.loc[rows.index, "__stage15_patient_id"]
    )

    ccw_ids = set(rows["patient_id_norm"].dropna().unique())
    landmark_ids = set(cohort["patient_id_norm"].dropna().unique())
    intersection = len(ccw_ids & landmark_ids)
    landmark_coverage = intersection / len(landmark_ids)
    if landmark_coverage < 0.999:
        raise RuntimeError(
            f"Exact runtime map covers only {landmark_coverage:.2%} of landmark patients "
            f"({intersection}/{len(landmark_ids)})."
        )

    rows = rows.merge(
        cohort[
            ["patient_id_norm", "ato_weight_stage15", "treatment_stage15"]
        ],
        on="patient_id_norm",
        how="inner",
        validate="many_to_one",
    )
    rows = add_clone_id(rows)

    merged_patients = rows["patient_id_norm"].nunique()
    if merged_patients != len(cohort):
        raise RuntimeError(
            f"Expected all {len(cohort)} landmark patients after merge, found {merged_patients}."
        )

    original_summary, original_effect = conditional_strategy_effect(
        rows,
        float(design["diagnosis_time_end_day"]),
        float(design["landmark_day"]),
    )

    ato_rows = rows.copy()
    ato_rows["weight"] = (
        ato_rows["weight"] * ato_rows["ato_weight_stage15"]
    )
    ato_summary, ato_effect = conditional_strategy_effect(
        ato_rows,
        float(design["diagnosis_time_end_day"]),
        float(design["landmark_day"]),
    )

    cohort["unweighted_stage15"] = 1.0
    unweighted_summary, unweighted_effect = weighted_landmark_km(
        cohort,
        "treatment_stage15",
        event_col,
        time_col,
        "unweighted_stage15",
        float(design["post_landmark_horizon_days"]),
    )
    overlap_summary, overlap_effect = weighted_landmark_km(
        cohort,
        "treatment_stage15",
        event_col,
        time_col,
        "ato_weight_stage15",
        float(design["post_landmark_horizon_days"]),
    )
    landmark_aipw = float(design["expected_landmark_aipw_days"])

    effects = pd.DataFrame(
        [
            {
                "analysis": "landmark_unweighted_km",
                "target": "observed landmark cohort",
                "adjustment": "none",
                "rmst_effect_days": unweighted_effect,
            },
            {
                "analysis": "landmark_overlap_weighted_km",
                "target": "landmark ATO",
                "adjustment": "frozen overlap weights",
                "rmst_effect_days": overlap_effect,
            },
            {
                "analysis": "landmark_overlap_aipw",
                "target": "landmark ATO",
                "adjustment": "overlap weights plus outcome augmentation",
                "rmst_effect_days": landmark_aipw,
            },
            {
                "analysis": "ccw_conditional_post180_original",
                "target": "diagnosis-time CCW survivors at day 180",
                "adjustment": "clone/adherence weights",
                "rmst_effect_days": original_effect,
            },
            {
                "analysis": "ccw_conditional_post180_multiplied_by_landmark_ato",
                "target": "landmark ATO bridge",
                "adjustment": "clone/adherence weights x frozen overlap weights",
                "rmst_effect_days": ato_effect,
            },
        ]
    )
    components = pd.DataFrame(
        [
            {
                "contrast": "outcome_augmentation_bridge",
                "difference_days": landmark_aipw - overlap_effect,
            },
            {
                "contrast": "target_weight_bridge_within_ccw",
                "difference_days": ato_effect - original_effect,
            },
            {
                "contrast": "remaining_ccw_vs_landmark_weighted_km",
                "difference_days": overlap_effect - ato_effect,
            },
        ]
    )

    if np.sign(ato_effect) == np.sign(landmark_aipw):
        bridge_status = "ATO_TARGET_WEIGHTING_RECONCILES_DIRECTION"
    elif np.sign(overlap_effect) != np.sign(landmark_aipw):
        bridge_status = "OUTCOME_AUGMENTATION_IS_PRIMARY_SIGN_BRIDGE"
    else:
        bridge_status = (
            "CCW_ADHERENCE_OR_CENSORING_MODEL_REMAINS_PRIMARY_DIFFERENCE"
        )

    diagnostics = pd.DataFrame(
        [
            {
                "mapped_clone_table": str(mapped_path.relative_to(root)),
                "long_candidate_rows": len(long_df),
                "landmark_rows": len(cohort),
                "long_patient_id_column": "__stage15_patient_id",
                "landmark_patient_id_column": cohort_id,
                "ccw_unique_patients": len(ccw_ids),
                "landmark_unique_patients": len(landmark_ids),
                "patient_intersection": intersection,
                "landmark_patient_coverage": landmark_coverage,
                "treatment_column": treatment_col,
                "event_column": event_col,
                "time_column": time_col,
                "propensity_column": propensity_col,
                "merged_long_rows": len(rows),
                "merged_patients": merged_patients,
                "landmark_candidate_path": cohort_path,
                "bridge_status": bridge_status,
            }
        ]
    )

    write_csv(effects, tables / "57_common_target_estimator_bridge.csv")
    write_csv(components, tables / "57_bridge_component_differences.csv")
    write_csv(diagnostics, tables / "57_bridge_diagnostics.csv")
    write_csv(
        original_summary,
        tables / "57_ccw_original_conditional_summary.csv",
    )
    write_csv(
        ato_summary,
        tables / "57_ccw_ato_bridge_conditional_summary.csv",
    )
    write_csv(
        unweighted_summary,
        tables / "57_landmark_unweighted_km_summary.csv",
    )
    write_csv(
        overlap_summary,
        tables / "57_landmark_overlap_km_summary.csv",
    )
    write_text(
        "# Stage 15 common-target estimator bridge\n\n"
        + markdown_table(effects)
        + "\n\n"
        + markdown_table(components)
        + "\n\n"
        + markdown_table(diagnostics)
        + "\n\nThe bridge is diagnostic and is not a new prespecified primary estimator.",
        tables / "57_common_target_estimator_bridge.md",
    )

    print("=" * 116)
    print("STAGE 57 — COMMON-TARGET ESTIMATOR BRIDGE")
    print("=" * 116)
    print(diagnostics.to_string(index=False))
    print("\nEffects")
    print(effects.to_string(index=False))
    print("\nComponent differences")
    print(components.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
