#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from _stage16_utils import (
    ensure_dirs,
    load_config,
    markdown_table,
    project_root,
    read_csv,
    write_csv,
    write_json,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    thresholds = cfg["decision_thresholds"]
    tables = root / "results/tables"

    preflight = read_csv(tables / "61_stage16_preflight_checks.csv")
    exact = read_csv(tables / "62_exact_landmark_aipw_decomposition.csv")
    models = read_csv(tables / "63_outcome_model_robustness.csv")
    stability = read_csv(tables / "64_fold_stability_summary.csv")

    exact_pass = bool(preflight["pass"].all())
    primary = models[models["model"] == "arm_ridge_unbounded"].iloc[0]
    bounded = models[models["model"] == "arm_ridge_bounded"].iloc[0]
    confirmatory_models = models[
        models["model"].isin(
            [
                "arm_ridge_bounded",
                "pooled_interaction_ridge_bounded",
                "arm_hist_gradient_boosting_bounded",
            ]
        )
    ]
    effects = pd.to_numeric(
        confirmatory_models["estimate_days"], errors="coerce"
    )
    sign_stable = bool((effects > 0).all() or (effects < 0).all())
    model_spread = float(effects.max() - effects.min())
    bounded_shift = float(
        bounded["estimate_days"] - primary["estimate_days"]
    )
    loo_spread = float(
        stability.loc[
            stability["model"].isin(confirmatory_models["model"]),
            "loo_effect_spread",
        ].max()
    )
    outside_fraction = float(
        max(
            primary["fraction_mu0_outside_0_horizon"],
            primary["fraction_mu1_outside_0_horizon"],
        )
    )

    if not exact_pass:
        decision = "DEBUG_EXACT_LANDMARK_RECONSTRUCTION"
    elif not sign_stable:
        decision = (
            "OUTCOME_AUGMENTATION_SIGN_UNSTABLE_HOLD_PUBLICATION_BOOTSTRAP"
        )
    elif (
        model_spread > thresholds["model_effect_spread_warning_days"]
        or loo_spread
        > thresholds["leave_one_fold_out_spread_warning_days"]
        or abs(bounded_shift)
        > thresholds["bounded_ridge_shift_warning_days"]
        or outside_fraction
        > thresholds["outside_prediction_fraction_warning"]
    ):
        decision = (
            "POSITIVE_DIRECTION_BUT_OUTCOME_MODEL_DEPENDENT_"
            "HOLD_PUBLICATION_BOOTSTRAP"
        )
    else:
        decision = (
            "OUTCOME_AUGMENTATION_ROBUST_"
            "PROCEED_TO_SHARED_NUISANCE_BOOTSTRAP_PILOT"
        )

    row = pd.DataFrame(
        [
            {
                "stage16_decision": decision,
                "exact_reconstruction_passed": exact_pass,
                "confirmatory_outcome_model_sign_stable": sign_stable,
                "confirmatory_model_effect_spread_days": model_spread,
                "ridge_bounding_shift_days": bounded_shift,
                "maximum_leave_one_fold_out_spread_days": loo_spread,
                "unbounded_ridge_max_outside_fraction": outside_fraction,
                "exact_aipw_days": primary["estimate_days"],
                "exact_direct_ato_ipw_days": primary[
                    "direct_ato_ipw_effect_days"
                ],
                "exact_plugin_component_days": primary[
                    "plugin_component_days"
                ],
                "exact_total_residual_augmentation_days": primary[
                    "total_residual_augmentation_days"
                ],
                "protocol_status": "CANDIDATE_V6_NOT_LOCKED",
                "publication_bootstrap_locked": True,
                "paper_claim": (
                    "reliability across time-zero, target weighting, "
                    "censoring, and outcome nuisance choices"
                ),
            }
        ]
    )
    write_csv(row, tables / "65_stage16_decision.csv")

    report = f"""# Stage 16 decision report

**Decision:** `{decision}`

**Protocol status:** `CANDIDATE_V6_NOT_LOCKED`

## Exact AIPW decomposition

{markdown_table(exact)}

## Fixed outcome-model registry

{markdown_table(models)}

## Fold stability

{markdown_table(stability)}

## Interpretation rules

- The Stage 12 estimator is decomposed on its exact IPCW-RMST pseudo-outcome. This avoids
  attributing the whole difference between AIPW and a separately constructed Kaplan-Meier
  estimator to outcome augmentation.
- Bounded and nonlinear outcome models use the same patients, folds, propensity scores,
  censoring model, pseudo-outcomes, and ATO score.
- No model is selected based on a favorable effect estimate.
- Patient-level influence diagnostics contain no patient identifier and remain local-only.
- The 300/200 publication bootstrap remains locked until the outcome-nuisance gate is resolved.
"""
    write_text(report, tables / "65_stage16_decision.md")

    old_plan = root / "paper_A_treatment_effects/analysis_plan_CANDIDATE_V5.md"
    new_plan = root / "paper_A_treatment_effects/analysis_plan_CANDIDATE_V6.md"
    if old_plan.exists():
        amendment = f"""

## Stage 16 outcome-augmentation amendment

Status: **CANDIDATE_V6_NOT_LOCKED**.

The exact landmark AIPW estimator was decomposed into direct ATO-IPW, plug-in, treated-residual,
and control-residual components using the original Stage 12 implementation. A fixed robustness
registry compared arm-specific means, unbounded and bounded ridge regression, pooled
treatment-interaction ridge regression, and conservative histogram gradient boosting. All variants
used the same folds, propensity scores, censoring model, IPCW-RMST pseudo-outcome, and ATO score.

Stage 16 decision: `{decision}`.

Generated: {datetime.now(timezone.utc).isoformat()}.
"""
        write_text(
            old_plan.read_text(encoding="utf-8", errors="ignore").rstrip()
            + amendment,
            new_plan,
        )

    old_json = root / "paper_A_treatment_effects/primary_estimand_CANDIDATE_V5.json"
    new_json = root / "paper_A_treatment_effects/primary_estimand_CANDIDATE_V6.json"
    if old_json.exists():
        try:
            data = json.loads(old_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["protocol_status"] = "CANDIDATE_V6_NOT_LOCKED"
        data["stage16"] = {
            "decision": decision,
            "fixed_outcome_model_registry": cfg["outcome_models"],
            "publication_bootstrap_locked": True,
        }
        write_json(data, new_json)

    print("=" * 118)
    print("STAGE 65 — STAGE 16 DECISION")
    print("=" * 118)
    print(row.to_string(index=False))
    print(f"\nReport: {tables / '65_stage16_decision.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
