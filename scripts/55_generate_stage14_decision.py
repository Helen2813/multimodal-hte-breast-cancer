#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from _stage14_utils import (
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
    tables = root / "results/tables"

    weight = read_csv(tables / "52_ccw_bootstrap_weight_audit.csv").iloc[0]
    decomposition = read_csv(tables / "54_ccw_rmst_decomposition.csv").iloc[0]
    cap = read_csv(tables / "54_ccw_fixed_weight_cap_sensitivity.csv")
    center = read_csv(tables / "47_bootstrap_centering_audit.csv")

    curve_pass = str(decomposition["curve_status"]) == "CCW_CURVE_REPRODUCTION_PASSED"
    weight_status = str(weight["weight_stability_status"])
    weight_unstable = weight_status == "BOOTSTRAP_WEIGHT_INSTABILITY_REQUIRES_TRUNCATION_SENSITIVITY"

    landmark_row = center[center["analysis"].astype(str).str.lower() == "landmark"]
    landmark_z = float(landmark_row.iloc[0]["centering_z"]) if not landmark_row.empty else float("nan")
    landmark_centering = (
        "BORDERLINE_ACCEPTABLE"
        if 2.0 < landmark_z <= float(cfg.get("centering_z_threshold", 2.5))
        else "ACCEPTABLE"
    )

    cap_effects = pd.to_numeric(cap["effect_days"], errors="coerce")
    cap_direction_stable = bool((cap_effects <= 0).all() or (cap_effects >= 0).all())

    if not curve_pass:
        decision = "HOLD_AND_DEBUG_CCW_CURVE_REPRODUCTION"
    elif weight_unstable:
        decision = "RUN_REESTIMATED_CCW_WEIGHT_TRUNCATION_SENSITIVITY_BEFORE_PUBLICATION_BOOTSTRAP"
    else:
        decision = "PROCEED_TO_PUBLICATION_BOOTSTRAP_AS_SEPARATE_DESIGN_SENSITIVITY_ESTIMANDS"

    summary = pd.DataFrame(
        [
            {
                "stage14_decision": decision,
                "curve_reproduction_passed": curve_pass,
                "bootstrap_weight_status": weight_status,
                "fixed_weight_cap_direction_stable": cap_direction_stable,
                "landmark_centering_interpretation": landmark_centering,
                "landmark_centering_z": landmark_z,
                "total_ccw_effect_days": decomposition["total_rmst_effect_day0_to_day910"],
                "pre_landmark_ccw_component_days": decomposition["pre_landmark_rmst_effect_day0_to_day180"],
                "post_landmark_conditional_ccw_effect_days": decomposition[
                    "post_landmark_conditional_effect_given_survival_to_day180"
                ],
                "protocol_status": "CANDIDATE_V4_NOT_LOCKED",
                "paper_claim": "design sensitivity, temporal identification, and reliability",
            }
        ]
    )
    write_csv(summary, tables / "55_stage14_decision.csv")

    report = f"""# Stage 14 decision report

**Decision:** `{decision}`

**Protocol status:** `CANDIDATE_V4_NOT_LOCKED`

## Curve-based decomposition

{markdown_table(pd.DataFrame([decomposition]))}

## Bootstrap weight audit

{markdown_table(pd.DataFrame([weight]))}

## Fixed-weight cap sensitivity

{markdown_table(cap)}

## Interpretation

1. The landmark and diagnosis-time CCW analyses remain separate estimands.
2. Curve decomposition shows whether the diagnosis-time contrast is driven mainly by the
   grace-period interval or by the post-day-180 survival experience.
3. The 30-repetition landmark centering check is accepted by the prespecified numerical gate,
   but its z statistic is `{landmark_z:.3f}` and is therefore interpreted as
   `{landmark_centering}`, not as proof of zero bootstrap bias.
4. Repetition-level clone weights are evaluated separately from the favorable full-data p99.
5. A fixed-weight cap sensitivity is descriptive only. When repetition-level weights are unstable,
   the next analysis must re-estimate the adherence/censoring model inside each capped bootstrap
   repetition rather than merely capping the final full-data weights.
6. No treatment-efficacy claim or post-hoc selection of the positive landmark estimate is permitted.
"""
    write_text(report, tables / "55_stage14_decision.md")

    old_plan = root / "paper_A_treatment_effects/analysis_plan_CANDIDATE_V3.md"
    new_plan = root / "paper_A_treatment_effects/analysis_plan_CANDIDATE_V4.md"
    if old_plan.exists():
        amendment = f"""

## Stage 14 CCW curve and bootstrap-weight amendment

Status: **CANDIDATE_V4_NOT_LOCKED**.

The diagnosis-time clone-censor-weight survival curves were exported and decomposed into
day-0-to-day-180 and day-180-to-day-910 components. This decomposition is descriptive and does not
make the CCW and landmark ATO estimands interchangeable.

Stage 14 decision: `{decision}`.

The 30-repetition landmark bootstrap centering gate passed but is described as
`{landmark_centering}`. Repetition-level maximum clone weights are explicitly reported. When their
instability gate is triggered, a re-estimated weight-truncation sensitivity is required before the
publication bootstrap.

Generated: {datetime.now(timezone.utc).isoformat()}.
"""
        write_text(old_plan.read_text(encoding="utf-8", errors="ignore").rstrip() + amendment, new_plan)

    old_json = root / "paper_A_treatment_effects/primary_estimand_CANDIDATE_V3.json"
    new_json = root / "paper_A_treatment_effects/primary_estimand_CANDIDATE_V4.json"
    if old_json.exists():
        try:
            data = json.loads(old_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["protocol_status"] = "CANDIDATE_V4_NOT_LOCKED"
        data["stage14"] = {
            "decision": decision,
            "curve_reproduction_passed": curve_pass,
            "bootstrap_weight_status": weight_status,
            "landmark_centering_interpretation": landmark_centering,
            "publication_bootstrap_locked": True,
        }
        write_json(data, new_json)

    print("=" * 116)
    print("STAGE 55 — STAGE 14 DECISION")
    print("=" * 116)
    print(summary.to_string(index=False))
    print(f"\nReport: {tables / '55_stage14_decision.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
