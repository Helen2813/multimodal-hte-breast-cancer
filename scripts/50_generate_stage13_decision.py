#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _stage13_utils import (
    ensure_output_dirs,
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
    ensure_output_dirs(root)
    cfg = load_config(root)
    tables = root / "results" / "tables"

    centering = read_csv(tables / "47_bootstrap_centering_audit.csv")
    invariants = read_csv(tables / "49_ccw_invariant_checks.csv")
    harmonization = read_csv(tables / "46_estimand_harmonization_summary.csv").iloc[0]

    enough = bool((centering["successful_reps"] >= cfg["minimum_reps_for_centering_gate"]).all())
    centered = bool((centering["centering_status"] == "BOOTSTRAP_CENTERING_ACCEPTABLE").all())
    ccw_consistent = bool(invariants["pass"].all())
    direct = str(harmonization["direct_comparability"])
    curve_status = str(harmonization["ccw_curve_status"])

    if not ccw_consistent:
        decision = "HOLD_FULL_BOOTSTRAP_AND_DEBUG_CCW"
    elif not enough:
        decision = "HOLD_FULL_BOOTSTRAP_PENDING_CENTERING_PILOT"
    elif not centered:
        decision = "HOLD_FULL_BOOTSTRAP_DUE_TO_CENTERING_CONCERN"
    elif curve_status != "CCW_CONDITIONAL_RMST_COMPUTED":
        decision = "CENTERING_PASSED_EXPORT_CCW_CURVES_BEFORE_FULL_BOOTSTRAP"
    else:
        decision = "PROCEED_FULL_BOOTSTRAP_AS_DESIGN_SENSITIVITY_STUDY"

    rows = pd.DataFrame(
        [
            {
                "stage13_decision": decision,
                "minimum_reps_reached": enough,
                "bootstrap_centering_acceptable": centered,
                "ccw_internal_consistency_passed": ccw_consistent,
                "direct_estimand_comparability": direct,
                "ccw_curve_status": curve_status,
                "paper_claim": "design sensitivity and reliability; not treatment efficacy",
                "protocol_status": "CANDIDATE_V3_NOT_LOCKED",
            }
        ]
    )
    write_csv(rows, tables / "50_stage13_decision.csv")

    report = f"""# Stage 13 decision report

**Protocol status:** `CANDIDATE_V3_NOT_LOCKED`

**Decision:** `{decision}`

## What Stage 12 established

The primary landmark estimator reproduced exactly, and the short landmark bootstrap remained
positive. The diagnosis-time clone-censor-weight estimate had the opposite point-estimate direction.
Stage 13 therefore treats this as a **design-sensitivity problem**, not as a contest in which the
more favorable estimator is selected.

## Bootstrap-centering audit

{markdown_table(centering)}

## CCW internal-consistency audit

{markdown_table(invariants)}

## Estimand interpretation

- Direct comparability: `{direct}`.
- CCW conditional-curve status: `{curve_status}`.
- The landmark effect and diagnosis-time CCW effect must remain separate estimands.
- No pooled estimate, efficacy claim, or post-hoc choice of the positive result is permitted.
- The strongest defensible Paper A framing is reliability of early-treatment effect conclusions
  under alternative time-zero and strategy-emulation designs.

## Next gate

1. Obtain at least {cfg["minimum_reps_for_centering_gate"]} successful repetitions for both
   checkpointed pilots and verify bootstrap centering.
2. Export weighted CCW survival curves so that the diagnosis-time result can be decomposed into
   pre-landmark and conditional post-landmark components.
3. Only after those checks, expand to the planned publication-grade bootstrap.
"""
    write_text(report, tables / "50_stage13_decision.md")

    plan_v2 = root / "paper_A_treatment_effects" / "analysis_plan_CANDIDATE_V2.md"
    plan_v3 = root / "paper_A_treatment_effects" / "analysis_plan_CANDIDATE_V3.md"
    if plan_v2.exists():
        original = plan_v2.read_text(encoding="utf-8", errors="ignore").rstrip()
        appendix = f"""

## Stage 13 estimand-harmonization amendment

Status: **CANDIDATE_V3_NOT_LOCKED**.

The 180-day landmark and diagnosis-time clone-censor-weight analyses are retained as separate
estimands. They differ in time zero, eligibility conditioning, follow-up scale, and target weighting.
Their point estimates must not be pooled or used interchangeably. The final manuscript will frame
opposite-direction estimates as design sensitivity and will not select an analysis based on effect
direction.

Current gate: `{decision}`.

Generated: {datetime.now(timezone.utc).isoformat()}.
"""
        write_text(original + appendix, plan_v3)

    config_v2 = root / "paper_A_treatment_effects" / "primary_estimand_CANDIDATE_V2.json"
    config_v3 = root / "paper_A_treatment_effects" / "primary_estimand_CANDIDATE_V3.json"
    if config_v2.exists():
        try:
            data = json.loads(config_v2.read_text(encoding="utf-8"))
        except Exception:
            data = {"source_file": str(config_v2)}
        data["protocol_status"] = "CANDIDATE_V3_NOT_LOCKED"
        data["stage13"] = {
            "decision": decision,
            "landmark_and_ccw_are_distinct_estimands": True,
            "full_bootstrap_locked": False,
            "ccw_curve_export_required": curve_status != "CCW_CONDITIONAL_RMST_COMPUTED",
        }
        write_json(data, config_v3)

    print("=" * 112)
    print("STAGE 50 — STAGE 13 DECISION")
    print("=" * 112)
    print(rows.to_string(index=False))
    print(f"\nReport: {tables / '50_stage13_decision.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
