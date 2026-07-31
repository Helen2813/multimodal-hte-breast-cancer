#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from _stage15_utils import ensure_dirs, markdown_table, project_root, read_csv, write_csv, write_json, write_text


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    tables = root / "results/tables"
    bridge = read_csv(tables / "57_common_target_estimator_bridge.csv")
    diagnostics = read_csv(tables / "57_bridge_diagnostics.csv").iloc[0]
    truncation = read_csv(tables / "58_reestimated_truncation_bootstrap_summary.csv")
    paired = read_csv(tables / "58_reestimated_truncation_paired_summary.csv")

    robust = bool((paired["truncation_status"] == "DIRECTION_ROBUST_TO_REESTIMATED_TRUNCATION").all())
    bridge_status = str(diagnostics["bridge_status"])
    if not robust:
        decision = "HOLD_PUBLICATION_BOOTSTRAP_CCW_IS_TRUNCATION_SENSITIVE"
    elif bridge_status == "ATO_TARGET_WEIGHTING_RECONCILES_DIRECTION":
        decision = "TARGET_POPULATION_EXPLAINS_SIGN_DISAGREEMENT_LOCK_SEPARATE_ESTIMANDS"
    elif bridge_status == "OUTCOME_AUGMENTATION_IS_PRIMARY_SIGN_BRIDGE":
        decision = "OUTCOME_AUGMENTATION_DRIVES_SIGN_DISAGREEMENT_REQUIRE_ESTIMATOR_SENSITIVITY"
    else:
        decision = "ADHERENCE_OR_CENSORING_MODEL_DRIVES_SIGN_DISAGREEMENT_REQUIRE_BRIDGE_ESTIMATOR"

    summary = pd.DataFrame([{
        "stage15_decision": decision,
        "bridge_status": bridge_status,
        "reestimated_truncation_robust": robust,
        "protocol_status": "CANDIDATE_V5_NOT_LOCKED",
        "publication_bootstrap_locked": True,
        "paper_claim": "reliability across time-zero, target-population, and nuisance-model choices",
    }])
    write_csv(summary, tables / "59_stage15_decision.csv")
    write_text(
        f"""# Stage 15 decision report

**Decision:** `{decision}`

**Protocol status:** `CANDIDATE_V5_NOT_LOCKED`

## Common-target estimator bridge

{markdown_table(bridge)}

## Re-estimated CCW truncation bootstrap

{markdown_table(truncation)}

## Paired truncation comparisons

{markdown_table(paired)}

The full publication bootstrap remains locked. No post-hoc selection of the positive landmark
estimate is permitted.
""",
        tables / "59_stage15_decision.md",
    )

    old_plan = root / "paper_A_treatment_effects/analysis_plan_CANDIDATE_V4.md"
    new_plan = root / "paper_A_treatment_effects/analysis_plan_CANDIDATE_V5.md"
    if old_plan.exists():
        amendment = f"""

## Stage 15 common-target and re-estimated truncation amendment

Status: **CANDIDATE_V5_NOT_LOCKED**.

Decision: `{decision}`.

Bridge classification: `{bridge_status}`.

The re-estimated truncation bootstrap is interpreted as
`{'direction-robust' if robust else 'truncation-sensitive'}`. The publication bootstrap remains
locked until the estimator bridge is reflected in the final protocol.

Generated: {datetime.now(timezone.utc).isoformat()}.
"""
        write_text(old_plan.read_text(encoding="utf-8", errors="ignore").rstrip() + amendment, new_plan)

    old_json = root / "paper_A_treatment_effects/primary_estimand_CANDIDATE_V4.json"
    new_json = root / "paper_A_treatment_effects/primary_estimand_CANDIDATE_V5.json"
    if old_json.exists():
        try:
            data = json.loads(old_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["protocol_status"] = "CANDIDATE_V5_NOT_LOCKED"
        data["stage15"] = {
            "decision": decision,
            "bridge_status": bridge_status,
            "reestimated_truncation_robust": robust,
            "publication_bootstrap_locked": True,
        }
        write_json(data, new_json)

    print("=" * 116)
    print("STAGE 59 — STAGE 15 DECISION")
    print("=" * 116)
    print(summary.to_string(index=False))
    print(f"\nReport: {tables / '59_stage15_decision.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
