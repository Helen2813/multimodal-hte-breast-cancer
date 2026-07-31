#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage18_utils import (
    dataframe_console,
    ensure_stage18_dirs,
    load_stage18_config,
    markdown_table,
    project_root,
    read_csv,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_stage18_dirs(root)
    cfg = load_stage18_config(root)
    tables = root / "results/tables"

    decision_path = tables / "70_stage17_decision.csv"
    checks_path = tables / "70_stage17_decision_checks.csv"
    if not decision_path.exists() or not checks_path.exists():
        raise FileNotFoundError("Stage 17 decision outputs are required; Stage 17 is not rerun.")

    decision = read_csv(decision_path)
    checks = read_csv(checks_path)
    if len(decision) != 1:
        raise RuntimeError(f"Expected one Stage 17 decision row, found {len(decision)}")

    pass_values = checks["pass"].map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"true", "1", "yes"}
    )
    failed = checks.loc[~pass_values].copy()
    expected_failure = "original_fold_loo_spread_reduction_fraction"
    if len(failed) != 1 or str(failed.iloc[0]["check"]) != expected_failure:
        raise RuntimeError(
            "Stage 18 pilot is authorized only when the sole failed Stage 17 gate is "
            f"{expected_failure}. Observed failures: {failed['check'].tolist()}"
        )

    row = decision.iloc[0]
    amendment = pd.DataFrame(
        [
            {
                "stage18_protocol_status": cfg["protocol_status"],
                "stage17_decision": row["stage17_decision"],
                "stage17_passed_gates": int(pass_values.sum()),
                "stage17_total_gates": int(len(checks)),
                "sole_failed_gate": expected_failure,
                "failed_gate_observed": float(failed.iloc[0]["observed"]),
                "failed_gate_threshold": float(failed.iloc[0]["threshold_or_expected"]),
                "stage17_aggregated_effect_days": float(row["aggregated_30_repeat_effect_days"]),
                "stage17_if_se_days": float(row["aggregated_30_repeat_if_se_days"]),
                "stage17_fraction_positive_repeats": float(row["repeat_fraction_positive"]),
                "stage17_between_split_sd_days": float(row["repeat_sd_effect_days"]),
                "amended_interpretation": cfg["design_amendment"]["interpretation"],
                "bootstrap_pilot_authorized": True,
                "full_publication_bootstrap_locked": True,
            }
        ]
    )
    write_csv(amendment, tables / "71_stage18_protocol_amendment.csv")

    report = f"""# Stage 18 protocol amendment

This amendment does not alter or delete the Stage 17 output. It records why a small patient-level
bootstrap pilot is now appropriate.

{markdown_table(amendment)}

## Rationale

The sole failed Stage 17 check deleted an entire original nuisance fold, approximately one fifth of
the cohort. That diagnostic measures sensitivity to patient composition. Repeated cross-fitting can
reduce nuisance-partition randomness, but it cannot be expected to eliminate uncertainty caused by
removing a large fraction of the patients. Patient-level resampling is therefore the next diagnostic.

The full publication bootstrap remains locked. Stage 18 runs only a 30-repetition computational and
numerical feasibility pilot.
"""
    write_text(report, tables / "71_stage18_protocol_amendment.md")

    print("=" * 124)
    print("STAGE 71 - STAGE 18 PROTOCOL AMENDMENT AND PREFLIGHT")
    print("=" * 124)
    print("Stage 17 decision")
    print(dataframe_console(decision))
    print("\nStage 17 checks")
    print(dataframe_console(checks))
    print("\nProtocol amendment")
    print(dataframe_console(amendment))
    print("\nImportant: Stage 17 is not rerun. The full publication bootstrap remains locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
