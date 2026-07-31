#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage16_utils import (
    aipw_components,
    ensure_dirs,
    exact_landmark_payload,
    factual_prediction_metrics,
    load_config,
    markdown_table,
    project_root,
    write_csv,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    design = cfg["design"]
    tables = root / "results/tables"
    local = root / "data/derived/stage16"

    payload = exact_landmark_payload()
    comp = aipw_components(
        payload["y"],
        payload["a"],
        payload["e"],
        payload["mu0"],
        payload["mu1"],
    )
    summary = pd.DataFrame(
        [
            {
                "model": "arm_ridge_unbounded_exact_stage12",
                **comp["summary"],
                "exact_stage12_estimate_days": payload["theta"],
                "replication_difference_days": comp["summary"]["estimate_days"]
                - payload["theta"],
                "horizon_days": payload["horizon"],
                "n": len(payload["frame"]),
                "treated": int(payload["a"].sum()),
                "control": int((1 - payload["a"]).sum()),
                "events": int(payload["event"].sum()),
                "pseudo_mean": float(np.mean(payload["y"])),
                "pseudo_sd": float(np.std(payload["y"], ddof=1)),
                "pseudo_p99": float(np.quantile(payload["y"], 0.99)),
                "pseudo_max": float(np.max(payload["y"])),
            }
        ]
    )
    calibration = factual_prediction_metrics(
        payload["y"],
        payload["a"],
        payload["e"],
        payload["mu0"],
        payload["mu1"],
        payload["horizon"],
        "arm_ridge_unbounded",
    )

    patient = pd.DataFrame(
        {
            "local_row_index": np.arange(len(payload["frame"])),
            "fold": payload["fold"],
            "treatment": payload["a"],
            "event": payload["event"],
            "observed_time": payload["observed_time"],
            "propensity": payload["e"],
            "ipcw_rmst_pseudo": payload["y"],
            "mu0": payload["mu0"],
            "mu1": payload["mu1"],
        }
    )
    patient = pd.concat(
        [patient.reset_index(drop=True), comp["patient"].reset_index(drop=True)],
        axis=1,
    )
    # Deliberately exclude patient identifiers. This file is local diagnostic material.
    write_csv(patient, local / "62_exact_aipw_patient_components_LOCAL_ONLY.csv")
    write_csv(summary, tables / "62_exact_landmark_aipw_decomposition.csv")
    write_csv(calibration, tables / "62_exact_ridge_calibration.csv")

    write_text(
        f"""# Exact Stage 12 landmark AIPW decomposition

{markdown_table(summary)}

## Factual out-of-fold prediction diagnostics

{markdown_table(calibration)}

The `direct_ato_ipw_effect_days` uses the same IPCW-RMST pseudo-outcome and frozen propensity
scores as the AIPW estimator but removes outcome augmentation. It is a cleaner bridge than
comparing AIPW directly with a separately constructed weighted Kaplan-Meier curve.

Patient-level component rows are written without patient identifiers under
`data/derived/stage16` and must not be committed.
""",
        tables / "62_exact_landmark_aipw_decomposition.md",
    )

    print("=" * 118)
    print("STAGE 62 — EXACT LANDMARK AIPW DECOMPOSITION")
    print("=" * 118)
    print(summary.to_string(index=False))
    print("\nCalibration")
    print(calibration.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
