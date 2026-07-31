#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage16_utils import (
    aipw_components,
    ensure_dirs,
    exact_landmark_payload,
    factual_prediction_metrics,
    generate_outcome_predictions,
    load_config,
    markdown_table,
    project_root,
    write_csv,
    write_json,
    write_text,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    tables = root / "results/tables"
    local = root / "data/derived/stage16"

    payload = exact_landmark_payload()
    model_rows = []
    calibration_rows = []
    prediction_frame = pd.DataFrame(
        {
            "local_row_index": np.arange(len(payload["frame"])),
            "fold": payload["fold"],
            "treatment": payload["a"],
            "event": payload["event"],
            "observed_time": payload["observed_time"],
            "propensity": payload["e"],
            "ipcw_rmst_pseudo": payload["y"],
        }
    )

    for model_name in cfg["outcome_models"]:
        mu0, mu1 = generate_outcome_predictions(payload, model_name, cfg)
        comp = aipw_components(
            payload["y"], payload["a"], payload["e"], mu0, mu1
        )
        model_rows.append(
            {
                "model": model_name,
                **comp["summary"],
                "mu0_min": float(mu0.min()),
                "mu0_max": float(mu0.max()),
                "mu1_min": float(mu1.min()),
                "mu1_max": float(mu1.max()),
                "fraction_mu0_outside_0_horizon": float(
                    np.mean((mu0 < 0) | (mu0 > payload["horizon"]))
                ),
                "fraction_mu1_outside_0_horizon": float(
                    np.mean((mu1 < 0) | (mu1 > payload["horizon"]))
                ),
            }
        )
        calibration_rows.append(
            factual_prediction_metrics(
                payload["y"],
                payload["a"],
                payload["e"],
                mu0,
                mu1,
                payload["horizon"],
                model_name,
            )
        )
        prediction_frame[f"{model_name}__mu0"] = mu0
        prediction_frame[f"{model_name}__mu1"] = mu1

    summary = pd.DataFrame(model_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    exact = float(
        summary.loc[
            summary["model"] == "arm_ridge_unbounded", "estimate_days"
        ].iloc[0]
    )
    summary["difference_from_exact_ridge_days"] = summary["estimate_days"] - exact

    write_csv(summary, tables / "63_outcome_model_robustness.csv")
    write_csv(calibration, tables / "63_outcome_model_calibration.csv")
    write_csv(
        prediction_frame,
        local / "63_outcome_nuisance_predictions_LOCAL_ONLY.csv",
    )
    write_json(
        {
            "status": "FIXED_SENSITIVITY_REGISTRY_NOT_FOR_MODEL_SELECTION",
            "models": cfg["outcome_models"],
            "hgb_parameters": cfg["fixed_hgb_parameters"],
            "common_elements": [
                "same 559 patients",
                "same repeat-1 outer folds",
                "same frozen propensity scores",
                "same censoring model",
                "same IPCW-RMST pseudo-outcomes",
                "same ATO AIPW formula",
            ],
        },
        tables / "63_fixed_outcome_model_registry.json",
    )
    write_text(
        f"""# Outcome-model robustness

## AIPW estimates and components

{markdown_table(summary)}

## Factual out-of-fold calibration

{markdown_table(calibration)}

These models are a fixed robustness registry. They must not be ranked or selected according to
which one produces the most favorable treatment effect.
""",
        tables / "63_outcome_model_robustness.md",
    )

    print("=" * 118)
    print("STAGE 63 — OUTCOME-MODEL ROBUSTNESS")
    print("=" * 118)
    print(summary.to_string(index=False))
    print("\nFactual calibration")
    print(calibration.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
