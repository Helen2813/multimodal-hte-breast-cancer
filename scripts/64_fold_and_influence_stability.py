#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage16_utils import (
    aipw_components,
    ensure_dirs,
    load_config,
    project_root,
    read_csv,
    subset_aipw_effect,
    write_csv,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    tables = root / "results/tables"
    local = root / "data/derived/stage16"
    cfg = load_config(root)

    path = local / "63_outcome_nuisance_predictions_LOCAL_ONLY.csv"
    df = read_csv(path)
    y = pd.to_numeric(df["ipcw_rmst_pseudo"], errors="raise").to_numpy(float)
    a = pd.to_numeric(df["treatment"], errors="raise").astype(int).to_numpy()
    e = pd.to_numeric(df["propensity"], errors="raise").to_numpy(float)
    fold = pd.to_numeric(df["fold"], errors="raise").astype(int).to_numpy()

    fold_rows = []
    loo_rows = []
    influence_rows = []
    for model in cfg["outcome_models"]:
        mu0 = pd.to_numeric(df[f"{model}__mu0"], errors="raise").to_numpy(float)
        mu1 = pd.to_numeric(df[f"{model}__mu1"], errors="raise").to_numpy(float)
        full = aipw_components(y, a, e, mu0, mu1)
        patient = full["patient"].copy()
        patient["model"] = model
        patient["local_row_index"] = df["local_row_index"].to_numpy()
        patient["fold"] = fold
        influence_rows.append(patient)

        for f in sorted(np.unique(fold)):
            test = fold == f
            keep = fold != f
            fold_rows.append(
                {
                    "model": model,
                    "fold": int(f),
                    "fold_n": int(test.sum()),
                    "fold_treated": int(a[test].sum()),
                    "fold_control": int((1 - a[test]).sum()),
                    "fold_effect_days": subset_aipw_effect(
                        y, a, e, mu0, mu1, test
                    ),
                }
            )
            loo_rows.append(
                {
                    "model": model,
                    "omitted_fold": int(f),
                    "retained_n": int(keep.sum()),
                    "leave_one_fold_out_effect_days": subset_aipw_effect(
                        y, a, e, mu0, mu1, keep
                    ),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    loo_df = pd.DataFrame(loo_rows)
    influence = pd.concat(influence_rows, ignore_index=True)
    summary_rows = []
    for model in cfg["outcome_models"]:
        fvals = fold_df.loc[fold_df["model"] == model, "fold_effect_days"]
        lvals = loo_df.loc[
            loo_df["model"] == model, "leave_one_fold_out_effect_days"
        ]
        summary_rows.append(
            {
                "model": model,
                "fold_effect_min": fvals.min(),
                "fold_effect_max": fvals.max(),
                "fold_effect_spread": fvals.max() - fvals.min(),
                "fold_fraction_positive": (fvals > 0).mean(),
                "loo_effect_min": lvals.min(),
                "loo_effect_max": lvals.max(),
                "loo_effect_spread": lvals.max() - lvals.min(),
                "loo_fraction_positive": (lvals > 0).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    exact = influence[influence["model"] == "arm_ridge_unbounded"].copy()
    exact["absolute_influence"] = exact["influence"].abs()
    top = exact.nlargest(25, "absolute_influence")[
        [
            "local_row_index",
            "fold",
            "normalized_contribution_days",
            "influence",
            "absolute_influence",
        ]
    ]

    write_csv(fold_df, tables / "64_fold_specific_effects.csv")
    write_csv(loo_df, tables / "64_leave_one_fold_out_effects.csv")
    write_csv(summary, tables / "64_fold_stability_summary.csv")
    write_csv(
        influence,
        local / "64_patient_influence_components_LOCAL_ONLY.csv",
    )
    write_csv(top, tables / "64_top_influence_rows_deidentified.csv")

    print("=" * 118)
    print("STAGE 64 — FOLD AND INFLUENCE STABILITY")
    print("=" * 118)
    print(summary.to_string(index=False))
    print("\nFold-specific effects")
    print(fold_df.to_string(index=False))
    print("\nLeave-one-fold-out effects")
    print(loo_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
