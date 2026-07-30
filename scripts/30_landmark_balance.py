from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _stage9_utils import (
    LANDMARKS,
    crossfit_propensity,
    effective_sample_size,
    smd,
)


COHORTS = (
    "outer_hormone_hrpos_her2neg",
    "outer_chemo_tnbc",
)


def main() -> int:
    ensure_dirs()
    compact_dir = DERIVED_DIR / "landmark_compact"
    split_dir = DERIVED_DIR / "landmark_splits"
    weight_dir = DERIVED_DIR / "landmark_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    print("=" * 115)
    print("STAGE 30 — LANDMARK PROPENSITY AND BALANCE")
    print("=" * 115)

    summary_rows = []

    for cohort in COHORTS:
        for landmark in LANDMARKS:
            key = f"{cohort}_landmark{landmark}"
            compact = read_table(
                compact_dir / f"{key}_compact.csv"
            )
            splits = read_table(split_dir / f"{key}_splits.csv")
            features = [
                c for c in compact.columns if c.startswith("W_")
            ] + ["diagnosis_year", "diagnosis_year_missing"]

            ps, tuning = crossfit_propensity(
                compact, features, splits, repeat=1, seed=3000
            )
            a = compact["analysis_treatment"].astype(int).to_numpy()
            ow = np.where(a == 1, 1.0 - ps, ps)

            balance_rows = []
            for col in features:
                x = pd.to_numeric(
                    compact[col], errors="coerce"
                ).to_numpy(float)
                balance_rows.append(
                    {
                        "cohort": cohort,
                        "landmark_day": landmark,
                        "feature": col,
                        "smd_unweighted": smd(
                            x, a, np.ones(len(a))
                        ),
                        "smd_overlap": smd(x, a, ow),
                        "missing_fraction": float(
                            np.mean(~np.isfinite(x))
                        ),
                    }
                )
            balance = pd.DataFrame(balance_rows)
            balance["abs_smd_overlap"] = balance["smd_overlap"].abs()
            balance = balance.sort_values(
                "abs_smd_overlap", ascending=False
            )
            balance.to_csv(
                table_dir / f"30_balance_{key}.csv",
                index=False,
            )
            tuning["cohort"] = cohort
            tuning["landmark_day"] = landmark
            tuning.to_csv(
                table_dir / f"30_tuning_{key}.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "patient_id_normalized": compact[
                        "patient_id_normalized"
                    ],
                    "analysis_treatment": a,
                    "propensity_score_oof": ps,
                    "overlap_weight": ow,
                }
            ).to_csv(
                weight_dir / f"{key}_weights.csv",
                index=False,
            )

            max_smd = float(balance["abs_smd_overlap"].max())
            mean_smd = float(balance["abs_smd_overlap"].mean())
            ess_t = effective_sample_size(ow[a == 1])
            ess_c = effective_sample_size(ow[a == 0])
            events = int(compact["analysis_event"].sum())

            if (
                max_smd <= 0.10
                and ess_t >= 80
                and ess_c >= 80
                and events >= 30
            ):
                status = "PRIMARY_LANDMARK_READY"
            elif (
                max_smd <= 0.15
                and ess_t >= 40
                and ess_c >= 30
                and events >= 20
            ):
                status = "EXPLORATORY_LANDMARK_READY"
            else:
                status = "LANDMARK_NOT_READY"

            row = {
                "cohort": cohort,
                "landmark_day": landmark,
                "n": len(compact),
                "treated": int(a.sum()),
                "control": int((a == 0).sum()),
                "events": events,
                "max_abs_smd_overlap": max_smd,
                "mean_abs_smd_overlap": mean_smd,
                "ess_treated": ess_t,
                "ess_control": ess_c,
                "ps_min": float(ps.min()),
                "ps_p05": float(np.quantile(ps, 0.05)),
                "ps_median": float(np.median(ps)),
                "ps_p95": float(np.quantile(ps, 0.95)),
                "ps_max": float(ps.max()),
                "balance_status": status,
            }
            summary_rows.append(row)

            print("\n" + "-" * 115)
            print(f"{cohort}, landmark={landmark}")
            print(pd.DataFrame([row]).to_string(index=False))
            print("\nTop residual imbalances")
            print(
                balance[
                    ["feature", "smd_unweighted", "smd_overlap"]
                ].head(15).to_string(index=False)
            )
            print("\nPropensity tuning")
            print(tuning.to_string(index=False))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        table_dir / "30_landmark_balance_summary.csv", index=False
    )
    print("\n" + "=" * 115)
    print("FINAL LANDMARK BALANCE SUMMARY")
    print("=" * 115)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
