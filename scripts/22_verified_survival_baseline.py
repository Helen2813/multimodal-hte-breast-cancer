from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


TAU_DAYS = 1825.0
BOOTSTRAPS = 500
SEED = 20260729


def weighted_km(
    times: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    tau: float,
) -> tuple[float, float]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(times)
        & np.isfinite(events)
        & np.isfinite(weights)
        & (times >= 0)
        & (weights >= 0)
    )
    times, events, weights = times[valid], events[valid], weights[valid]

    event_times = np.sort(
        np.unique(times[(events == 1) & (times <= tau)])
    )
    survival = 1.0
    previous = 0.0
    rmst = 0.0

    for time in event_times:
        rmst += survival * max(0.0, time - previous)
        at_risk = weights[times >= time].sum()
        deaths = weights[(times == time) & (events == 1)].sum()
        if at_risk > 0:
            survival *= max(0.0, 1.0 - deaths / at_risk)
        previous = float(time)

    if previous < tau:
        rmst += survival * (tau - previous)
    return float(survival), float(rmst)


def contrast(
    df: pd.DataFrame, weights: np.ndarray
) -> dict[str, float]:
    t = pd.to_numeric(
        df["analysis_treatment"], errors="raise"
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        df["analysis_event"], errors="raise"
    ).astype(int).to_numpy()
    time = pd.to_numeric(
        df["analysis_time"], errors="coerce"
    ).to_numpy(float)

    values = {}
    for arm, name in ((0, "control"), (1, "treated")):
        mask = t == arm
        surv, rmst = weighted_km(
            time[mask], event[mask], weights[mask], TAU_DAYS
        )
        values[f"survival_{name}"] = surv
        values[f"rmst_{name}"] = rmst

    values["survival_difference"] = (
        values["survival_treated"] - values["survival_control"]
    )
    values["rmst_difference_days"] = (
        values["rmst_treated"] - values["rmst_control"]
    )
    return values


def fixed_weight_bootstrap(
    df: pd.DataFrame,
    weights: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(df)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = df.iloc[idx].reset_index(drop=True)
        t = pd.to_numeric(
            boot["analysis_treatment"], errors="raise"
        ).astype(int)
        if t.nunique() < 2:
            continue
        est = contrast(boot, weights[idx])
        estimates.append(
            [est["survival_difference"], est["rmst_difference_days"]]
        )
    arr = np.asarray(estimates)
    return {
        "survival_ci_low": float(np.quantile(arr[:, 0], 0.025)),
        "survival_ci_high": float(np.quantile(arr[:, 0], 0.975)),
        "rmst_ci_low_days": float(np.quantile(arr[:, 1], 0.025)),
        "rmst_ci_high_days": float(np.quantile(arr[:, 1], 0.975)),
        "successful_bootstraps": len(arr),
    }


def load_weights(
    cohort: str, strategy: str
) -> pd.DataFrame:
    table_dir = RESULTS_DIR / "tables"
    if strategy == "compact_overlap":
        path = table_dir / f"19_verified_weights_{cohort}.csv"
        weights = read_table(path)[
            ["patient_id_normalized", "overlap_weight"]
        ].rename(columns={"overlap_weight": "weight"})
    elif strategy == "full_elastic_net_overlap":
        path = table_dir / f"21_full_weights_{cohort}.csv"
        weights = read_table(path)[
            [
                "patient_id_normalized",
                "overlap_weight_full_elastic_net",
            ]
        ].rename(
            columns={
                "overlap_weight_full_elastic_net": "weight"
            }
        )
    else:
        raise ValueError(strategy)
    return weights


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    cohort_dir = DERIVED_DIR / "verified_cohorts"

    print("=" * 110)
    print("STAGE 22 — VERIFIED SURVIVAL BASELINE")
    print("=" * 110)
    print(
        "These are weighted Kaplan–Meier diagnostics with a "
        "patient-level fixed-weight bootstrap. They are not the final "
        "doubly robust Paper A estimator."
    )

    strategies = (
        "compact_overlap",
        "full_elastic_net_overlap",
    )
    rows = []

    for cohort_path in sorted(cohort_dir.glob("*_verified.csv")):
        cohort = cohort_path.stem.replace("_verified", "")
        df = read_table(cohort_path)

        print("\n" + "=" * 110)
        print(f"COHORT: {cohort}")
        print(
            f"n={len(df)}, treated="
            f"{int(pd.to_numeric(df['analysis_treatment']).sum())}, "
            f"controls={int((1-pd.to_numeric(df['analysis_treatment'])).sum())}, "
            f"events={int(pd.to_numeric(df['analysis_event']).sum())}"
        )

        for strategy in strategies:
            weight_df = load_weights(cohort, strategy)
            merged = df.merge(
                weight_df,
                on="patient_id_normalized",
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(df):
                raise ValueError(
                    f"{cohort}/{strategy}: weight rows do not match cohort."
                )
            w = pd.to_numeric(
                merged["weight"], errors="raise"
            ).to_numpy(float)
            point = contrast(merged, w)
            ci = fixed_weight_bootstrap(
                merged, w, BOOTSTRAPS, SEED
            )
            row = {
                "cohort": cohort,
                "strategy": strategy,
                "tau_days": TAU_DAYS,
                "n": len(merged),
                "treated": int(
                    pd.to_numeric(
                        merged["analysis_treatment"]
                    ).sum()
                ),
                "events": int(
                    pd.to_numeric(merged["analysis_event"]).sum()
                ),
                **point,
                **ci,
                "bootstrap_note": (
                    "patient_bootstrap_with_fixed_estimated_weights"
                ),
            }
            rows.append(row)
            print("\nStrategy:", strategy)
            print(pd.DataFrame([row]).to_string(index=False))

    results = pd.DataFrame(rows).sort_values(
        ["cohort", "strategy"]
    )
    results.to_csv(
        table_dir / "22_verified_survival_baseline.csv",
        index=False,
    )

    print("\n" + "=" * 110)
    print("FINAL STAGE 22 SURVIVAL SUMMARY")
    print("=" * 110)
    print(
        results[
            [
                "cohort",
                "strategy",
                "survival_difference",
                "survival_ci_low",
                "survival_ci_high",
                "rmst_difference_days",
                "rmst_ci_low_days",
                "rmst_ci_high_days",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {table_dir / '22_verified_survival_baseline.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
