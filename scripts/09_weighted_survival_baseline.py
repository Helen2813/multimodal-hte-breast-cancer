from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


TAU_DAYS = 1825.0
BOOTSTRAPS = 500
SEED = 2026


def weighted_km(
    times: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    tau: float,
) -> tuple[float, float, pd.DataFrame]:
    """
    Weighted Kaplan–Meier point estimate.

    Returns survival at tau, RMST to tau, and the step curve.
    This accounts for right censoring through the KM risk set. The treatment
    weights define the target population; it is not a doubly robust estimator.
    """
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

    event_times = np.sort(np.unique(times[(events == 1) & (times <= tau)]))
    survival = 1.0
    previous_time = 0.0
    rmst = 0.0
    rows = [{"time": 0.0, "survival": 1.0}]

    for time in event_times:
        rmst += survival * max(0.0, time - previous_time)
        at_risk = weights[times >= time].sum()
        event_weight = weights[(times == time) & (events == 1)].sum()
        if at_risk > 0:
            survival *= max(0.0, 1.0 - event_weight / at_risk)
        rows.append({"time": float(time), "survival": float(survival)})
        previous_time = float(time)

    if previous_time < tau:
        rmst += survival * (tau - previous_time)

    return float(survival), float(rmst), pd.DataFrame(rows)


def estimate_contrast(
    df: pd.DataFrame,
    weights: np.ndarray,
    tau: float,
) -> dict[str, float]:
    t = pd.to_numeric(df["analysis_treatment"], errors="raise").astype(int).to_numpy()
    time = pd.to_numeric(df["analysis_time"], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(df["analysis_event"], errors="raise").astype(int).to_numpy()

    results = {}
    curves = {}
    for group, name in ((0, "control"), (1, "treated")):
        mask = t == group
        survival, rmst, curve = weighted_km(
            time[mask], event[mask], weights[mask], tau
        )
        results[f"survival_{name}_tau"] = survival
        results[f"rmst_{name}_days"] = rmst
        curves[name] = curve

    results["survival_difference_treated_minus_control"] = (
        results["survival_treated_tau"] - results["survival_control_tau"]
    )
    results["rmst_difference_days_treated_minus_control"] = (
        results["rmst_treated_days"] - results["rmst_control_days"]
    )
    results["_curves"] = curves
    return results


def bootstrap_contrast(
    df: pd.DataFrame,
    weights: np.ndarray,
    tau: float,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """
    Patient-level bootstrap with fixed estimated treatment weights.

    This quantifies sampling variability conditional on the fitted propensity
    model. It is a baseline diagnostic, not the final inferential procedure.
    """
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = df.iloc[idx].reset_index(drop=True)
        boot_weights = weights[idx]
        t = pd.to_numeric(boot["analysis_treatment"], errors="raise").astype(int)
        if t.nunique() < 2:
            continue
        est = estimate_contrast(boot, boot_weights, tau)
        estimates.append(
            (
                est["survival_difference_treated_minus_control"],
                est["rmst_difference_days_treated_minus_control"],
            )
        )

    arr = np.asarray(estimates, dtype=float)
    if arr.shape[0] < max(50, int(0.8 * n_boot)):
        raise RuntimeError("Too many bootstrap samples failed.")

    return {
        "survival_difference_ci_low": float(np.quantile(arr[:, 0], 0.025)),
        "survival_difference_ci_high": float(np.quantile(arr[:, 0], 0.975)),
        "rmst_difference_ci_low_days": float(np.quantile(arr[:, 1], 0.025)),
        "rmst_difference_ci_high_days": float(np.quantile(arr[:, 1], 0.975)),
        "bootstrap_successful": int(arr.shape[0]),
    }


def run_one(
    cohort_path: Path,
    propensity_path: Path,
    cohort_name: str,
) -> list[dict[str, object]]:
    df = read_table(cohort_path)
    ps = read_table(propensity_path)

    merged = df.merge(
        ps[
            [
                "patient_id_normalized",
                "ipw_truncated_p99",
                "overlap_weight",
                "retained_trim_0.10_0.90",
            ]
        ],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df):
        raise ValueError(f"{cohort_name}: propensity rows do not match cohort rows.")

    methods = {
        "overlap_weighting": pd.to_numeric(
            merged["overlap_weight"], errors="raise"
        ).to_numpy(dtype=float),
        "stabilized_ipw_p99": pd.to_numeric(
            merged["ipw_truncated_p99"], errors="raise"
        ).to_numpy(dtype=float),
        "trim_0.10_0.90_unweighted": pd.to_numeric(
            merged["retained_trim_0.10_0.90"], errors="raise"
        ).to_numpy(dtype=float),
    }

    rows = []
    for method, weights in methods.items():
        point = estimate_contrast(merged, weights, TAU_DAYS)
        ci = bootstrap_contrast(
            merged, weights, TAU_DAYS, BOOTSTRAPS, SEED
        )
        rows.append(
            {
                "cohort": cohort_name,
                "method": method,
                "tau_days": TAU_DAYS,
                "n": len(merged),
                "treated": int(
                    pd.to_numeric(merged["analysis_treatment"]).sum()
                ),
                "events": int(pd.to_numeric(merged["analysis_event"]).sum()),
                "survival_control_tau": point["survival_control_tau"],
                "survival_treated_tau": point["survival_treated_tau"],
                "survival_difference_treated_minus_control": point[
                    "survival_difference_treated_minus_control"
                ],
                "survival_difference_ci_low": ci[
                    "survival_difference_ci_low"
                ],
                "survival_difference_ci_high": ci[
                    "survival_difference_ci_high"
                ],
                "rmst_control_days": point["rmst_control_days"],
                "rmst_treated_days": point["rmst_treated_days"],
                "rmst_difference_days_treated_minus_control": point[
                    "rmst_difference_days_treated_minus_control"
                ],
                "rmst_difference_ci_low_days": ci[
                    "rmst_difference_ci_low_days"
                ],
                "rmst_difference_ci_high_days": ci[
                    "rmst_difference_ci_high_days"
                ],
                "bootstrap_type": "patient_bootstrap_fixed_treatment_weights",
                "bootstrap_successful": ci["bootstrap_successful"],
            }
        )

        plt.figure(figsize=(8, 5))
        for group_name, curve in point["_curves"].items():
            plt.step(
                curve["time"],
                curve["survival"],
                where="post",
                label=group_name.capitalize(),
            )
        plt.xlim(0, TAU_DAYS)
        plt.ylim(0, 1.02)
        plt.xlabel("Days")
        plt.ylabel("Weighted survival")
        plt.title(f"{cohort_name.replace('_', ' ')} — {method}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR
            / "figures"
            / f"09_weighted_survival_{cohort_name}_{method}.png",
            dpi=220,
        )
        plt.close()

    return rows


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    table_dir = RESULTS_DIR / "tables"

    analyses = {
        "outer_hormone_hrpos_her2neg": (
            cohort_dir / "outer_hormone_hrpos_her2neg.csv",
            table_dir
            / "06_compact_propensity_outer_hormone_hrpos_her2neg.csv",
        ),
        "outer_chemo_tnbc": (
            cohort_dir / "outer_chemo_tnbc.csv",
            table_dir / "06_compact_propensity_outer_chemo_tnbc.csv",
        ),
    }

    rows = []
    for cohort_name, (cohort_path, propensity_path) in analyses.items():
        if not cohort_path.exists():
            raise FileNotFoundError(cohort_path)
        if not propensity_path.exists():
            raise FileNotFoundError(propensity_path)
        print(f"Weighted survival baseline: {cohort_name}")
        rows.extend(run_one(cohort_path, propensity_path, cohort_name))

    results = pd.DataFrame(rows)
    results.to_csv(
        table_dir / "09_weighted_survival_baseline.csv", index=False
    )

    print("\nWeighted survival baseline:")
    print(
        results[
            [
                "cohort",
                "method",
                "survival_difference_treated_minus_control",
                "survival_difference_ci_low",
                "survival_difference_ci_high",
                "rmst_difference_days_treated_minus_control",
                "rmst_difference_ci_low_days",
                "rmst_difference_ci_high_days",
            ]
        ].to_string(index=False)
    )
    print(
        "\nImportant: these are weighted Kaplan–Meier baseline estimates with "
        "fixed-weight bootstrap, not final doubly robust inference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
