#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage12_utils import (
    assemble_landmark_data,
    crossfit_arm_outcomes,
    crossfit_censor_survival,
    crossfit_propensity,
    ipcw_rmst_pseudo,
)
from _stage17_utils import effect_and_patient_components
from _stage18_utils import aggregate_partition_patient_scores, make_grouped_bootstrap_folds
from _stage20_utils import (
    dataframe_console,
    ensure_dirs,
    load_stage20_config,
    project_root,
    refuse_existing_lock,
    write_csv,
)


def fit_partition(
    frame: pd.DataFrame,
    features: list[str],
    a: np.ndarray,
    event: np.ndarray,
    observed_time: np.ndarray,
    partition_number: int,
    base_seed: int,
    cfg: dict,
) -> tuple[dict, pd.DataFrame]:
    est = cfg["final_estimator"]
    groups = np.arange(len(frame), dtype=int)
    last_error: Exception | None = None
    for retry in range(int(est["maximum_partition_retries"])):
        seed = int(base_seed) + retry * 100_000
        try:
            fold, stratification, fold_retry = make_grouped_bootstrap_folds(
                a, event, groups, seed, int(est["n_folds"])
            )
            e = crossfit_propensity(frame, features, fold, "analysis_treatment", seed + 10)
            G, starts, ends, censor_metrics = crossfit_censor_survival(
                frame,
                features,
                fold,
                float(est["horizon_days"]),
                float(est["interval_days"]),
                seed + 100,
            )
            y = ipcw_rmst_pseudo(
                observed_time,
                G,
                starts,
                ends,
                float(est["horizon_days"]),
                float(est["primary_g_min"]),
            )
            mu0_raw, mu1_raw = crossfit_arm_outcomes(frame, features, y, fold, seed + 200)
            mu0 = np.clip(mu0_raw, 0.0, float(est["horizon_days"]))
            mu1 = np.clip(mu1_raw, 0.0, float(est["horizon_days"]))
            summary, patient = effect_and_patient_components(y, a, e, mu0, mu1)
            patient = patient[["h", "score_numerator"]].copy()
            patient.insert(0, "row_index", np.arange(len(frame), dtype=int))
            patient.insert(0, "partition", partition_number)
            row = {
                "partition": partition_number,
                "base_seed": int(base_seed),
                "seed_used": int(seed),
                "nuisance_retry": int(retry),
                "fold_retry": int(fold_retry),
                "fold_stratification": str(stratification),
                **summary,
                "propensity_min": float(np.min(e)),
                "propensity_p01": float(np.quantile(e, 0.01)),
                "propensity_p99": float(np.quantile(e, 0.99)),
                "propensity_max": float(np.max(e)),
                "censor_log_loss": float(censor_metrics["censor_log_loss"]),
                "censor_brier": float(censor_metrics["censor_brier"]),
                "G_min_raw": float(censor_metrics["G_min"]),
                "G_p01_raw": float(censor_metrics["G_p01"]),
                "pseudo_mean": float(np.mean(y)),
                "pseudo_sd": float(np.std(y, ddof=1)),
                "pseudo_p99": float(np.quantile(y, 0.99)),
                "pseudo_max": float(np.max(y)),
                "fraction_mu0_outside_before_bounding": float(
                    np.mean((mu0_raw < 0.0) | (mu0_raw > float(est["horizon_days"])))
                ),
                "fraction_mu1_outside_before_bounding": float(
                    np.mean((mu1_raw < 0.0) | (mu1_raw > float(est["horizon_days"])))
                ),
            }
            return row, patient
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Final point-estimate partition {partition_number} failed after "
        f"{est['maximum_partition_retries']} retries: {last_error}"
    ) from last_error


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    refuse_existing_lock(root)
    cfg = load_stage20_config(root)
    est = cfg["final_estimator"]

    frame, features, _, metadata = assemble_landmark_data()
    frame = frame.copy().reset_index(drop=True)
    a = pd.to_numeric(frame["analysis_treatment"], errors="raise").astype(int).to_numpy()
    event = pd.to_numeric(frame["analysis_event"], errors="raise").astype(int).to_numpy()
    observed_time = pd.to_numeric(frame["analysis_time"], errors="coerce").to_numpy(float)

    checks = pd.DataFrame([
        {"check": "n", "observed": len(frame), "expected": est["expected_n"], "pass": len(frame) == int(est["expected_n"])},
        {"check": "treated", "observed": int(a.sum()), "expected": est["expected_treated"], "pass": int(a.sum()) == int(est["expected_treated"])},
        {"check": "control", "observed": int((1-a).sum()), "expected": est["expected_control"], "pass": int((1-a).sum()) == int(est["expected_control"])},
        {"check": "events", "observed": int(event.sum()), "expected": est["expected_events"], "pass": int(event.sum()) == int(est["expected_events"])},
        {"check": "features", "observed": len(features), "expected": est["expected_features"], "pass": len(features) == int(est["expected_features"])},
        {"check": "partition_seed_count", "observed": len(est["partition_base_seeds"]), "expected": 20, "pass": len(est["partition_base_seeds"]) == 20},
    ])
    if not bool(checks["pass"].all()):
        raise RuntimeError("Final estimator preflight failed.\n" + dataframe_console(checks))

    print("=" * 124)
    print("STAGE 79 - FINAL CANDIDATE V9 20-PARTITION POINT ESTIMATE")
    print("=" * 124)
    print("This is a new final-estimator calculation using the 20 partition seeds approved by Stage 19.")
    print("Stages 15 through 19 are not rerun.")
    print("Preflight checks")
    print(dataframe_console(checks))
    print(f"Cohort: {metadata['cohort']}; n={metadata['n']}; treated={metadata['treated']}; control={metadata['control']}; events={metadata['events']}")

    rows: list[dict] = []
    scores: list[pd.DataFrame] = []
    for partition_number, base_seed in enumerate(est["partition_base_seeds"], start=1):
        row, patient = fit_partition(
            frame, features, a, event, observed_time,
            partition_number, int(base_seed), cfg,
        )
        rows.append(row)
        scores.append(patient)
        print(
            f"partition={partition_number:02d} seed={base_seed} "
            f"effect={row['estimate_days']:.6f} IF_SE={row['if_se_days']:.6f} "
            f"G_min={row['G_min_raw']:.6f} pseudo_p99={row['pseudo_p99']:.6f} "
            f"pseudo_max={row['pseudo_max']:.6f}"
        )

    partitions = pd.DataFrame(rows)
    score_df = pd.concat(scores, ignore_index=True).rename(columns={"row_index": "bootstrap_row_index"})
    score_df["original_patient_group"] = score_df["bootstrap_row_index"]

    prefix_rows: list[dict] = []
    for prefix in (5, 10, 15, 20):
        agg = aggregate_partition_patient_scores(score_df[score_df["partition"] <= prefix])
        vals = partitions.loc[partitions["partition"] <= prefix, "estimate_days"].to_numpy(float)
        prefix_rows.append({
            "prefix_partitions": prefix,
            "estimate_days": float(agg["estimate_days"]),
            "if_se_days": float(agg["if_se_days"]),
            "if_ci_low_days": float(agg["if_ci_low_days"]),
            "if_ci_high_days": float(agg["if_ci_high_days"]),
            "partition_mean_days": float(np.mean(vals)),
            "partition_sd_days": float(np.std(vals, ddof=1)),
            "partition_mcse_days": float(np.std(vals, ddof=1) / np.sqrt(prefix)),
        })
    prefixes = pd.DataFrame(prefix_rows)
    final = prefixes[prefixes["prefix_partitions"] == 20].copy()
    final.insert(0, "protocol_candidate", "PAPER_A_CANDIDATE_V9")
    final["n"] = len(frame)
    final["treated"] = int(a.sum())
    final["control"] = int((1-a).sum())
    final["events"] = int(event.sum())
    final["horizon_days"] = float(est["horizon_days"])
    final["landmark_day"] = int(est["landmark_day"])
    final["g_min"] = float(est["primary_g_min"])
    final["partitions"] = 20

    tables = root / "results/tables"
    local = root / "data/derived/stage20"
    write_csv(checks, tables / "79_candidate_v9_preflight_checks.csv")
    write_csv(partitions, tables / "79_candidate_v9_partition_estimates.csv")
    write_csv(prefixes, tables / "79_candidate_v9_prefix_convergence.csv")
    write_csv(final, tables / "79_candidate_v9_final_point_estimate.csv")
    write_csv(score_df, local / "79_candidate_v9_patient_scores_LOCAL_ONLY.csv")

    print("\nAll 20 partition estimates")
    print(dataframe_console(partitions[[
        "partition", "base_seed", "estimate_days", "if_se_days",
        "direct_ato_ipw_effect_days", "G_min_raw", "propensity_p01",
        "propensity_p99", "pseudo_p99", "pseudo_max", "nuisance_retry"
    ]]))
    print("\nPrefix convergence")
    print(dataframe_console(prefixes))
    print("\nFinal Candidate V9 point estimate")
    print(dataframe_console(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
