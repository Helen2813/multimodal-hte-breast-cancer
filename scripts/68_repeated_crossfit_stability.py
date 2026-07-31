#!/usr/bin/env python3
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from _stage12_utils import (
    BASE_SEED,
    LANDMARK_HORIZON,
    LANDMARK_INTERVAL,
    crossfit_arm_outcomes,
    crossfit_censor_survival,
    crossfit_propensity,
    ipcw_rmst_pseudo,
)
from _stage17_utils import (
    append_replace_repeat,
    checkpoint_identity,
    completed_repeat_numbers,
    dataframe_console,
    effect_and_patient_components,
    ensure_stage17_dirs,
    exact_landmark_payload,
    load_stage17_config,
    make_repeated_fold_assignment,
    original_fold_effects,
    project_root,
    read_csv,
    stable_repeat_seeds,
    validate_or_create_checkpoint_identity,
    write_csv,
)


def empty_or_read(path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def fit_one_repeat(
    payload: dict[str, object],
    cfg: dict,
    repeat_number: int,
    nominal_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame = payload["frame"]
    features = list(payload["features"])
    a = np.asarray(payload["a"], dtype=int)
    event = np.asarray(payload["event"], dtype=int)
    observed_time = np.asarray(payload["observed_time"], dtype=float)
    frozen_e = np.asarray(payload["e"], dtype=float)
    original_fold = np.asarray(payload["fold"], dtype=int)
    rcfg = cfg["repeated_crossfit"]
    n_folds = int(rcfg["n_folds"])
    primary_g = float(rcfg["primary_g_min"])

    last_error: Exception | None = None
    for attempt in range(20):
        split_seed = nominal_seed + attempt * 100_000
        try:
            repeated_fold, stratification = make_repeated_fold_assignment(
                a, event, split_seed, n_folds
            )
            refitted_e = crossfit_propensity(
                frame,
                features,
                repeated_fold,
                "analysis_treatment",
                split_seed + 10,
            )
            G, starts, ends, censor_metrics = crossfit_censor_survival(
                frame,
                features,
                repeated_fold,
                float(LANDMARK_HORIZON),
                float(LANDMARK_INTERVAL),
                split_seed + 100,
            )
            break
        except Exception as exc:  # deterministic retry for a nuisance-invalid split
            last_error = exc
            if attempt == 19:
                raise
    else:  # pragma: no cover
        raise RuntimeError(f"Repeat {repeat_number} failed") from last_error

    estimates_rows: list[dict] = []
    score_rows: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []
    loo_rows: list[pd.DataFrame] = []

    e_tracks = {
        "frozen_stage30": frozen_e,
        "refitted_stage30_specification": refitted_e,
    }

    for g_min in [float(v) for v in rcfg["g_min_values"]]:
        y = ipcw_rmst_pseudo(
            observed_time,
            G,
            starts,
            ends,
            float(LANDMARK_HORIZON),
            g_min,
        )
        mu0_unbounded, mu1_unbounded = crossfit_arm_outcomes(
            frame,
            features,
            y,
            repeated_fold,
            split_seed + 200 + int(round(g_min * 1000)),
        )
        mu0_bounded = np.clip(mu0_unbounded, 0.0, float(LANDMARK_HORIZON))
        mu1_bounded = np.clip(mu1_unbounded, 0.0, float(LANDMARK_HORIZON))

        constraints = {
            "bounded_0_to_horizon": (mu0_bounded, mu1_bounded),
            "unbounded": (mu0_unbounded, mu1_unbounded),
        }
        for track, e in e_tracks.items():
            for constraint, (mu0, mu1) in constraints.items():
                summary, patient = effect_and_patient_components(y, a, e, mu0, mu1)
                row = {
                    "repeat": repeat_number,
                    "nominal_seed": nominal_seed,
                    "split_seed_used": split_seed,
                    "split_retry_attempt": attempt,
                    "fold_stratification": stratification,
                    "n_folds": int(len(np.unique(repeated_fold))),
                    "propensity_track": track,
                    "g_min": g_min,
                    "prediction_constraint": constraint,
                    **summary,
                    "pseudo_mean": float(np.mean(y)),
                    "pseudo_sd": float(np.std(y, ddof=1)),
                    "pseudo_p95": float(np.quantile(y, 0.95)),
                    "pseudo_p99": float(np.quantile(y, 0.99)),
                    "pseudo_max": float(np.max(y)),
                    "mu0_min_unbounded": float(np.min(mu0_unbounded)),
                    "mu0_max_unbounded": float(np.max(mu0_unbounded)),
                    "mu1_min_unbounded": float(np.min(mu1_unbounded)),
                    "mu1_max_unbounded": float(np.max(mu1_unbounded)),
                    "fraction_mu0_outside": float(
                        np.mean((mu0_unbounded < 0.0) | (mu0_unbounded > LANDMARK_HORIZON))
                    ),
                    "fraction_mu1_outside": float(
                        np.mean((mu1_unbounded < 0.0) | (mu1_unbounded > LANDMARK_HORIZON))
                    ),
                    "propensity_min": float(np.min(e)),
                    "propensity_p01": float(np.quantile(e, 0.01)),
                    "propensity_p99": float(np.quantile(e, 0.99)),
                    "propensity_max": float(np.max(e)),
                    "censor_log_loss": float(censor_metrics["censor_log_loss"]),
                    "censor_brier": float(censor_metrics["censor_brier"]),
                    "G_min_raw": float(censor_metrics["G_min"]),
                    "G_p01_raw": float(censor_metrics["G_p01"]),
                    "max_inverse_G_after_floor": float(
                        np.max(1.0 / np.clip(G, g_min, 1.0))
                    ),
                }

                if constraint == "bounded_0_to_horizon":
                    fdf, ldf = original_fold_effects(
                        y, a, e, mu0, mu1, repeated_fold
                    )
                    row.update(
                        {
                            "fold_effect_min": float(fdf["fold_effect_days"].min()),
                            "fold_effect_max": float(fdf["fold_effect_days"].max()),
                            "fold_effect_spread": float(
                                fdf["fold_effect_days"].max()
                                - fdf["fold_effect_days"].min()
                            ),
                            "fold_fraction_positive": float(
                                np.mean(fdf["fold_effect_days"] > 0)
                            ),
                            "loo_effect_min": float(
                                ldf["leave_one_fold_out_effect_days"].min()
                            ),
                            "loo_effect_max": float(
                                ldf["leave_one_fold_out_effect_days"].max()
                            ),
                            "loo_effect_spread": float(
                                ldf["leave_one_fold_out_effect_days"].max()
                                - ldf["leave_one_fold_out_effect_days"].min()
                            ),
                            "loo_fraction_positive": float(
                                np.mean(ldf["leave_one_fold_out_effect_days"] > 0)
                            ),
                        }
                    )
                    fdf.insert(0, "prediction_constraint", constraint)
                    fdf.insert(0, "g_min", g_min)
                    fdf.insert(0, "propensity_track", track)
                    fdf.insert(0, "repeat", repeat_number)
                    ldf.insert(0, "prediction_constraint", constraint)
                    ldf.insert(0, "g_min", g_min)
                    ldf.insert(0, "propensity_track", track)
                    ldf.insert(0, "repeat", repeat_number)
                    fold_rows.append(fdf)
                    loo_rows.append(ldf)

                    if np.isclose(g_min, primary_g):
                        primary = patient[
                            [
                                "h",
                                "score_numerator",
                                "normalized_contribution_days",
                                "influence",
                            ]
                        ].copy()
                        primary.insert(0, "repeated_fold", repeated_fold)
                        primary.insert(0, "original_fold", original_fold)
                        primary.insert(0, "event", event)
                        primary.insert(0, "treatment", a)
                        primary.insert(0, "local_row_index", np.arange(len(a)))
                        primary.insert(0, "g_min", g_min)
                        primary.insert(0, "propensity_track", track)
                        primary.insert(0, "split_seed_used", split_seed)
                        primary.insert(0, "repeat", repeat_number)
                        score_rows.append(primary)
                estimates_rows.append(row)

    censor_row = pd.DataFrame(
        [
            {
                "repeat": repeat_number,
                "nominal_seed": nominal_seed,
                "split_seed_used": split_seed,
                "split_retry_attempt": attempt,
                "fold_stratification": stratification,
                "n_folds": int(len(np.unique(repeated_fold))),
                "censor_log_loss": float(censor_metrics["censor_log_loss"]),
                "censor_brier": float(censor_metrics["censor_brier"]),
                "G_min_raw": float(censor_metrics["G_min"]),
                "G_p01_raw": float(censor_metrics["G_p01"]),
                "refitted_ps_min": float(np.min(refitted_e)),
                "refitted_ps_p01": float(np.quantile(refitted_e, 0.01)),
                "refitted_ps_p99": float(np.quantile(refitted_e, 0.99)),
                "refitted_ps_max": float(np.max(refitted_e)),
            }
        ]
    )
    return (
        pd.DataFrame(estimates_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.concat(fold_rows, ignore_index=True),
        pd.concat(loo_rows, ignore_index=True),
        censor_row,
        {
            "split_seed_used": split_seed,
            "split_retry_attempt": attempt,
            "fold_stratification": stratification,
        },
    )


def main() -> int:
    root = project_root()
    ensure_stage17_dirs(root)
    cfg = load_stage17_config(root)
    rcfg = cfg["repeated_crossfit"]
    tables = root / "results/tables"
    local = root / "data/derived/stage17"

    estimates_path = tables / "68_repeated_crossfit_estimates_checkpoint.csv"
    scores_path = local / "68_primary_patient_scores_LOCAL_ONLY.csv"
    folds_path = tables / "68_repeated_fold_effects_checkpoint.csv"
    loo_path = tables / "68_repeated_leave_one_fold_out_checkpoint.csv"
    censor_path = tables / "68_repeated_censoring_checkpoint.csv"
    errors_path = tables / "68_repeated_crossfit_errors.csv"
    identity_path = local / "68_checkpoint_identity.json"

    validate_or_create_checkpoint_identity(
        identity_path, checkpoint_identity(root, cfg)
    )

    estimates = empty_or_read(estimates_path)
    scores = empty_or_read(scores_path)
    fold_effects = empty_or_read(folds_path)
    loo_effects = empty_or_read(loo_path)
    censoring = empty_or_read(censor_path)
    errors = empty_or_read(
        errors_path,
        ["repeat", "nominal_seed", "error_type", "error_message", "traceback"],
    )

    completed = completed_repeat_numbers(estimates, cfg)
    payload = exact_landmark_payload()
    seeds = stable_repeat_seeds(cfg)

    print("=" * 124)
    print("STAGE 68 — PRESPECIFIED REPEATED CROSS-FITTING STABILITY AUDIT")
    print("=" * 124)
    print(f"Target repeats: {len(seeds)}")
    print(f"Already complete: {len(completed)} -> {sorted(completed)}")
    print("Primary model: arm-specific ridge bounded to [0, 730] days")
    print("Sensitivity: the same ridge predictions without bounding")
    print("Propensity tracks: frozen Stage 30 and fully refitted Stage 30 specification")
    print(f"G-min values: {rcfg['g_min_values']}")
    print("The publication bootstrap is NOT started.")

    for repeat_number, nominal_seed in enumerate(seeds, start=1):
        if repeat_number in completed:
            print(f"Repeat {repeat_number:02d}/{len(seeds)} already complete; skipping.")
            continue
        print("-" * 124)
        print(f"RUNNING REPEAT {repeat_number:02d}/{len(seeds)}; nominal seed={nominal_seed}")
        try:
            (
                new_estimates,
                new_scores,
                new_folds,
                new_loo,
                new_censor,
                split_info,
            ) = fit_one_repeat(payload, cfg, repeat_number, nominal_seed)
        except Exception as exc:
            error_row = pd.DataFrame(
                [
                    {
                        "repeat": repeat_number,
                        "nominal_seed": nominal_seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ]
            )
            errors = append_replace_repeat(errors, error_row, repeat_number)
            write_csv(errors, errors_path)
            print(error_row[["repeat", "nominal_seed", "error_type", "error_message"]].to_string(index=False))
            raise

        estimates = append_replace_repeat(estimates, new_estimates, repeat_number)
        scores = append_replace_repeat(scores, new_scores, repeat_number)
        fold_effects = append_replace_repeat(fold_effects, new_folds, repeat_number)
        loo_effects = append_replace_repeat(loo_effects, new_loo, repeat_number)
        censoring = append_replace_repeat(censoring, new_censor, repeat_number)
        if not errors.empty:
            errors = errors[pd.to_numeric(errors["repeat"], errors="coerce") != repeat_number]

        write_csv(estimates.sort_values(["repeat", "propensity_track", "g_min", "prediction_constraint"]), estimates_path)
        write_csv(scores.sort_values(["repeat", "propensity_track", "local_row_index"]), scores_path)
        write_csv(fold_effects.sort_values(["repeat", "propensity_track", "g_min", "fold"]), folds_path)
        write_csv(loo_effects.sort_values(["repeat", "propensity_track", "g_min", "omitted_fold"]), loo_path)
        write_csv(censoring.sort_values("repeat"), censor_path)
        write_csv(errors, errors_path)

        primary = new_estimates[
            (new_estimates["prediction_constraint"] == "bounded_0_to_horizon")
            & np.isclose(new_estimates["g_min"], float(rcfg["primary_g_min"]))
        ][
            [
                "repeat",
                "propensity_track",
                "estimate_days",
                "if_se_days",
                "if_ci_low_days",
                "if_ci_high_days",
                "direct_ato_ipw_effect_days",
                "pseudo_p99",
                "pseudo_max",
                "loo_effect_spread",
            ]
        ]
        g_table = new_estimates[
            new_estimates["prediction_constraint"] == "bounded_0_to_horizon"
        ].pivot_table(
            index="g_min",
            columns="propensity_track",
            values="estimate_days",
            aggfunc="first",
        ).reset_index()
        print(
            f"Split used: seed={split_info['split_seed_used']}; "
            f"retry={split_info['split_retry_attempt']}; "
            f"stratification={split_info['fold_stratification']}"
        )
        print("Primary G-min=0.10 bounded results")
        print(dataframe_console(primary))
        print("G-min sensitivity for this repeat")
        print(dataframe_console(g_table))
        print(
            f"Checkpoint saved: completed repeats now "
            f"{len(completed_repeat_numbers(estimates, cfg))}/{len(seeds)}"
        )

    completed = completed_repeat_numbers(estimates, cfg)
    if len(completed) != len(seeds):
        raise RuntimeError(
            f"Only {len(completed)}/{len(seeds)} repeated cross-fits are complete."
        )

    primary_all = estimates[
        (estimates["prediction_constraint"] == "bounded_0_to_horizon")
        & np.isclose(estimates["g_min"], float(rcfg["primary_g_min"]))
    ][
        [
            "repeat",
            "nominal_seed",
            "split_seed_used",
            "propensity_track",
            "estimate_days",
            "if_se_days",
            "if_ci_low_days",
            "if_ci_high_days",
            "direct_ato_ipw_effect_days",
            "pseudo_mean",
            "pseudo_sd",
            "pseudo_p99",
            "pseudo_max",
            "fold_effect_spread",
            "loo_effect_spread",
            "censor_log_loss",
            "censor_brier",
            "G_min_raw",
            "G_p01_raw",
        ]
    ].sort_values(["repeat", "propensity_track"])

    print("=" * 124)
    print("STAGE 68 COMPLETED — ALL PRIMARY REPEAT RESULTS")
    print("=" * 124)
    print(dataframe_console(primary_all))
    print("\nCheckpoint and local-score files")
    for path in (estimates_path, scores_path, folds_path, loo_path, censor_path, errors_path):
        print(f"- {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
