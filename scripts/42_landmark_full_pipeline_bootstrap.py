from __future__ import annotations

import os, traceback
import numpy as np
import pandas as pd
from _common import RESULTS_DIR, ensure_dirs
from _stage12_utils import BASE_SEED, assemble_landmark_data, bootstrap_folds, checkpoint_config, landmark_estimate, make_bootstrap_sample, percentile_interval, validate_or_write_config

TARGET_REPS = int(os.environ.get("STAGE12_LANDMARK_REPS", "300"))


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    checkpoint = table_dir / "42_landmark_bootstrap_CHECKPOINT.csv"
    errors_path = table_dir / "42_landmark_bootstrap_errors.csv"
    config_path = table_dir / "42_landmark_bootstrap_config.json"
    base, features, _, metadata = assemble_landmark_data()
    config = checkpoint_config("landmark_full_pipeline_bootstrap", TARGET_REPS, {"cohort_sha256": metadata["cohort_sha256"], "compact_sha256": metadata["compact_sha256"], "weights_sha256": metadata["weights_sha256"]})
    validate_or_write_config(config_path, config)
    completed = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    errors = pd.read_csv(errors_path) if errors_path.exists() else pd.DataFrame()
    completed_reps = set(completed["replicate"].astype(int)) if not completed.empty else set()
    rows = completed.to_dict("records") if not completed.empty else []
    error_rows = errors.to_dict("records") if not errors.empty else []
    print("=" * 118)
    print("STAGE 42 — FULL-PIPELINE LANDMARK BOOTSTRAP")
    print("=" * 118)
    print(f"Target repetitions={TARGET_REPS}; already completed={len(completed_reps)}")
    for replicate in range(TARGET_REPS):
        if replicate in completed_reps:
            continue
        seed = BASE_SEED + 10000 + replicate
        try:
            sample = make_bootstrap_sample(base, seed)
            fold = bootstrap_folds(sample, "analysis_treatment", "analysis_event", seed=seed + 1)
            result = landmark_estimate(sample, features, fold, seed=seed + 2)
            rows.append({"replicate": replicate, "seed": seed, "n": len(sample), "treated": int(sample["analysis_treatment"].sum()), "control": int((1 - sample["analysis_treatment"]).sum()), "events": int(sample["analysis_event"].sum()), **result, "status": "OK"})
            completed_reps.add(replicate)
        except Exception as exc:
            error_rows.append({"replicate": replicate, "seed": seed, "error_type": type(exc).__name__, "error_message": str(exc), "traceback_tail": traceback.format_exc()[-1000:]})
        pd.DataFrame(rows).sort_values("replicate").to_csv(checkpoint, index=False)
        pd.DataFrame(error_rows).to_csv(errors_path, index=False)
        print(f"Checkpoint: attempted={replicate + 1}, successful={len(completed_reps)}, errors={len(error_rows)}")
    results = pd.DataFrame(rows).sort_values("replicate")
    successful = results[results["replicate"].astype(int) < TARGET_REPS]
    success_rate = len(successful) / TARGET_REPS
    if success_rate < 0.90:
        raise RuntimeError(f"Landmark bootstrap success rate {success_rate:.3f} is below 0.90.")
    estimates = successful["estimate_days"].to_numpy(float)
    ci_low, ci_high = percentile_interval(estimates)
    summary = pd.DataFrame([{
        "target_reps": TARGET_REPS,
        "successful_reps": len(successful),
        "success_rate": success_rate,
        "point_estimate_days": metadata["candidate_effect_days"],
        "bootstrap_mean_days": float(np.mean(estimates)),
        "bootstrap_sd_days": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else np.nan,
        "percentile_ci_low_days": ci_low,
        "percentile_ci_high_days": ci_high,
        "fraction_positive": float(np.mean(estimates > 0)),
        "median_days": float(np.median(estimates)),
        "status": "PILOT_COMPLETE" if TARGET_REPS < 100 else "FINAL_BOOTSTRAP_COMPLETE",
    }])
    successful.to_csv(table_dir / "42_landmark_bootstrap_results.csv", index=False)
    summary.to_csv(table_dir / "42_landmark_bootstrap_summary.csv", index=False)
    print("\nLandmark bootstrap summary")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
