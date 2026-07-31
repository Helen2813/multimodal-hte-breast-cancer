from __future__ import annotations

import pandas as pd
from _common import RESULTS_DIR, ensure_dirs
from _stage12_utils import BASE_SEED, assemble_landmark_data, assemble_source_data, bootstrap_folds, ccw_estimate, landmark_estimate


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    landmark, features, assignment, metadata = assemble_landmark_data()
    fold = assignment["fold"].astype(int).to_numpy()
    frozen_stage30_ps = pd.to_numeric(
        landmark["propensity_score_oof_stage30"], errors="raise"
    ).to_numpy(float)
    landmark_result = landmark_estimate(
        landmark, features, fold, seed=BASE_SEED,
        propensity_scores=frozen_stage30_ps,
    )
    difference = landmark_result["estimate_days"] - metadata["candidate_effect_days"]
    if abs(difference) > 0.05:
        raise RuntimeError(
            "Stage 12 estimator does not reproduce the candidate result: "
            f"{landmark_result['estimate_days']:.6f} versus {metadata['candidate_effect_days']:.6f}; "
            f"difference={difference:.6f} days."
        )
    source, source_features, source_metadata = assemble_source_data()
    source = source.copy()
    source["original_patient_id"] = source["patient_id_normalized"].astype(str)
    ccw_fold = bootstrap_folds(source, "early_initiation", "analysis_event", seed=BASE_SEED + 700)
    ccw_result = ccw_estimate(source, source_features, ccw_fold, seed=BASE_SEED + 800)
    landmark_row = {"analysis": "landmark_primary", **metadata, **landmark_result, "candidate_difference_days": difference, "replication_status": "EXACT_REPLICATION_PASSED", "replication_propensity_source": "saved_stage30_oof_propensity_scores"}
    ccw_row = {"analysis": "ccw_sensitivity", **source_metadata, **ccw_result, "replication_status": "CCW_POINT_ESTIMATE_COMPLETED"}
    pd.DataFrame([landmark_row]).to_csv(table_dir / "41_landmark_replication_check.csv", index=False)
    pd.DataFrame([ccw_row]).to_csv(table_dir / "41_ccw_point_estimate.csv", index=False)
    print("=" * 118)
    print("STAGE 41 — ESTIMATOR REPLICATION AND CCW POINT ESTIMATE")
    print("=" * 118)
    print("\nLandmark replication")
    print(pd.DataFrame([landmark_row]).to_string(index=False))
    print("\nCCW sensitivity point estimate")
    print(pd.DataFrame([ccw_row]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
