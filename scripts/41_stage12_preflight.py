from __future__ import annotations

import importlib.util
import pandas as pd
from _stage12_utils import assemble_landmark_data, assemble_source_data, project_paths


def main() -> int:
    landmark, W_landmark, assignment, lm_meta = assemble_landmark_data()
    source, W_source, source_meta = assemble_source_data()
    _ = landmark[W_landmark].apply(pd.to_numeric, errors="coerce")
    frozen_ps = pd.to_numeric(landmark["propensity_score_oof_stage30"], errors="raise")
    if not frozen_ps.between(0, 1).all():
        raise ValueError("Saved Stage 30 propensity scores are outside [0, 1].")
    _ = source[W_source].apply(pd.to_numeric, errors="coerce")
    _ = assignment["fold"].astype(int)
    deps = {name: int(importlib.util.find_spec(name) is not None) for name in ("numpy", "pandas", "sklearn", "scipy")}
    if not all(deps.values()):
        raise ImportError(f"Missing required packages: {deps}")
    summary = {**lm_meta, **source_meta, "landmark_features": len(W_landmark), "source_features": len(W_source), **{f"dependency_{k}": v for k, v in deps.items()}}
    print("=" * 118)
    print("STAGE 12 PREFLIGHT PASSED")
    print("=" * 118)
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nExact landmark and diagnosis-time CCW inputs were assembled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
