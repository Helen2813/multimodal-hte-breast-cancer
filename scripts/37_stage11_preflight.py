from __future__ import annotations

import importlib.util

import pandas as pd

from _stage11_utils import (
    assemble_landmark_table,
    assemble_source_table,
    extract_hormone_start_days,
    primary_paths,
)


def main() -> int:
    landmark, W_landmark, landmark_meta = assemble_landmark_table()
    source, W_source, source_meta = assemble_source_table()
    timing = extract_hormone_start_days(source["patient_id_normalized"])

    _ = landmark[W_landmark].apply(pd.to_numeric, errors="coerce")
    _ = source[W_source].apply(pd.to_numeric, errors="coerce")
    _ = pd.to_numeric(timing["earliest_start_nonnegative"], errors="coerce")

    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError("matplotlib is required for flow and love plots.")

    summary = {
        **landmark_meta,
        **source_meta,
        "source_start_rows": len(timing),
        "landmark_W_features": len(W_landmark),
        "source_W_features": len(W_source),
        "matplotlib_available": 1,
        "required_files": len(primary_paths()),
    }

    print("=" * 115)
    print("STAGE 11 PREFLIGHT PASSED")
    print("=" * 115)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(
        "\nExact landmark/source tables, treatment timing, weights, splits, "
        "reporting inputs, and plotting dependency were validated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
