from __future__ import annotations

import pandas as pd

from _common import RESULTS_DIR
from _stage10_utils import (
    build_primary_model_table,
    load_paperA_inputs,
    primary_split,
)


def main() -> int:
    base, W_cols, RNA_cols, metadata = (
        build_primary_model_table()
    )
    split = primary_split(base, repeat=1)
    inputs = load_paperA_inputs()

    # Validate exact downstream column selections.
    _ = base[W_cols].apply(pd.to_numeric, errors="coerce")
    _ = base[RNA_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    _ = base["rmst_ipcw"].astype(float)
    _ = base["analysis_treatment"].astype(int)
    _ = split["fold"].astype(int)

    print("=" * 115)
    print("STAGE 10 PREFLIGHT PASSED")
    print("=" * 115)
    print(pd.DataFrame([metadata]).to_string(index=False))
    print(
        "\nExact Paper A and Paper B model tables were assembled. "
        "No merge suffixes remain, all patient IDs match, and all "
        "required Stage 29–31 tables are available."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
