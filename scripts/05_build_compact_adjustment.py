from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table
from _compact_adjustment import build_compact_adjustment, manifest_to_frame


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    compact_dir = DERIVED_DIR / "compact_adjustment"
    compact_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    cohort_files = sorted(cohort_dir.glob("*_hormone_hrpos_her2neg.csv")) + sorted(
        cohort_dir.glob("*_chemo_tnbc.csv")
    )
    if not cohort_files:
        raise FileNotFoundError(
            "No treatment-specific cohort files found. Run scripts 00–04 first."
        )

    summary_rows = []
    for path in cohort_files:
        label = path.stem
        print(f"Building compact adjustment matrix: {label}")
        df = read_table(path)
        if "patient_id_normalized" not in df.columns:
            raise ValueError(f"{path} has no patient_id_normalized column.")
        for required in ("analysis_treatment", "analysis_event"):
            if required not in df.columns:
                raise ValueError(f"{path} has no {required} column.")

        W, manifest = build_compact_adjustment(df)
        if W.shape[1] < 3:
            raise ValueError(
                f"{label}: only {W.shape[1]} compact covariates were created. "
                "Inspect clinical column names before continuing."
            )

        metadata_cols = [
            c for c in (
                "patient_id_normalized",
                "analysis_treatment",
                "analysis_event",
                "analysis_time",
                "ER_status",
                "PR_status",
                "HER2_status",
            )
            if c in df.columns
        ]
        compact = pd.concat(
            [df[metadata_cols].reset_index(drop=True), W.reset_index(drop=True)],
            axis=1,
        )

        compact_path = compact_dir / f"{label}_compact.csv"
        compact.to_csv(compact_path, index=False)

        manifest_df = manifest_to_frame(manifest)
        manifest_path = compact_dir / f"{label}_compact_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        summary_rows.append(
            {
                "cohort": label,
                "n": int(len(compact)),
                "treated": int(pd.to_numeric(compact["analysis_treatment"]).sum()),
                "events": int(pd.to_numeric(compact["analysis_event"]).sum()),
                "n_compact_covariates": int(W.shape[1]),
                "covariates": "|".join(W.columns),
                "output": str(compact_path),
                "manifest": str(manifest_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "05_compact_adjustment_summary.csv", index=False)
    print("\nCompact adjustment summary:")
    print(summary[["cohort", "n", "treated", "events", "n_compact_covariates"]].to_string(index=False))
    print(f"\nOutputs saved to: {compact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
