from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


def main() -> int:
    ensure_dirs()
    verified_dir = DERIVED_DIR / "verified_sources"
    cohort_dir = DERIVED_DIR / "verified_cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    receptor_path = verified_dir / "16_recovered_observed_receptor_labels.csv"
    treatment_path = verified_dir / "17_verified_treatment_flags.csv"
    if not receptor_path.exists():
        raise FileNotFoundError(receptor_path)
    if not treatment_path.exists():
        raise FileNotFoundError(
            f"{treatment_path}\nRun Stage 17 after copying original clinical.tsv."
        )

    receptors = read_table(receptor_path)
    treatments = read_table(treatment_path)
    source = receptors.merge(
        treatments,
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )

    summary_rows = []
    definitions = {
        "hormone_hrpos_her2neg": {
            "mask": (
                (
                    source["ER_observed_binary"].eq(1)
                    | source["PR_observed_binary"].eq(1)
                )
                & source["HER2_observed_binary"].eq(0)
            ),
            "treatment": "T_hormone_verified",
            "description": (
                "Observed ER-positive or PR-positive, with observed HER2-negative"
            ),
        },
        "chemo_tnbc": {
            "mask": (
                source["ER_observed_binary"].eq(0)
                & source["PR_observed_binary"].eq(0)
                & source["HER2_observed_binary"].eq(0)
            ),
            "treatment": "T_chemo_verified",
            "description": (
                "Observed ER-negative, PR-negative, and HER2-negative"
            ),
        },
    }

    for representation in ("outer", "complete_case"):
        master_path = DERIVED_DIR / "cohorts" / f"master_{representation}.csv"
        if not master_path.exists():
            raise FileNotFoundError(master_path)
        master = read_table(master_path)
        merged = master.merge(
            source,
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_verified_source"),
        )

        for cohort_name, definition in definitions.items():
            cohort = merged.loc[definition["mask"]].copy()
            treatment_col = definition["treatment"]
            cohort["analysis_treatment"] = pd.to_numeric(
                cohort[treatment_col], errors="raise"
            ).astype(int)
            cohort["analysis_event"] = pd.to_numeric(
                cohort["OS"], errors="raise"
            ).astype(int)
            cohort["analysis_time"] = pd.to_numeric(
                cohort["OS.time"], errors="coerce"
            )
            cohort["eligibility_receptor_source"] = (
                "exact_standardized_observed_modes_only"
            )
            cohort["treatment_source"] = "original_clinical_tsv_verified"

            output = (
                cohort_dir / f"{representation}_{cohort_name}_verified.csv"
            )
            cohort.to_csv(output, index=False)

            summary_rows.append(
                {
                    "representation": representation,
                    "cohort": cohort_name,
                    "description": definition["description"],
                    "n": len(cohort),
                    "treated": int(cohort["analysis_treatment"].sum()),
                    "control": int(
                        len(cohort) - cohort["analysis_treatment"].sum()
                    ),
                    "events": int(cohort["analysis_event"].sum()),
                    "event_rate": float(cohort["analysis_event"].mean())
                    if len(cohort)
                    else np.nan,
                    "output": str(output),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        table_dir / "18_verified_cohort_summary.csv",
        index=False,
    )

    legacy_path = table_dir / "03_cohort_summary.csv"
    if legacy_path.exists():
        legacy = read_table(legacy_path)[
            ["representation", "cohort", "n", "treated", "control", "events"]
        ].rename(
            columns={
                "n": "legacy_n",
                "treated": "legacy_treated",
                "control": "legacy_control",
                "events": "legacy_events",
            }
        )
        comparison = summary.merge(
            legacy, on=["representation", "cohort"], how="left"
        )
        for metric in ("n", "treated", "control", "events"):
            comparison[f"change_{metric}"] = (
                comparison[metric] - comparison[f"legacy_{metric}"]
            )
        comparison.to_csv(
            table_dir / "18_verified_vs_legacy_cohorts.csv",
            index=False,
        )

    print("\nVerified cohort summary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
