from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    DERIVED_DIR,
    RESULTS_DIR,
    ensure_dirs,
    find_ite_file,
    detect_id_column,
    normalize_patient_id,
    read_table,
)


RECEPTORS = ("ER_status", "PR_status", "HER2_status")


def recover_two_modal_binary(series: pd.Series, label: str) -> tuple[pd.Series, dict[str, object]]:
    """
    Recover only the exactly observed binary states after standardization.

    In the legacy preprocessing, observed 0/1 receptor values became two highly
    repeated standardized modes. MICE-imputed observations became non-modal
    continuous values. We map the lower repeated mode to 0, the higher repeated
    mode to 1, and leave every non-modal value missing.

    This does not invent labels for imputed cases.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    counts = numeric.value_counts(dropna=True)
    if len(counts) < 2:
        raise ValueError(f"{label}: fewer than two numeric levels.")

    top = counts.head(2)
    modal_values = sorted([float(x) for x in top.index.tolist()])
    lower, upper = modal_values
    separation = upper - lower
    if separation <= 0.25:
        raise ValueError(
            f"{label}: top modes are insufficiently separated: {lower}, {upper}"
        )

    combined_fraction = float(top.sum() / numeric.notna().sum())
    if combined_fraction < 0.80:
        raise ValueError(
            f"{label}: top two modes cover only {combined_fraction:.3f} of values."
        )

    tolerance = max(1e-10, separation * 1e-9)
    is_lower = np.isclose(numeric, lower, atol=tolerance, rtol=1e-10)
    is_upper = np.isclose(numeric, upper, atol=tolerance, rtol=1e-10)

    recovered = pd.Series(pd.NA, index=series.index, dtype="Int64")
    recovered.loc[is_lower] = 0
    recovered.loc[is_upper] = 1

    old_threshold = (numeric.fillna(0) > 0.5).astype(int)
    observed_mask = recovered.notna()
    agreement = float(
        (old_threshold.loc[observed_mask] == recovered.loc[observed_mask].astype(int)).mean()
    )

    report = {
        "receptor": label,
        "lower_mode_negative": lower,
        "upper_mode_positive": upper,
        "lower_mode_count": int((recovered == 0).sum()),
        "upper_mode_count": int((recovered == 1).sum()),
        "observed_recovered": int(recovered.notna().sum()),
        "nonmodal_imputed_or_unknown": int(recovered.isna().sum()),
        "observed_fraction": float(recovered.notna().mean()),
        "top_two_mode_fraction_among_numeric": combined_fraction,
        "agreement_with_legacy_threshold_on_observed": agreement,
    }
    return recovered, report


def main() -> int:
    ensure_dirs()
    verified_dir = DERIVED_DIR / "verified_sources"
    verified_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    ite_path = find_ite_file()
    if ite_path is None:
        raise FileNotFoundError("ITE ready dataset was not found.")
    ite = read_table(ite_path)
    id_col = detect_id_column(ite)
    if id_col is None:
        raise ValueError("Patient ID column not found in ITE table.")

    output = pd.DataFrame(
        {"patient_id_normalized": ite[id_col].map(normalize_patient_id)}
    )
    reports = []

    for receptor in RECEPTORS:
        if receptor not in ite.columns:
            raise ValueError(f"Missing receptor column: {receptor}")
        recovered, report = recover_two_modal_binary(ite[receptor], receptor)
        short = receptor.replace("_status", "")
        output[f"{short}_observed_binary"] = recovered
        output[f"{short}_label_source"] = np.where(
            recovered.notna(),
            "exact_standardized_observed_mode",
            "nonmodal_imputed_or_unknown",
        )
        reports.append(report)

    if output["patient_id_normalized"].duplicated().any():
        raise ValueError("Duplicate patient IDs in recovered receptor table.")

    output.to_csv(
        verified_dir / "16_recovered_observed_receptor_labels.csv",
        index=False,
    )
    report_df = pd.DataFrame(reports)
    report_df.to_csv(
        table_dir / "16_receptor_mode_recovery_summary.csv",
        index=False,
    )

    # Cross-tab of missingness across the three receptor fields.
    receptor_cols = [
        "ER_observed_binary",
        "PR_observed_binary",
        "HER2_observed_binary",
    ]
    missing_patterns = (
        output[receptor_cols]
        .isna()
        .astype(int)
        .rename(columns=lambda c: c.replace("_observed_binary", "_missing"))
        .value_counts()
        .reset_index(name="patients")
    )
    missing_patterns.to_csv(
        table_dir / "16_receptor_missingness_patterns.csv",
        index=False,
    )

    print("\nRecovered observed receptor labels:")
    print(report_df.to_string(index=False))
    print(
        "\nNon-modal MICE/imputed values remain missing and will not be thresholded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
