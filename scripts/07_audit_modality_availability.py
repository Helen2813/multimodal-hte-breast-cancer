from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


MODALITY_PREFIXES = {
    "RNA": ("RNA_",),
    "CNV": ("CNV_",),
    "Mutation": ("MUT_", "MUTATION_"),
    "Methylation": ("METH_", "METHYLATION_"),
    "miRNA": ("MIRNA_", "miRNA_"),
    "Protein": ("PROT_", "PROTEIN_"),
}

INDICATOR_TOKENS = (
    "missing",
    "available",
    "availability",
    "observed",
    "present",
    "indicator",
    "has_",
)


def modality_columns(columns: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [
        col for col in columns
        if any(str(col).startswith(prefix) for prefix in prefixes)
    ]


def is_indicator_name(column: str) -> bool:
    low = column.lower()
    return any(token in low for token in INDICATOR_TOKENS)


def infer_indicator_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    named = [c for c in columns if is_indicator_name(c)]
    if named:
        return named

    # Fallback: identify binary columns whose values are strongly associated with
    # all biological features being zero. We do not automatically remove them;
    # this only flags likely availability indicators for review.
    candidates = []
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    for col in columns:
        values = numeric[col].dropna()
        unique = set(values.unique().tolist())
        if len(unique) <= 2 and unique.issubset({0.0, 1.0}):
            candidates.append(col)
    return candidates


def patient_availability(
    df: pd.DataFrame,
    biological_cols: list[str],
    indicator_cols: list[str],
) -> pd.Series:
    if indicator_cols:
        # Prefer explicitly named missing/available indicators.
        named = [c for c in indicator_cols if is_indicator_name(c)]
        if named:
            out = pd.Series(True, index=df.index)
            for col in named:
                values = pd.to_numeric(df[col], errors="coerce")
                low = col.lower()
                if "missing" in low:
                    out &= values.fillna(1).eq(0)
                else:
                    out &= values.fillna(0).gt(0.5)
            return out

    # Legacy outer matrices used zero-filled molecular features plus an indicator.
    # When the indicator cannot be identified by name, report a conservative
    # feature-information flag: at least one nonzero biological value.
    if biological_cols:
        x = df[biological_cols].apply(pd.to_numeric, errors="coerce")
        return x.fillna(0).abs().sum(axis=1).gt(0)
    return pd.Series(False, index=df.index)


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    table_dir = RESULTS_DIR / "tables"

    cohort_paths = {
        "outer_hormone_hrpos_her2neg": cohort_dir / "outer_hormone_hrpos_her2neg.csv",
        "outer_chemo_tnbc": cohort_dir / "outer_chemo_tnbc.csv",
    }
    missing = [str(p) for p in cohort_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required outer cohorts are missing. Run stages 00–06 first.\n"
            + "\n".join(missing)
        )

    summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    availability_frames: list[pd.DataFrame] = []

    for cohort_name, path in cohort_paths.items():
        print(f"Auditing modality availability: {cohort_name}")
        df = read_table(path)
        all_columns = list(map(str, df.columns))

        availability = pd.DataFrame(
            {"patient_id_normalized": df["patient_id_normalized"]}
        )

        for modality, prefixes in MODALITY_PREFIXES.items():
            cols = modality_columns(all_columns, prefixes)
            indicators = infer_indicator_columns(df, cols)
            named_indicators = [c for c in indicators if is_indicator_name(c)]

            # Only explicitly named indicator columns are excluded automatically.
            biological = [c for c in cols if c not in named_indicators]
            x = (
                df[biological].apply(pd.to_numeric, errors="coerce")
                if biological
                else pd.DataFrame(index=df.index)
            )

            observed = patient_availability(df, biological, named_indicators)
            availability[f"available_{modality}"] = observed.astype(int)

            constant = []
            all_zero = []
            near_zero = []
            for col in biological:
                values = pd.to_numeric(df[col], errors="coerce")
                n_unique = int(values.nunique(dropna=True))
                zero_fraction = float(values.fillna(0).eq(0).mean())
                if n_unique <= 1:
                    constant.append(col)
                if zero_fraction >= 0.999:
                    all_zero.append(col)
                if zero_fraction >= 0.95:
                    near_zero.append(col)
                feature_rows.append(
                    {
                        "cohort": cohort_name,
                        "modality": modality,
                        "feature": col,
                        "n_unique": n_unique,
                        "missing_fraction": float(values.isna().mean()),
                        "zero_fraction": zero_fraction,
                        "variance": float(values.var(ddof=1))
                        if values.notna().sum() > 1
                        else np.nan,
                        "flag_constant": int(n_unique <= 1),
                        "flag_all_zero": int(zero_fraction >= 0.999),
                        "flag_near_zero": int(zero_fraction >= 0.95),
                    }
                )

            summary_rows.append(
                {
                    "cohort": cohort_name,
                    "modality": modality,
                    "n_patients": int(len(df)),
                    "n_prefixed_columns": len(cols),
                    "n_named_indicator_columns": len(named_indicators),
                    "named_indicator_columns": "|".join(named_indicators),
                    "n_biological_columns": len(biological),
                    "n_constant_columns": len(constant),
                    "n_all_zero_columns": len(all_zero),
                    "n_near_zero_columns": len(near_zero),
                    "available_patients": int(observed.sum()),
                    "available_fraction": float(observed.mean()),
                }
            )

        availability["cohort"] = cohort_name
        availability_frames.append(availability)

    summary = pd.DataFrame(summary_rows)
    features = pd.DataFrame(feature_rows)
    availability = pd.concat(availability_frames, ignore_index=True)

    summary.to_csv(table_dir / "07_modality_availability_summary.csv", index=False)
    features.to_csv(table_dir / "07_modality_feature_audit.csv", index=False)
    availability.to_csv(
        DERIVED_DIR / "manifests" / "07_patient_modality_availability.csv",
        index=False,
    )

    print("\nModality availability summary:")
    print(
        summary[
            [
                "cohort",
                "modality",
                "n_prefixed_columns",
                "n_named_indicator_columns",
                "n_biological_columns",
                "available_patients",
                "available_fraction",
                "n_constant_columns",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
