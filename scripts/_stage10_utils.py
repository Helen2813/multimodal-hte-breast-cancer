from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import DERIVED_DIR, RESULTS_DIR, read_table


PRIMARY_COHORT = "outer_hormone_hrpos_her2neg"
PRIMARY_LANDMARK = 180
PRIMARY_HORIZON = 730
PRIMARY_G_MIN = 0.10


def resolve_primary_pseudooutcome() -> Path:
    key = f"{PRIMARY_COHORT}_landmark{PRIMARY_LANDMARK}"
    pseudo_dir = DERIVED_DIR / "landmark_pseudooutcomes"
    candidates = sorted(
        pseudo_dir.glob(
            f"{key}_{PRIMARY_HORIZON}d_classical_g*.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No primary pseudo-outcome file found under {pseudo_dir}"
        )

    mapping = {
        "005": 0.05,
        "05": 0.05,
        "050": 0.05,
        "01": 0.10,
        "010": 0.10,
        "10": 0.10,
        "100": 0.10,
    }

    parsed = []
    for path in candidates:
        token = path.stem.rsplit("_g", 1)[-1]
        value = mapping.get(token)
        if value is None:
            try:
                value = float(token)
                if value > 1:
                    value /= 100.0
            except ValueError:
                continue
        parsed.append((path, float(value)))

    if not parsed:
        raise ValueError(
            f"Could not parse G-min suffixes: {[p.name for p in candidates]}"
        )

    path, value = min(
        parsed, key=lambda item: abs(item[1] - PRIMARY_G_MIN)
    )
    return path


def build_primary_model_table() -> tuple[
    pd.DataFrame, list[str], list[str], dict[str, object]
]:
    key = f"{PRIMARY_COHORT}_landmark{PRIMARY_LANDMARK}"
    cohort_path = (
        DERIVED_DIR / "landmark_cohorts" / f"{key}.csv"
    )
    compact_path = (
        DERIVED_DIR / "landmark_compact" / f"{key}_compact.csv"
    )
    split_path = (
        DERIVED_DIR / "landmark_splits" / f"{key}_splits.csv"
    )
    pseudo_path = resolve_primary_pseudooutcome()

    required_paths = {
        "cohort": cohort_path,
        "compact": compact_path,
        "splits": split_path,
        "pseudo": pseudo_path,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Stage 10 preflight missing files:\n" + "\n".join(missing)
        )

    cohort = read_table(cohort_path)
    compact = read_table(compact_path)
    splits = read_table(split_path)
    pseudo = read_table(pseudo_path)

    id_col = "patient_id_normalized"
    for name, frame in (
        ("cohort", cohort),
        ("compact", compact),
        ("splits", splits),
        ("pseudo", pseudo),
    ):
        if id_col not in frame.columns:
            raise ValueError(f"{name} is missing {id_col}")

    W_cols = [c for c in compact.columns if c.startswith("W_")]
    for col in ("diagnosis_year", "diagnosis_year_missing"):
        if col in compact.columns:
            W_cols.append(col)
    W_cols = list(dict.fromkeys(W_cols))
    if not W_cols:
        raise ValueError("No compact adjustment features found.")

    cohort_side = cohort.drop(
        columns=[c for c in W_cols if c in cohort.columns],
        errors="ignore",
    )
    compact_side = compact[[id_col] + W_cols].copy()
    pseudo_side = pseudo[[id_col, "rmst_ipcw"]].copy()

    base = (
        cohort_side.merge(
            compact_side,
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pseudo_side,
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
    )

    suffix_cols = [
        c for c in base.columns if c.endswith("_x") or c.endswith("_y")
    ]
    if suffix_cols:
        raise ValueError(
            f"Unexpected merge suffix columns: {suffix_cols}"
        )

    RNA_cols = [
        c
        for c in base.columns
        if c.startswith("RNA_")
        and not any(
            token in c.lower()
            for token in ("missing", "available", "indicator")
        )
    ]
    if not RNA_cols:
        raise ValueError("No biological RNA features found.")

    required_analysis = {
        "analysis_treatment",
        "analysis_event",
        "analysis_time",
        "rmst_ipcw",
    }
    missing_analysis = required_analysis - set(base.columns)
    if missing_analysis:
        raise ValueError(
            f"Missing analysis columns: {sorted(missing_analysis)}"
        )

    reference_ids = set(base[id_col].astype(str))
    split_ids = set(splits[id_col].astype(str))
    if reference_ids != split_ids:
        raise ValueError(
            "Patient IDs differ between assembled table and split file."
        )

    # Exact dry-run used by downstream code.
    _ = base[W_cols].apply(pd.to_numeric, errors="coerce")
    _ = base[RNA_cols].apply(pd.to_numeric, errors="coerce")
    _ = pd.to_numeric(
        base["analysis_treatment"], errors="raise"
    ).astype(int)
    _ = pd.to_numeric(base["rmst_ipcw"], errors="raise")

    metadata = {
        "cohort": PRIMARY_COHORT,
        "landmark_day": PRIMARY_LANDMARK,
        "post_landmark_horizon_days": PRIMARY_HORIZON,
        "pseudooutcome_path": str(pseudo_path),
        "n": len(base),
        "treated": int(
            pd.to_numeric(
                base["analysis_treatment"], errors="raise"
            ).sum()
        ),
        "controls": int(
            (
                1
                - pd.to_numeric(
                    base["analysis_treatment"], errors="raise"
                )
            ).sum()
        ),
        "events": int(
            pd.to_numeric(
                base["analysis_event"], errors="raise"
            ).sum()
        ),
        "compact_features": len(W_cols),
        "rna_features": len(RNA_cols),
        "split_repeats": int(splits["repeat"].nunique()),
        "split_folds": int(splits["fold"].nunique()),
        "merge_suffix_columns": 0,
    }
    return base, W_cols, RNA_cols, metadata


def primary_split(
    base: pd.DataFrame,
    repeat: int = 1,
) -> pd.DataFrame:
    key = f"{PRIMARY_COHORT}_landmark{PRIMARY_LANDMARK}"
    splits = read_table(
        DERIVED_DIR / "landmark_splits" / f"{key}_splits.csv"
    )
    assignment = splits[splits["repeat"] == repeat][
        ["patient_id_normalized", "fold"]
    ]
    out = base[["patient_id_normalized"]].merge(
        assignment,
        on="patient_id_normalized",
        how="left",
        validate="one_to_one",
    )
    if out["fold"].isna().any():
        raise ValueError("Missing fold assignments.")
    return out


def load_paperA_inputs() -> dict[str, pd.DataFrame]:
    table_dir = RESULTS_DIR / "tables"
    paths = {
        "landmark_summary": table_dir / "29_landmark_cohort_summary.csv",
        "balance": table_dir / "30_landmark_balance_summary.csv",
        "gate": table_dir / "31_landmark_paperA_gate.csv",
        "results": table_dir / "31_landmark_aipw_results.csv",
        "skipped": table_dir / "31_skipped_not_ready_designs.csv",
    }
    frames = {}
    for name, path in paths.items():
        if not path.exists():
            if name == "skipped":
                frames[name] = pd.DataFrame()
                continue
            raise FileNotFoundError(path)
        frames[name] = read_table(path)
    return frames
