from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


SEEDS = [42, 123, 456, 789, 1337]
DEFAULT_FOLDS = 5
MIN_FOLDS = 3


def choose_strata(df: pd.DataFrame) -> tuple[pd.Series, str]:
    t = pd.to_numeric(df["analysis_treatment"], errors="raise").astype(int)
    e = pd.to_numeric(df["analysis_event"], errors="raise").astype(int)
    joint = (2 * t + e).astype(str)
    counts = joint.value_counts()
    if counts.min() >= DEFAULT_FOLDS:
        return joint, "treatment_x_event"
    if t.value_counts().min() >= DEFAULT_FOLDS:
        return t.astype(str), "treatment"
    return e.astype(str), "event"


def choose_folds(strata: pd.Series) -> int:
    min_count = int(strata.value_counts().min())
    return max(MIN_FOLDS, min(DEFAULT_FOLDS, min_count))


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    split_dir = DERIVED_DIR / "nested_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    table_dir = RESULTS_DIR / "tables"

    cohort_paths = {
        "outer_hormone_hrpos_her2neg": cohort_dir / "outer_hormone_hrpos_her2neg.csv",
        "outer_chemo_tnbc": cohort_dir / "outer_chemo_tnbc.csv",
    }

    all_assignments = []
    summary_rows = []

    for cohort_name, path in cohort_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_table(path)
        strata, strategy = choose_strata(df)
        n_folds = choose_folds(strata)

        base = pd.DataFrame(
            {
                "patient_id_normalized": df["patient_id_normalized"],
                "analysis_treatment": pd.to_numeric(
                    df["analysis_treatment"], errors="raise"
                ).astype(int),
                "analysis_event": pd.to_numeric(
                    df["analysis_event"], errors="raise"
                ).astype(int),
            }
        )

        for repeat, seed in enumerate(SEEDS, start=1):
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )
            fold_assignment = np.full(len(df), -1, dtype=int)
            for fold, (_, test_idx) in enumerate(
                splitter.split(np.zeros(len(df)), strata), start=1
            ):
                fold_assignment[test_idx] = fold

            assignment = base.copy()
            assignment["cohort"] = cohort_name
            assignment["repeat"] = repeat
            assignment["seed"] = seed
            assignment["fold"] = fold_assignment
            assignment["stratification"] = strategy
            assignment["n_folds"] = n_folds
            all_assignments.append(assignment)

            for fold in range(1, n_folds + 1):
                test = assignment[assignment["fold"] == fold]
                train = assignment[assignment["fold"] != fold]
                summary_rows.append(
                    {
                        "cohort": cohort_name,
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "stratification": strategy,
                        "n_folds": n_folds,
                        "train_n": len(train),
                        "train_treated": int(train["analysis_treatment"].sum()),
                        "train_events": int(train["analysis_event"].sum()),
                        "test_n": len(test),
                        "test_treated": int(test["analysis_treatment"].sum()),
                        "test_events": int(test["analysis_event"].sum()),
                    }
                )

    assignments = pd.concat(all_assignments, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    assignments.to_csv(
        split_dir / "08_repeated_outer_fold_assignments.csv", index=False
    )
    summary.to_csv(table_dir / "08_repeated_outer_split_summary.csv", index=False)

    print("\nRepeated outer split summary:")
    print(
        summary.groupby(["cohort", "repeat", "n_folds", "stratification"])
        .agg(
            min_test_n=("test_n", "min"),
            min_test_treated=("test_treated", "min"),
            min_test_events=("test_events", "min"),
        )
        .reset_index()
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
