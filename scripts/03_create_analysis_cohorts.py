from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from _common import (
    DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table,
    exact_or_case_insensitive_column, parse_binary_status,
    safe_numeric_event, write_markdown,
)


def require_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    column = exact_or_case_insensitive_column(df, candidates)
    if column is None:
        raise ValueError(f"Required column for {label} not found. Tried: {candidates}")
    return column


def choose_outcome(df: pd.DataFrame):
    event = exact_or_case_insensitive_column(df, ["OS", "Y_died_5yr", "Y", "event", "status"])
    time = exact_or_case_insensitive_column(df, [
        "OS.time", "OS_time", "survival_time", "time", "days_to_event",
        "days_to_death", "days_to_last_follow_up",
    ])
    if event is None:
        raise ValueError("No event/outcome column found")
    return event, time


def create_one(df: pd.DataFrame, representation: str, cohort_dir: Path):
    er_col = require_column(df, ["ER_status"], "ER")
    pr_col = require_column(df, ["PR_status"], "PR")
    her2_col = require_column(df, ["HER2_status"], "HER2")
    hormone_col = require_column(df, ["T_hormone"], "hormone treatment")
    chemo_col = require_column(df, ["T_chemo"], "chemotherapy")
    event_col, time_col = choose_outcome(df)

    er = parse_binary_status(df[er_col])
    pr = parse_binary_status(df[pr_col])
    her2 = parse_binary_status(df[her2_col])
    hormone = parse_binary_status(df[hormone_col])
    chemo = parse_binary_status(df[chemo_col])
    event = safe_numeric_event(df[event_col])
    known = er.notna() & pr.notna() & her2.notna()

    definitions = {
        "hormone_hrpos_her2neg": {
            "mask": known & ((er == 1) | (pr == 1)) & (her2 == 0),
            "treatment": hormone,
            "treatment_column": hormone_col,
            "description": "ER-positive or PR-positive and HER2-negative",
        },
        "chemo_tnbc": {
            "mask": known & (er == 0) & (pr == 0) & (her2 == 0),
            "treatment": chemo,
            "treatment_column": chemo_col,
            "description": "Triple-negative breast cancer",
        },
    }

    summaries = []
    for cohort_name, definition in definitions.items():
        mask = definition["mask"] & definition["treatment"].notna() & event.notna()
        cohort = df.loc[mask].copy()
        cohort["analysis_treatment"] = definition["treatment"].loc[mask].astype(int)
        cohort["analysis_event"] = event.loc[mask].astype(int)
        if time_col:
            cohort["analysis_time"] = pd.to_numeric(cohort[time_col], errors="coerce")

        output = cohort_dir / f"{representation}_{cohort_name}.csv"
        cohort.to_csv(output, index=False)
        treated = int(cohort["analysis_treatment"].sum())
        control = int(len(cohort) - treated)
        events = int(cohort["analysis_event"].sum())
        summaries.append({
            "representation": representation,
            "cohort": cohort_name,
            "description": definition["description"],
            "n": int(len(cohort)),
            "treated": treated,
            "control": control,
            "events": events,
            "event_rate": float(cohort["analysis_event"].mean()) if len(cohort) else np.nan,
            "treatment_column": definition["treatment_column"],
            "event_column": event_col,
            "time_column": time_col or "",
            "output": str(output),
            "status": "OK" if treated >= 10 and control >= 10 and events >= 5 else "REVIEW_SMALL_COUNTS",
        })
    return summaries


def main() -> int:
    ensure_dirs()
    cohort_dir = DERIVED_DIR / "cohorts"
    table_dir = RESULTS_DIR / "tables"
    masters = {
        "outer": cohort_dir / "master_outer.csv",
        "complete_case": cohort_dir / "master_complete_case.csv",
    }
    missing = [str(path) for path in masters.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run 02_build_master_tables.py first:\n" + "\n".join(missing))

    summaries = []
    for representation, path in masters.items():
        print(f"Creating cohorts from {representation}")
        summaries.extend(create_one(read_table(path), representation, cohort_dir))

    summary = pd.DataFrame(summaries)
    summary.to_csv(table_dir / "03_cohort_summary.csv", index=False)
    lines = ["# Analysis cohort summary", ""]
    for row in summaries:
        lines.append(
            f"- `{row['representation']} / {row['cohort']}`: n={row['n']}, "
            f"treated={row['treated']}, control={row['control']}, events={row['events']}, "
            f"status={row['status']}"
        )
    write_markdown(lines, table_dir / "03_cohort_summary.md")
    print("\nCohort summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {cohort_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
