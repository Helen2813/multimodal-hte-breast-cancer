from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _metabric_m5_utils import (
    exact_column, load_config, normalize_event, out_dir, print_table,
    project_root, read_rows, write_csv
)


def endpoint_candidates(frame: pd.DataFrame, source: str) -> list[dict]:
    rows = []
    for column in frame.columns:
        normalized = str(column).upper()
        if any(token in normalized for token in (
            "OS", "RFS", "DFS", "PFI", "DSS", "SURV", "EVENT", "TIME", "MONTH"
        )):
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            rows.append({
                "source": source,
                "column": column,
                "nonmissing": int(series.notna().sum()),
                "unique_nonmissing": int(series.dropna().astype(str).nunique()),
                "numeric_nonmissing": int(numeric.notna().sum()),
                "minimum_numeric": float(numeric.min()) if numeric.notna().any() else "",
                "maximum_numeric": float(numeric.max()) if numeric.notna().any() else "",
                "examples": " | ".join(series.dropna().astype(str).unique()[:8]),
            })
    return rows


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M5.29 - ENDPOINT AND COHORT ALIGNMENT AUDIT")
    print("=" * 124)

    tcga_path = root / cfg["tcga_canonical_table"]
    tcga = pd.read_csv(tcga_path, low_memory=False)
    endpoint_rows = endpoint_candidates(tcga, cfg["tcga_canonical_table"])

    paper1_outcome_rows = []
    for modality, folder_name in cfg["paper1_modalities"].items():
        path = root / cfg["paper1_processed_root"] / folder_name / "statistical_filtered" / "outcome.csv"
        if not path.exists():
            paper1_outcome_rows.append({
                "modality": modality,
                "path": str(path),
                "found": False,
                "rows": 0,
                "columns": "",
            })
            continue
        frame = pd.read_csv(path, low_memory=False)
        paper1_outcome_rows.append({
            "modality": modality,
            "path": path.relative_to(root).as_posix(),
            "found": True,
            "rows": len(frame),
            "columns": " | ".join(map(str, frame.columns)),
        })
        endpoint_rows.extend(endpoint_candidates(frame, path.relative_to(root).as_posix()))

    metabric_master_path = (
        root / cfg["metabric_m2_dir"] / "m06_metabric_clinical_master_LOCAL_ONLY.csv"
    )
    metabric = pd.read_csv(metabric_master_path, low_memory=False)
    endpoint_rows.extend(endpoint_candidates(metabric, metabric_master_path.relative_to(root).as_posix()))

    write_csv(out / "m29_endpoint_candidates.csv", endpoint_rows)
    write_csv(out / "m29_paper1_outcome_files.csv", paper1_outcome_rows)

    cohort_rows = []
    masks = {
        "all_metabric": pd.Series(True, index=metabric.index),
        "hrpos_er_or_pr_her2neg": metabric["hrpos_er_or_pr_her2neg"].astype(bool),
        "tnbc_ihc": metabric["tnbc_ihc"].astype(bool),
    }
    for cohort_name, mask in masks.items():
        cohort = metabric.loc[mask].copy()
        for endpoint, time_col, event_col in (
            ("OS", "os_months", "os_event"),
            ("RFS", "rfs_months", "rfs_event"),
        ):
            complete = cohort[
                pd.to_numeric(cohort[time_col], errors="coerce").notna()
                & pd.to_numeric(cohort[event_col], errors="coerce").notna()
            ]
            cohort_rows.append({
                "cohort": cohort_name,
                "endpoint": endpoint,
                "n": len(cohort),
                "complete_n": len(complete),
                "events": int((pd.to_numeric(complete[event_col], errors="coerce") == 1).sum()),
                "median_time_months": float(pd.to_numeric(complete[time_col], errors="coerce").median()) if len(complete) else "",
                "p90_time_months": float(pd.to_numeric(complete[time_col], errors="coerce").quantile(0.9)) if len(complete) else "",
            })

    write_csv(out / "m29_metabric_endpoint_counts.csv", cohort_rows)

    tcga_time_candidates = [
        row for row in endpoint_rows
        if row["source"] == cfg["tcga_canonical_table"]
        and any(token in str(row["column"]).upper() for token in ("OS_MONTH", "OS_TIME", "SURVIVAL_TIME"))
    ]
    tcga_event_candidates = [
        row for row in endpoint_rows
        if row["source"] == cfg["tcga_canonical_table"]
        and any(token in str(row["column"]).upper() for token in ("OS_STATUS", "OS_EVENT", "VITAL"))
    ]

    resolution = {
        "metabric_primary_endpoint": "OS",
        "metabric_sensitivity_endpoint": "RFS",
        "paper1_outcome_files_found": sum(bool(row["found"]) for row in paper1_outcome_rows),
        "paper1_outcome_files_expected": len(paper1_outcome_rows),
        "tcga_os_time_candidate_count": len(tcga_time_candidates),
        "tcga_os_event_candidate_count": len(tcga_event_candidates),
        "tcga_os_time_candidates": [row["column"] for row in tcga_time_candidates],
        "tcga_os_event_candidates": [row["column"] for row in tcga_event_candidates],
        "endpoint_lock_ready": (
            all(bool(row["found"]) for row in paper1_outcome_rows)
            and len(cohort_rows) > 0
        ),
        "important_rule": (
            "Endpoint identity is determined from Paper-1 outcome files and variable semantics, not by whichever endpoint yields better METABRIC performance."
        ),
    }
    (out / "m29_endpoint_resolution.json").write_text(
        json.dumps(resolution, indent=2), encoding="utf-8"
    )

    print("Paper-1 outcome files")
    print_table(paper1_outcome_rows, ["modality", "path", "found", "rows", "columns"])

    print("\nEndpoint candidates")
    print_table(
        endpoint_rows,
        [
            "source", "column", "nonmissing", "unique_nonmissing",
            "numeric_nonmissing", "minimum_numeric", "maximum_numeric", "examples"
        ],
        max_rows=120,
    )

    print("\nMETABRIC endpoint counts")
    print_table(
        cohort_rows,
        ["cohort", "endpoint", "n", "complete_n", "events", "median_time_months", "p90_time_months"],
    )

    print("\nEndpoint resolution")
    for key, value in resolution.items():
        print(f"  {key}: {value}")

    if not resolution["endpoint_lock_ready"]:
        raise RuntimeError("Endpoint files are incomplete; protocol cannot be locked.")

    print("\nPASS: endpoint/cohort alignment audited without fitting a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
