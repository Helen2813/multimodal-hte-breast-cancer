#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage25_v10_utils import (
    build_treatment_registry,
    compact_features,
    dataframe_console,
    ensure_dirs,
    load_config,
    load_json,
    project_root,
    read_csv,
    sha256_file,
    verify_v9_lock,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_config(root)
    ensure_dirs(root, config)

    print("=" * 128)
    print("STAGE 94 - RECONSTRUCT AND FREEZE THE CANDIDATE V10 COHORT")
    print("=" * 128)
    print("No treatment-effect model is fitted.")

    lock_check = verify_v9_lock(root, config)
    source = config["source"]
    output = config["output"]
    expected = config["expected_v9"]
    stage24 = config["stage24_reproduction"]

    v9_cohort_path = root / source["v9_cohort"]
    v9_compact_path = root / source["v9_compact"]
    clinical_path = root / source["clinical"]

    for path in (
        v9_cohort_path,
        v9_compact_path,
        clinical_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    v9 = read_csv(v9_cohort_path)
    compact = read_csv(v9_compact_path)
    clinical = pd.read_csv(
        clinical_path,
        sep="\t",
        low_memory=False,
    )

    if "patient_id_normalized" not in v9.columns:
        raise RuntimeError(
            "V9 cohort lacks patient_id_normalized."
        )
    if "patient_id_normalized" not in compact.columns:
        raise RuntimeError(
            "V9 compact table lacks patient_id_normalized."
        )

    v9["patient_id_normalized"] = (
        v9["patient_id_normalized"].astype(str)
    )
    compact["patient_id_normalized"] = (
        compact["patient_id_normalized"].astype(str)
    )

    features = compact_features(compact)
    treatment = pd.to_numeric(
        v9["analysis_treatment"],
        errors="raise",
    ).astype(int)
    event = pd.to_numeric(
        v9["analysis_event"],
        errors="raise",
    ).astype(int)

    v9_checks = pd.DataFrame([
        {
            "check": "V9 n",
            "observed": len(v9),
            "expected": expected["n"],
            "pass": len(v9) == int(expected["n"]),
        },
        {
            "check": "V9 treated",
            "observed": int(treatment.sum()),
            "expected": expected["treated"],
            "pass": int(treatment.sum())
            == int(expected["treated"]),
        },
        {
            "check": "V9 control",
            "observed": int((1 - treatment).sum()),
            "expected": expected["control"],
            "pass": int((1 - treatment).sum())
            == int(expected["control"]),
        },
        {
            "check": "V9 events",
            "observed": int(event.sum()),
            "expected": expected["events"],
            "pass": int(event.sum())
            == int(expected["events"]),
        },
        {
            "check": "V9 compact features",
            "observed": len(features),
            "expected": expected["features"],
            "pass": len(features)
            == int(expected["features"]),
        },
        {
            "check": "V9 unique IDs",
            "observed": v9[
                "patient_id_normalized"
            ].nunique(),
            "expected": len(v9),
            "pass": v9[
                "patient_id_normalized"
            ].nunique()
            == len(v9),
        },
    ])
    if not bool(v9_checks["pass"].all()):
        raise RuntimeError(
            "Candidate V9 source checks failed.\n"
            + dataframe_console(v9_checks)
        )

    registry, source_metadata, classification_audit = build_treatment_registry(
        clinical,
        v9["patient_id_normalized"],
        config,
    )

    merged = v9[
        [
            "patient_id_normalized",
            "analysis_treatment",
            "analysis_event",
            "analysis_time",
        ]
    ].merge(
        registry,
        on="patient_id_normalized",
        how="left",
        validate="one_to_one",
    )

    landmark = float(
        config["candidate_v10_population"]["landmark_day"]
    )
    hormone_start = pd.to_numeric(
        merged["earliest_hormone_start_day"],
        errors="coerce",
    )
    assignment = pd.to_numeric(
        merged["analysis_treatment"],
        errors="raise",
    ).astype(int)
    merged["hormone_assignment_timing_consistent"] = (
        (
            (assignment == 1)
            & hormone_start.between(
                0,
                landmark,
                inclusive="both",
            )
        )
        |
        (
            (assignment == 0)
            & (
                hormone_start.isna()
                | (hormone_start > landmark)
            )
        )
    ).astype(int)

    broad = (
        pd.to_numeric(
            merged["broad_no_chemo_start_by_day180"],
            errors="raise",
        ).astype(int)
        == 1
    )
    strict = (
        pd.to_numeric(
            merged["candidate_v10_eligible"],
            errors="raise",
        ).astype(int)
        == 1
    )

    broad_subset = merged[broad].copy()
    strict_ids = set(
        merged.loc[strict, "patient_id_normalized"].astype(str)
    )
    v10 = v9[
        v9["patient_id_normalized"].isin(strict_ids)
    ].copy()
    v10_compact = compact[
        compact["patient_id_normalized"].isin(strict_ids)
    ].copy()

    broad_treatment = pd.to_numeric(
        broad_subset["analysis_treatment"],
        errors="raise",
    ).astype(int)
    broad_event = pd.to_numeric(
        broad_subset["analysis_event"],
        errors="raise",
    ).astype(int)

    reproduction_checks = pd.DataFrame([
        {
            "check": "Stage24 broad no-chemo-by180 n",
            "observed": len(broad_subset),
            "expected": stage24[
                "broad_no_chemo_start_by_day180_n"
            ],
            "pass": len(broad_subset)
            == int(stage24[
                "broad_no_chemo_start_by_day180_n"
            ]),
        },
        {
            "check": "Stage24 broad treated",
            "observed": int(broad_treatment.sum()),
            "expected": stage24[
                "broad_no_chemo_start_by_day180_treated"
            ],
            "pass": int(broad_treatment.sum())
            == int(stage24[
                "broad_no_chemo_start_by_day180_treated"
            ]),
        },
        {
            "check": "Stage24 broad control",
            "observed": int((1 - broad_treatment).sum()),
            "expected": stage24[
                "broad_no_chemo_start_by_day180_control"
            ],
            "pass": int((1 - broad_treatment).sum())
            == int(stage24[
                "broad_no_chemo_start_by_day180_control"
            ]),
        },
        {
            "check": "Stage24 broad events",
            "observed": int(broad_event.sum()),
            "expected": stage24[
                "broad_no_chemo_start_by_day180_events"
            ],
            "pass": int(broad_event.sum())
            == int(stage24[
                "broad_no_chemo_start_by_day180_events"
            ]),
        },
        {
            "check": "Stage24 chemo-by180 n",
            "observed": int(
                pd.to_numeric(
                    merged["any_chemo_start_by_day180"],
                    errors="raise",
                ).sum()
            ),
            "expected": stage24[
                "any_chemo_start_by_day180_n"
            ],
            "pass": int(
                pd.to_numeric(
                    merged["any_chemo_start_by_day180"],
                    errors="raise",
                ).sum()
            )
            == int(stage24[
                "any_chemo_start_by_day180_n"
            ]),
        },
        {
            "check": "Hormone assignment timing consistency",
            "observed": int(
                merged[
                    "hormone_assignment_timing_consistent"
                ].sum()
            ),
            "expected": len(merged),
            "pass": bool(
                merged[
                    "hormone_assignment_timing_consistent"
                ].all()
            ),
        },
    ])
    if not bool(reproduction_checks["pass"].all()):
        raise RuntimeError(
            "Stage 24 timing reconstruction was not reproduced.\n"
            + dataframe_console(reproduction_checks)
        )

    v10_treatment = pd.to_numeric(
        v10["analysis_treatment"],
        errors="raise",
    ).astype(int)
    v10_event = pd.to_numeric(
        v10["analysis_event"],
        errors="raise",
    ).astype(int)

    strict_registry = merged[strict].copy()
    strict_checks = pd.DataFrame([
        {
            "check": "V10 IDs unique",
            "observed": v10[
                "patient_id_normalized"
            ].nunique(),
            "expected": len(v10),
            "pass": v10[
                "patient_id_normalized"
            ].nunique()
            == len(v10),
        },
        {
            "check": "V10 cohort/compact row match",
            "observed": len(v10_compact),
            "expected": len(v10),
            "pass": len(v10_compact) == len(v10),
        },
        {
            "check": "No V10 chemotherapy start by day180",
            "observed": int(
                strict_registry[
                    "any_chemo_start_by_day180"
                ].sum()
            ),
            "expected": 0,
            "pass": int(
                strict_registry[
                    "any_chemo_start_by_day180"
                ].sum()
            )
            == 0,
        },
        {
            "check": "All V10 chemotherapy timing ascertainable",
            "observed": int(
                strict_registry[
                    "all_chemo_start_timing_ascertainable"
                ].sum()
            ),
            "expected": len(strict_registry),
            "pass": bool(
                strict_registry[
                    "all_chemo_start_timing_ascertainable"
                ].all()
            ),
        },
        {
            "check": "V10 hormone assignment timing consistent",
            "observed": int(
                strict_registry[
                    "hormone_assignment_timing_consistent"
                ].sum()
            ),
            "expected": len(strict_registry),
            "pass": bool(
                strict_registry[
                    "hormone_assignment_timing_consistent"
                ].all()
            ),
        },
    ])
    if not bool(strict_checks["pass"].all()):
        raise RuntimeError(
            "Candidate V10 strict cohort checks failed.\n"
            + dataframe_console(strict_checks)
        )

    cohort_path = root / output["cohort"]
    compact_path = root / output["compact"]
    registry_path = root / output["sequence_registry"]
    write_csv(v10, cohort_path)
    write_csv(v10_compact, compact_path)
    write_csv(merged, registry_path)

    summary = {
        "v9_n": len(v9),
        "v9_treated": int(treatment.sum()),
        "v9_control": int((1 - treatment).sum()),
        "v9_events": int(event.sum()),
        "stage24_broad_no_chemo_by180_n": len(broad_subset),
        "excluded_for_unascertainable_chemo_start_timing": int(
            broad.sum() - strict.sum()
        ),
        "candidate_v10_n": len(v10),
        "candidate_v10_treated": int(v10_treatment.sum()),
        "candidate_v10_control": int(
            (1 - v10_treatment).sum()
        ),
        "candidate_v10_events": int(v10_event.sum()),
        "candidate_v10_treated_events": int(
            ((v10_treatment == 1) & (v10_event == 1)).sum()
        ),
        "candidate_v10_control_events": int(
            ((v10_treatment == 0) & (v10_event == 1)).sum()
        ),
        "candidate_v10_features": len(features),
        "candidate_v10_cohort_sha256": sha256_file(cohort_path),
        "candidate_v10_compact_sha256": sha256_file(
            compact_path
        ),
        "selection_rule": config[
            "candidate_v10_population"
        ]["selection_rule"],
        "future_treatment_rule": config[
            "candidate_v10_population"
        ]["future_treatment_rule"],
        "source_metadata": source_metadata,
    }

    table_dir = root / output["table_dir"]
    write_csv(
        lock_check,
        table_dir / "s25_94_v9_integrity_check.csv",
    )
    write_csv(
        v9_checks,
        table_dir / "s25_94_v9_source_checks.csv",
    )
    write_csv(
        reproduction_checks,
        table_dir / "s25_94_stage24_reproduction_checks.csv",
    )
    write_csv(
        strict_checks,
        table_dir / "s25_94_v10_strict_cohort_checks.csv",
    )
    write_json(
        summary,
        table_dir / "s25_94_v10_cohort_summary.json",
    )

    status_counts = (
        merged.groupby(
            [
                "chemo_timing_status",
                "analysis_treatment",
            ],
            dropna=False,
        )
        .agg(
            n=("patient_id_normalized", "size"),
            events=("analysis_event", "sum"),
        )
        .reset_index()
    )
    write_csv(
        status_counts,
        table_dir / "s25_94_chemo_timing_status_counts.csv",
    )
    write_csv(
        classification_audit,
        table_dir / "s25_94_treatment_classification_text_audit.csv",
    )

    print("V9 source checks")
    print(dataframe_console(v9_checks))
    print("\nStage 24 reproduction checks")
    print(dataframe_console(reproduction_checks))
    print("\nV10 strict cohort checks")
    print(dataframe_console(strict_checks))
    print("\nChemotherapy timing status counts")
    print(dataframe_console(status_counts))
    print("\nTreatment classification text audit")
    print(dataframe_console(classification_audit, max_rows=200))
    print("\nCandidate V10 cohort summary")
    print(json.dumps(summary, indent=2))

    print(
        "\nPASS: Candidate V10 cohort frozen. "
        "No treatment-effect estimate was computed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
