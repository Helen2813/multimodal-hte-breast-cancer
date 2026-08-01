from __future__ import annotations

import json
import numpy as np
import pandas as pd

from _metabric_m1_utils import (
    choose_column, column_profile, find_timing_columns, load_config, norm_col,
    normalize_event, normalize_receptor, normalize_yes_no, numeric, out_dir,
    print_table, project_root, raw_dir, read_cbio_table, safe_values, write_csv
)


def aggregate_sample_to_patient(sample: pd.DataFrame, patient_col: str) -> tuple[pd.DataFrame, int]:
    duplicate_rows = int(sample.duplicated(patient_col, keep=False).sum())
    if duplicate_rows == 0:
        return sample.copy(), 0

    rows = []
    for patient_id, group in sample.groupby(patient_col, dropna=False, sort=False):
        row = {patient_col: patient_id}
        for col in sample.columns:
            if col == patient_col:
                continue
            vals = group[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""].unique().tolist()
            row[col] = vals[0] if vals else np.nan
            if len(vals) > 1:
                row[f"__CONFLICT__{col}"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows), duplicate_rows


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    patient_path = raw / cfg["expected_files"]["clinical_patient"]
    sample_path = raw / cfg["expected_files"]["clinical_sample"]
    patient = read_cbio_table(patient_path)
    sample = read_cbio_table(sample_path)

    print("=" * 124)
    print("METABRIC M1.02 - CLINICAL, TREATMENT-TIMING, AND SURVIVAL AUDIT")
    print("=" * 124)
    print(f"Patient clinical rows: {len(patient)}; columns: {len(patient.columns)}")
    print(f"Sample clinical rows: {len(sample)}; columns: {len(sample.columns)}")

    patient_id_patient = choose_column(patient.columns, ["PATIENT_ID"], ["PATIENT", "ID"])
    patient_id_sample = choose_column(sample.columns, ["PATIENT_ID"], ["PATIENT", "ID"])
    sample_id = choose_column(sample.columns, ["SAMPLE_ID"], ["SAMPLE", "ID"])

    if patient_id_patient is None or patient_id_sample is None:
        raise RuntimeError("Could not resolve PATIENT_ID in both clinical files.")

    sample_agg, duplicate_sample_rows = aggregate_sample_to_patient(sample, patient_id_sample)
    if patient_id_sample != patient_id_patient:
        sample_agg = sample_agg.rename(columns={patient_id_sample: patient_id_patient})

    merged = patient.merge(sample_agg, on=patient_id_patient, how="outer", suffixes=("_PATIENT", "_SAMPLE"))

    profiles_patient = column_profile(patient, int(cfg["print_unique_examples_per_column"]))
    profiles_sample = column_profile(sample, int(cfg["print_unique_examples_per_column"]))
    profiles_merged = column_profile(merged, int(cfg["print_unique_examples_per_column"]))

    write_csv(out / "m02_patient_clinical_columns.csv", profiles_patient)
    write_csv(out / "m02_sample_clinical_columns.csv", profiles_sample)
    write_csv(out / "m02_merged_clinical_columns.csv", profiles_merged)

    print("\nPatient clinical columns")
    print_table(profiles_patient, ["column", "nonmissing", "missing_fraction", "unique_nonmissing", "examples"])

    print("\nSample clinical columns")
    print_table(profiles_sample, ["column", "nonmissing", "missing_fraction", "unique_nonmissing", "examples"])

    cols = list(merged.columns)
    fields = {
        "patient_id": patient_id_patient,
        "sample_id": choose_column(cols, ["SAMPLE_ID"], ["SAMPLE", "ID"]),
        "hormone_therapy": choose_column(
            cols,
            ["HORMONE_THERAPY", "ENDOCRINE_THERAPY", "HORMONAL_THERAPY"],
            ["HORMONE", "THERAPY"],
            ["START", "DATE", "DAY", "MONTH", "YEAR", "TIME"]
        ),
        "chemotherapy": choose_column(
            cols, ["CHEMOTHERAPY", "CHEMO_THERAPY"], ["CHEMO"],
            ["START", "DATE", "DAY", "MONTH", "YEAR", "TIME"]
        ),
        "radiotherapy": choose_column(
            cols, ["RADIO_THERAPY", "RADIOTHERAPY"], ["RADIO", "THERAPY"],
            ["START", "DATE", "DAY", "MONTH", "YEAR", "TIME"]
        ),
        "er_status": choose_column(cols, ["ER_STATUS", "ER_STATUS_BY_IHC"], ["ER", "STATUS"], ["HER2"]),
        "pr_status": choose_column(cols, ["PR_STATUS", "PR_STATUS_BY_IHC"], ["PR", "STATUS"]),
        "her2_status": choose_column(cols, ["HER2_STATUS", "HER2_STATUS_BY_IHC"], ["HER2", "STATUS"]),
        "os_months": choose_column(cols, ["OS_MONTHS", "OVERALL_SURVIVAL_MONTHS"], ["OS", "MONTH"]),
        "os_status": choose_column(cols, ["OS_STATUS", "OVERALL_SURVIVAL_STATUS"], ["OS", "STATUS"]),
        "rfs_months": choose_column(cols, ["RFS_MONTHS", "DFS_MONTHS"], ["RFS", "MONTH"]),
        "rfs_status": choose_column(cols, ["RFS_STATUS", "DFS_STATUS"], ["RFS", "STATUS"]),
        "age_at_diagnosis": choose_column(cols, ["AGE_AT_DIAGNOSIS"], ["AGE", "DIAGNOS"]),
        "diagnosis_year": choose_column(cols, ["YEAR_OF_DIAGNOSIS", "DIAGNOSIS_YEAR"], ["YEAR", "DIAGNOS"]),
        "stage": choose_column(cols, ["TUMOR_STAGE", "STAGE"], ["STAGE"]),
        "grade": choose_column(cols, ["NEOPLASM_HISTOLOGIC_GRADE", "GRADE"], ["GRADE"]),
        "tumor_size": choose_column(cols, ["TUMOR_SIZE"], ["TUMOR", "SIZE"]),
        "positive_nodes": choose_column(cols, ["LYMPH_NODES_EXAMINED_POSITIVE"], ["NODE", "POSITIVE"]),
    }

    field_rows = [{"role": k, "resolved_column": v or "", "found": v is not None} for k, v in fields.items()]
    write_csv(out / "m02_resolved_clinical_fields.csv", field_rows)

    print("\nResolved clinical fields")
    print_table(field_rows, ["role", "resolved_column", "found"])

    timing_columns = find_timing_columns(cols)
    timing_rows = []
    for col in timing_columns:
        s = merged[col]
        timing_rows.append({
            "column": col,
            "nonmissing": int(s.notna().sum()),
            "unique_nonmissing": int(s.dropna().astype(str).nunique()),
            "examples": " | ".join(safe_values(s, 8)),
        })
    write_csv(
        out / "m02_treatment_timing_columns.csv",
        timing_rows,
        fieldnames=["column", "nonmissing", "unique_nonmissing", "examples"],
    )

    print("\nTreatment timing candidates")
    print_table(timing_rows, ["column", "nonmissing", "unique_nonmissing", "examples"])

    analysis = pd.DataFrame(index=merged.index)
    analysis["patient_id"] = merged[patient_id_patient].astype(str)
    for role in ("hormone_therapy", "chemotherapy", "radiotherapy"):
        analysis[role] = normalize_yes_no(merged[fields[role]]) if fields[role] else np.nan
    for role in ("er_status", "pr_status", "her2_status"):
        analysis[role] = normalize_receptor(merged[fields[role]]) if fields[role] else np.nan
    analysis["os_months"] = numeric(merged[fields["os_months"]]) if fields["os_months"] else np.nan
    analysis["os_event"] = normalize_event(merged[fields["os_status"]]) if fields["os_status"] else np.nan

    analysis["hrpos_er_only_her2neg"] = (
        (analysis["er_status"] == 1) & (analysis["her2_status"] == 0)
    )
    analysis["hrpos_er_or_pr_her2neg"] = (
        ((analysis["er_status"] == 1) | (analysis["pr_status"] == 1))
        & (analysis["her2_status"] == 0)
    )

    landmark_months = float(cfg["target_landmark_days"]) / 30.4375
    total_target_months = float(cfg["target_landmark_days"] + cfg["target_post_landmark_horizon_days"]) / 30.4375
    analysis["observed_at_day180"] = analysis["os_months"] >= landmark_months
    analysis["observed_to_target_horizon"] = analysis["os_months"] >= total_target_months

    cohort_rows = []
    for definition in ("hrpos_er_only_her2neg", "hrpos_er_or_pr_her2neg"):
        cohort = analysis[analysis[definition]].copy()
        for therapy in ("hormone_therapy", "chemotherapy"):
            known = cohort[cohort[therapy].isin([0.0, 1.0])]
            for value, label in ((1.0, "treated"), (0.0, "control")):
                group = known[known[therapy] == value]
                cohort_rows.append({
                    "receptor_definition": definition,
                    "therapy": therapy,
                    "group": label,
                    "n": len(group),
                    "events": int((group["os_event"] == 1).sum()),
                    "event_status_known": int(group["os_event"].notna().sum()),
                    "os_months_known": int(group["os_months"].notna().sum()),
                    "median_os_months": float(group["os_months"].median()) if group["os_months"].notna().any() else np.nan,
                    "observed_at_day180": int(group["observed_at_day180"].sum()),
                    "observed_to_day910_equivalent": int(group["observed_to_target_horizon"].sum()),
                })

    write_csv(out / "m02_candidate_cohort_counts.csv", cohort_rows)

    print("\nCandidate HR+/HER2- treatment groups")
    print_table(
        cohort_rows,
        ["receptor_definition", "therapy", "group", "n", "events", "median_os_months",
         "observed_at_day180", "observed_to_day910_equivalent"]
    )

    timing_coverage = max((int(r["nonmissing"]) for r in timing_rows), default=0)
    exact_day180_feasible = bool(
        fields["hormone_therapy"]
        and fields["os_months"]
        and fields["os_status"]
        and timing_rows
        and timing_coverage > 0
    )

    summary = {
        "patient_rows": int(len(patient)),
        "sample_rows": int(len(sample)),
        "merged_patient_rows": int(len(merged)),
        "duplicate_sample_rows_before_patient_aggregation": duplicate_sample_rows,
        "patient_id_column": patient_id_patient,
        "sample_id_column": sample_id or "",
        "resolved_fields": fields,
        "treatment_timing_candidate_count": len(timing_rows),
        "maximum_timing_field_nonmissing": timing_coverage,
        "exact_day180_replication_preliminarily_feasible": exact_day180_feasible,
        "landmark_months_approx": landmark_months,
        "day910_equivalent_months_approx": total_target_months,
    }
    (out / "m02_clinical_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nClinical audit summary")
    for key, value in summary.items():
        if key == "resolved_fields":
            continue
        print(f"  {key}: {value}")

    print("\nPASS: clinical and timing audit completed. No treatment effect was estimated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
