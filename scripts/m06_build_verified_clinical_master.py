from __future__ import annotations

import json
import numpy as np
import pandas as pd

from _metabric_m2_utils import (
    exact_col, load_config, normalize_event, normalize_receptor,
    normalize_yes_no, out_dir, print_table, project_root, raw_dir,
    read_cbio, to_numeric, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M2.06 - VERIFIED PATIENT-LEVEL CLINICAL MASTER")
    print("=" * 124)

    patient = read_cbio(raw / cfg["files"]["clinical_patient"])
    sample = read_cbio(raw / cfg["files"]["clinical_sample"])

    patient_id = exact_col(patient.columns, ["PATIENT_ID"])
    sample_patient_id = exact_col(sample.columns, ["PATIENT_ID"])
    sample_id = exact_col(sample.columns, ["SAMPLE_ID"])
    if not all([patient_id, sample_patient_id, sample_id]):
        raise RuntimeError("Could not resolve PATIENT_ID/SAMPLE_ID exactly.")

    if patient[patient_id].duplicated().any():
        raise RuntimeError("Patient clinical table contains duplicate PATIENT_ID values.")
    if sample[sample_patient_id].duplicated().any():
        raise RuntimeError("Sample clinical table is not one sample per patient.")

    merged = patient.merge(
        sample,
        left_on=patient_id,
        right_on=sample_patient_id,
        how="outer",
        suffixes=("_PATIENT", "_SAMPLE"),
        validate="one_to_one",
    )

    exact_fields = {
        "hormone_therapy": exact_col(merged.columns, ["HORMONE_THERAPY"]),
        "chemotherapy": exact_col(merged.columns, ["CHEMOTHERAPY"]),
        "radiotherapy": exact_col(merged.columns, ["RADIO_THERAPY", "RADIOTHERAPY"]),
        "er_status": exact_col(merged.columns, ["ER_STATUS"]),
        "pr_status": exact_col(merged.columns, ["PR_STATUS"]),
        "her2_status": exact_col(merged.columns, ["HER2_STATUS"]),
        "os_months": exact_col(merged.columns, ["OS_MONTHS"]),
        "os_status": exact_col(merged.columns, ["OS_STATUS"]),
        "rfs_months": exact_col(merged.columns, ["RFS_MONTHS"]),
        "rfs_status": exact_col(merged.columns, ["RFS_STATUS"]),
        "age_at_diagnosis": exact_col(merged.columns, ["AGE_AT_DIAGNOSIS"]),
        "grade": exact_col(merged.columns, ["GRADE"]),
        "stage": exact_col(merged.columns, ["TUMOR_STAGE"]),
        "tumor_size": exact_col(merged.columns, ["TUMOR_SIZE"]),
        "positive_nodes": exact_col(merged.columns, ["LYMPH_NODES_EXAMINED_POSITIVE"]),
        "npi": exact_col(merged.columns, ["NPI"]),
        "cohort": exact_col(merged.columns, ["COHORT"]),
        "menopausal_state": exact_col(merged.columns, ["INFERRED_MENOPAUSAL_STATE"]),
        "cellularity": exact_col(merged.columns, ["CELLULARITY"]),
        "histological_subtype": exact_col(merged.columns, ["HISTOLOGICAL_SUBTYPE"]),
        "intclust": exact_col(merged.columns, ["INTCLUST"]),
    }

    master = pd.DataFrame({
        "patient_id": merged[patient_id].astype(str).str.strip(),
        "sample_id": merged[sample_id].astype(str).str.strip(),
    })
    for role in ("hormone_therapy", "chemotherapy", "radiotherapy"):
        col = exact_fields[role]
        master[role] = normalize_yes_no(merged[col]) if col else np.nan
    for role in ("er_status", "pr_status", "her2_status"):
        col = exact_fields[role]
        master[role] = normalize_receptor(merged[col]) if col else np.nan

    master["os_months"] = to_numeric(merged[exact_fields["os_months"]])
    master["os_event"] = normalize_event(merged[exact_fields["os_status"]])
    master["rfs_months"] = to_numeric(merged[exact_fields["rfs_months"]])
    master["rfs_event"] = normalize_event(merged[exact_fields["rfs_status"]])

    for role in ("age_at_diagnosis", "grade", "stage", "tumor_size", "positive_nodes", "npi"):
        col = exact_fields[role]
        master[role] = to_numeric(merged[col]) if col else np.nan

    for role in ("cohort", "menopausal_state", "cellularity", "histological_subtype", "intclust"):
        col = exact_fields[role]
        master[role] = merged[col] if col else np.nan

    master["hrpos_er_only_her2neg"] = (
        (master["er_status"] == 1) & (master["her2_status"] == 0)
    )
    master["hrpos_er_or_pr_her2neg"] = (
        ((master["er_status"] == 1) | (master["pr_status"] == 1))
        & (master["her2_status"] == 0)
    )
    master["tnbc_ihc"] = (
        (master["er_status"] == 0)
        & (master["pr_status"] == 0)
        & (master["her2_status"] == 0)
    )

    local_path = out / "m06_metabric_clinical_master_LOCAL_ONLY.csv"
    master.to_csv(local_path, index=False)

    summary_rows = []
    definitions = ["all", "hrpos_er_only_her2neg", "hrpos_er_or_pr_her2neg", "tnbc_ihc"]
    for definition in definitions:
        cohort = master if definition == "all" else master[master[definition]]
        summary_rows.append({
            "cohort": definition,
            "n": len(cohort),
            "os_time_known": int(cohort["os_months"].notna().sum()),
            "os_event_known": int(cohort["os_event"].notna().sum()),
            "os_events": int((cohort["os_event"] == 1).sum()),
            "hormone_known": int(cohort["hormone_therapy"].notna().sum()),
            "hormone_yes": int((cohort["hormone_therapy"] == 1).sum()),
            "hormone_no": int((cohort["hormone_therapy"] == 0).sum()),
            "chemotherapy_yes": int((cohort["chemotherapy"] == 1).sum()),
            "chemotherapy_no": int((cohort["chemotherapy"] == 0).sum()),
            "median_os_months": float(cohort["os_months"].median()) if cohort["os_months"].notna().any() else np.nan,
        })

    missing_rows = []
    for col in master.columns:
        if col in {"patient_id", "sample_id"}:
            continue
        missing_rows.append({
            "variable": col,
            "nonmissing": int(master[col].notna().sum()),
            "missing": int(master[col].isna().sum()),
            "missing_fraction": float(master[col].isna().mean()),
        })

    field_rows = [{"role": role, "exact_column": col or "", "found": col is not None}
                  for role, col in exact_fields.items()]
    field_rows.append({
        "role": "diagnosis_year",
        "exact_column": "",
        "found": False,
    })

    write_csv(out / "m06_clinical_cohort_summary.csv", summary_rows)
    write_csv(out / "m06_clinical_missingness.csv", missing_rows)
    write_csv(out / "m06_exact_field_registry.csv", field_rows)

    print("Clinical cohort summary")
    print_table(
        summary_rows,
        ["cohort", "n", "os_time_known", "os_events", "hormone_yes",
         "hormone_no", "chemotherapy_yes", "chemotherapy_no", "median_os_months"]
    )

    print("\nExact field registry")
    print_table(field_rows, ["role", "exact_column", "found"])

    print("\nClinical missingness")
    print_table(missing_rows, ["variable", "nonmissing", "missing", "missing_fraction"])

    print(f"\nLocal linked clinical master: {local_path}")
    print("Patient identifiers are not printed to the terminal.")
    print("\nPASS: verified patient-level clinical master created. No effect was estimated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
