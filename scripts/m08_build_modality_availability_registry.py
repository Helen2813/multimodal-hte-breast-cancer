from __future__ import annotations

import json

import pandas as pd

from _metabric_m2_utils import (
    load_config, out_dir, print_table, project_root, write_csv
)


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M2.08 - MULTIMODAL AVAILABILITY REGISTRY")
    print("=" * 124)

    master = pd.read_csv(out / "m06_metabric_clinical_master_LOCAL_ONLY.csv", low_memory=False)
    sample_sets = json.loads((out / "m05_exact_sample_sets_LOCAL_ONLY.json").read_text(encoding="utf-8"))
    sets = {role: set(values) for role, values in sample_sets.items()}

    modality_roles = ["mrna_raw", "mrna_zscores", "cna", "methylation", "gene_panel", "rna_cleaned", "mutations"]
    registry = master[["patient_id", "sample_id", "hrpos_er_only_her2neg",
                       "hrpos_er_or_pr_her2neg", "tnbc_ihc"]].copy()
    for role in modality_roles:
        registry[role] = registry["sample_id"].astype(str).isin(sets.get(role, set()))

    registry["rna_cna_mutations"] = registry[["mrna_raw", "cna", "mutations"]].all(axis=1)
    registry["rna_cna_mutations_methylation"] = registry[
        ["mrna_raw", "cna", "mutations", "methylation"]
    ].all(axis=1)
    registry.to_csv(out / "m08_patient_modality_registry_LOCAL_ONLY.csv", index=False)

    cohorts = {
        "all_clinical": pd.Series(True, index=registry.index),
        "hrpos_er_only_her2neg": registry["hrpos_er_only_her2neg"].fillna(False).astype(bool),
        "hrpos_er_or_pr_her2neg": registry["hrpos_er_or_pr_her2neg"].fillna(False).astype(bool),
        "tnbc_ihc": registry["tnbc_ihc"].fillna(False).astype(bool),
    }

    rows = []
    for cohort_name, mask in cohorts.items():
        cohort = registry[mask]
        row = {"cohort": cohort_name, "clinical_n": len(cohort)}
        for role in modality_roles + ["rna_cna_mutations", "rna_cna_mutations_methylation"]:
            row[role] = int(cohort[role].sum())
        rows.append(row)

    write_csv(out / "m08_modality_availability_summary.csv", rows)

    print("Multimodal availability")
    print_table(
        rows,
        ["cohort", "clinical_n", "mrna_raw", "cna", "mutations", "methylation",
         "rna_cna_mutations", "rna_cna_mutations_methylation", "gene_panel", "rna_cleaned"]
    )

    print("\nPatient/sample identifiers are stored only in LOCAL_ONLY outputs and are not printed.")
    print("\nPASS: exact patient-level modality availability registry created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
