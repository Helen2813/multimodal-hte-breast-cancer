from __future__ import annotations

import csv
import json

from _metabric_m2_utils import load_config, out_dir, print_table, project_root, write_csv


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_int(x):
    return int(float(x))


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    print("=" * 124)
    print("METABRIC M2.09 - TRANSPORT-VALIDATION READINESS DECISION")
    print("=" * 124)

    availability = read_rows(out / "m08_modality_availability_summary.csv")
    provenance = read_rows(out / "m07_rna_provenance_summary.csv")[0]
    target = next(r for r in availability if r["cohort"] == "hrpos_er_or_pr_her2neg")
    gates_cfg = cfg["transport_readiness"]

    checks = [
        {
            "check": "HR+/HER2- with raw RNA",
            "observed": as_int(target["mrna_raw"]),
            "threshold": int(gates_cfg["minimum_hrpos_her2neg_with_rna"]),
            "pass": as_int(target["mrna_raw"]) >= int(gates_cfg["minimum_hrpos_her2neg_with_rna"]),
        },
        {
            "check": "HR+/HER2- with CNA",
            "observed": as_int(target["cna"]),
            "threshold": int(gates_cfg["minimum_hrpos_her2neg_with_cna"]),
            "pass": as_int(target["cna"]) >= int(gates_cfg["minimum_hrpos_her2neg_with_cna"]),
        },
        {
            "check": "HR+/HER2- with mutations",
            "observed": as_int(target["mutations"]),
            "threshold": int(gates_cfg["minimum_hrpos_her2neg_with_mutations"]),
            "pass": as_int(target["mutations"]) >= int(gates_cfg["minimum_hrpos_her2neg_with_mutations"]),
        },
        {
            "check": "HR+/HER2- complete RNA+CNA+mutations",
            "observed": as_int(target["rna_cna_mutations"]),
            "threshold": int(gates_cfg["minimum_hrpos_her2neg_complete_rna_cna_mutations"]),
            "pass": as_int(target["rna_cna_mutations"]) >= int(gates_cfg["minimum_hrpos_her2neg_complete_rna_cna_mutations"]),
        },
        {
            "check": "Cleaned RNA provenance",
            "observed": provenance["provenance_status"],
            "threshold": "CLEANED_RNA_NUMERICALLY_MATCHES_RAW_TRANSPOSED_VALUES",
            "pass": provenance["provenance_status"] == "CLEANED_RNA_NUMERICALLY_MATCHES_RAW_TRANSPOSED_VALUES",
        },
    ]

    all_pass = all(bool(r["pass"]) for r in checks)
    decision = (
        "METABRIC_READY_FOR_MULTIMODAL_TRANSPORT_PROTOCOL_LOCK"
        if all_pass
        else "METABRIC_REQUIRES_PREPROCESSING_OR_LINKAGE_FIX_BEFORE_TRANSPORT"
    )

    next_step = (
        "Freeze a METABRIC external-transport protocol: harmonize clinical variables, map shared genes, "
        "derive pathway-level RNA/CNA/mutation representations, define complete-case and modality-specific cohorts, "
        "and evaluate transport of modality utility without claiming an exact day-180 causal replication."
        if all_pass
        else
        "Resolve the failed coverage or RNA-provenance gates before constructing shared feature matrices."
    )

    result = {
        "metabric_m2_decision": decision,
        "exact_day180_causal_replication_allowed": False,
        "multimodal_transport_protocol_lock_allowed": all_pass,
        "recommended_next_step": next_step,
        "paper_a_role": (
            "Data-availability and design-transport limitation; no exact treatment-initiation replication."
        ),
        "paper_b_role": (
            "Independent same-disease, cross-platform validation of shared biological representations and modality utility."
        ),
    }

    write_csv(out / "m09_transport_readiness_checks.csv", checks)
    write_csv(out / "m09_transport_readiness_decision.csv", [result])

    report = [
        "# METABRIC M2 decision",
        "",
        f"**Decision:** `{decision}`",
        "",
        "## Non-negotiable boundary",
        "",
        "METABRIC contains hormone-therapy status but no initiation timing. The locked TCGA day-180 causal estimand is therefore not replicated.",
        "",
        "## Paper roles",
        "",
        f"- Paper A: {result['paper_a_role']}",
        f"- Paper B: {result['paper_b_role']}",
        "",
        "## Recommended next step",
        "",
        next_step,
    ]
    (out / "m09_transport_readiness_decision.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Decision: {decision}")
    print("\nTransport readiness checks")
    print_table(checks, ["check", "observed", "threshold", "pass"])

    print("\nPaper A role")
    print(f"  {result['paper_a_role']}")
    print("\nPaper B role")
    print(f"  {result['paper_b_role']}")
    print("\nRecommended next step")
    print(f"  {next_step}")

    print("\nPASS: METABRIC M2 decision generated. No treatment-effect estimate was calculated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
