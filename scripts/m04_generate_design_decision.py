from __future__ import annotations

import csv
import json
from pathlib import Path

from _metabric_m1_utils import load_config, out_dir, print_table, project_root, write_csv


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    clinical = json.loads((out / "m02_clinical_audit_summary.json").read_text(encoding="utf-8"))
    omics = json.loads((out / "m03_omics_audit_summary.json").read_text(encoding="utf-8"))
    cohorts = read_csv(out / "m02_candidate_cohort_counts.csv")

    fields = clinical["resolved_fields"]
    exact = bool(clinical["exact_day180_replication_preliminarily_feasible"])
    hormone_found = bool(fields.get("hormone_therapy"))
    survival_found = bool(fields.get("os_months") and fields.get("os_status"))
    receptor_found = bool(fields.get("er_status") and fields.get("her2_status"))

    adequacy_rows = []
    for row in cohorts:
        if row["receptor_definition"] != "hrpos_er_or_pr_her2neg" or row["therapy"] != "hormone_therapy":
            continue
        n = as_int(row["n"])
        events = as_int(row["events"])
        adequacy_rows.append({
            "group": row["group"],
            "n": n,
            "events": events,
            "n_gate": n >= int(cfg["minimum_group_n_for_future_modeling"]),
            "event_gate": events >= int(cfg["minimum_group_events_for_future_modeling"]),
        })

    group_gates_pass = len(adequacy_rows) == 2 and all(r["n_gate"] and r["event_gate"] for r in adequacy_rows)

    if exact and receptor_found and hormone_found and survival_found and group_gates_pass:
        decision = "EXACT_DAY180_EXTERNAL_REPLICATION_DESIGN_POSSIBLE_AFTER_TIMING_VALIDATION"
        next_step = (
            "Validate the semantics and units of the treatment-timing field, freeze a METABRIC-specific protocol, "
            "then run an external day-180 ATO-RMST replication."
        )
    elif receptor_found and hormone_found and survival_found:
        decision = "NO_EXACT_TREATMENT_INITIATION_TIMING_USE_METABRIC_FOR_TRANSPORT_AND_ASSOCIATION"
        next_step = (
            "Do not copy the TCGA day-180 causal estimand. Use METABRIC first for multimodal transport validation "
            "and, at most, a clearly labelled ever-treated observational association. Seek a cohort with treatment dates "
            "for exact causal external validation."
        )
    else:
        decision = "METABRIC_CLINICAL_DESIGN_INCOMPLETE_FOR_TREATMENT_EFFECT_REPLICATION"
        next_step = (
            "Resolve missing receptor, treatment, or survival fields before any treatment-effect analysis. "
            "Omics transport work may proceed only for patients with verified clinical linkage."
        )

    checks = [
        {"check": "ER and HER2 fields resolved", "pass": receptor_found, "observed": f"{fields.get('er_status')} | {fields.get('her2_status')}"},
        {"check": "Hormone therapy field resolved", "pass": hormone_found, "observed": fields.get("hormone_therapy") or ""},
        {"check": "OS time and event fields resolved", "pass": survival_found, "observed": f"{fields.get('os_months')} | {fields.get('os_status')}"},
        {"check": "Treatment timing field detected", "pass": exact, "observed": clinical["treatment_timing_candidate_count"]},
        {"check": "Future treated/control size and event gates", "pass": group_gates_pass, "observed": adequacy_rows},
        {"check": "At least one omics modality linked to clinical samples", "pass": bool(omics["omics_modalities_with_detected_clinical_sample_overlap"]), "observed": omics["omics_modalities_with_detected_clinical_sample_overlap"]},
    ]

    summary = {
        "metabric_m1_decision": decision,
        "exact_tcga_day180_estimand_may_be_copied": decision.startswith("EXACT_DAY180"),
        "recommended_next_step": next_step,
        "candidate_group_adequacy": adequacy_rows,
        "available_omics_modalities": omics["omics_modalities_with_detected_clinical_sample_overlap"],
        "important_boundary": (
            "A treatment indicator without initiation timing does not identify the same day-180 treatment strategy used in TCGA."
        ),
    }
    (out / "m04_metabric_design_decision.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out / "m04_metabric_design_checks.csv", checks)

    report = [
        "# METABRIC M1 data-and-design audit decision",
        "",
        f"**Decision:** `{decision}`",
        "",
        "## Design checks",
        "",
    ]
    for row in checks:
        report.append(f"- {'PASS' if row['pass'] else 'FAIL'} — {row['check']}: `{row['observed']}`")
    report.extend([
        "",
        "## Interpretation boundary",
        "",
        summary["important_boundary"],
        "",
        "## Recommended next step",
        "",
        next_step,
        "",
        "No treatment-effect estimate was calculated in this audit.",
    ])
    (out / "m04_metabric_design_decision.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("=" * 124)
    print("METABRIC M1.04 - DATA-AND-DESIGN DECISION")
    print("=" * 124)
    print(f"Decision: {decision}")

    print("\nDesign checks")
    print_table(checks, ["check", "pass", "observed"])

    print("\nCandidate group adequacy")
    print_table(adequacy_rows, ["group", "n", "events", "n_gate", "event_gate"])

    print("\nAvailable linked omics modalities")
    print(f"  {omics['omics_modalities_with_detected_clinical_sample_overlap']}")

    print("\nInterpretation boundary")
    print(f"  {summary['important_boundary']}")

    print("\nRecommended next step")
    print(f"  {next_step}")

    print("\nPASS: METABRIC M1 audit completed. No treatment-effect estimate was calculated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
