from __future__ import annotations

import csv
import json

from _metabric_m3_utils import load_config, out_dir, print_table, project_root, write_csv


def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)

    ranked = read_rows(out / "m11_ranked_tcga_source_candidates.csv")
    audited = read_rows(out / "m12_candidate_identifier_overlap_audit.csv")

    modalities = [
        "clinical", "rna", "cna", "mutations", "methylation",
        "annotation", "pathway_gmt"
    ]
    template = {
        "status": "TEMPLATE_NOT_LOCKED",
        "scientific_role": {
            "paper_a": "design-transport limitation only; no exact METABRIC day-180 causal replication",
            "paper_b": "same-disease cross-platform validation of shared biological representations and modality utility"
        },
        "selected_sources": {},
        "harmonization_rules": {
            "clinical": [
                "age_at_diagnosis", "grade", "tumor_size",
                "positive_nodes", "ER", "PR", "HER2"
            ],
            "rna": "collapse duplicate symbols; within-cohort rank/normal-score transform; TCGA-only feature selection",
            "cna": "gene-symbol harmonization; signed or gain/loss representation; TCGA-only feature selection",
            "mutations": "binary gene-level indicators with panel-coverage awareness",
            "methylation": "optional gene/promoter modules only; no direct CpG identity requirement",
            "outcome_selection": "no METABRIC outcome may be inspected for source or feature selection"
        },
        "transport_tasks": {
            "primary": "cross-platform modality-utility transport for a common survival endpoint",
            "secondary": "exploratory ever-treated interaction analysis, clearly labelled noncausal",
            "forbidden": "claiming exact validation of TCGA day-180 treatment initiation"
        }
    }

    suggestions = []
    for modality in modalities:
        candidates = [r for r in ranked if r["modality"] == modality]
        top = candidates[0] if candidates else None
        template["selected_sources"][modality] = {
            "selected_path": "",
            "suggested_top_path": top["path"] if top else "",
            "suggested_top_score": int(top["score"]) if top else 0,
            "selection_requires_review": True,
        }
        suggestions.append({
            "modality": modality,
            "suggested_top_path": top["path"] if top else "",
            "suggested_top_score": top["score"] if top else "",
            "candidate_count": len(candidates),
        })

    path = out / "m13_m4_source_selection_TEMPLATE.json"
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    write_csv(out / "m13_source_suggestions.csv", suggestions)

    print("=" * 124)
    print("METABRIC M3.13 - M4 SOURCE-SELECTION TEMPLATE")
    print("=" * 124)
    print(f"Template: {path}")

    print("\nSuggested top source candidates")
    print_table(suggestions, ["modality", "suggested_top_path", "suggested_top_score", "candidate_count"])

    print("\nThe template is intentionally NOT locked.")
    print("No top-ranked source is accepted automatically merely because its filename or header scores well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
