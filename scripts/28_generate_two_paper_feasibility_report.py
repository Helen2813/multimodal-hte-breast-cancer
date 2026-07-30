from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import RESULTS_DIR, ensure_dirs, read_table


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"

    required = {
        "balance": table_dir
        / "24_compact_era_propensity_summary.csv",
        "timing": table_dir / "24_treatment_timing_gate_summary.csv",
        "horizons": table_dir / "25_horizon_feasibility_gate.csv",
        "censoring": table_dir / "25_censoring_model_summary.csv",
        "paperA": table_dir / "26_paperA_ai_aipw_feasibility.csv",
        "paperA_gate": table_dir / "26_paperA_feasibility_gate.csv",
        "paperB": table_dir / "27_paperB_ai_feasibility_summary.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    data = {name: read_table(path) for name, path in required.items()}

    lines = [
        "# Two-paper feasibility report",
        "",
        "## Status",
        "",
        "Exploratory feasibility review. Protocol remains **DRAFT_NOT_LOCKED**.",
        "",
        "## Paper A — AI-assisted causal survival",
        "",
    ]

    hormone_balance = data["balance"][
        data["balance"]["cohort"]
        == "outer_hormone_hrpos_her2neg"
    ]
    hormone_gate = data["paperA_gate"][
        data["paperA_gate"]["cohort"]
        == "outer_hormone_hrpos_her2neg"
    ]
    lines.append("### Verified outer hormone cohort")
    lines.append("")
    if not hormone_balance.empty:
        r = hormone_balance.iloc[0]
        lines += [
            f"- n={int(r['n'])}; treated={int(r['treated'])}; "
            f"controls={int(r['control'])}; events={int(r['events'])}.",
            f"- Compact+era overlap max |SMD|={r['max_abs_smd_overlap']:.3f}; "
            f"control ESS={r['ess_control']:.1f}.",
            f"- Balance gate: `{r['balance_status']}`.",
        ]
    if not hormone_gate.empty:
        r = hormone_gate.iloc[0]
        lines += [
            f"- Three-year AIPW direction stability: "
            f"{bool(r['three_year_direction_stable'])}.",
            f"- Strategy spread: {r['three_year_strategy_spread_days']:.1f} days.",
            f"- Feasibility gate: `{r['paperA_feasibility_status']}`.",
        ]

    hormone_a = data["paperA"][
        data["paperA"]["cohort"]
        == "outer_hormone_hrpos_her2neg"
    ]
    if not hormone_a.empty:
        lines += ["", "AIPW-RMST pilot estimates:", ""]
        for _, r in hormone_a.iterrows():
            lines.append(
                f"- {r['strategy']}, {r['horizon_years']:.1f} y: "
                f"{r['aipw_ato_rmst_difference_days']:.1f} days "
                f"(IF 95% CI {r['influence_ci_low_days']:.1f} to "
                f"{r['influence_ci_high_days']:.1f})."
            )

    timing_h = data["timing"][
        data["timing"]["cohort"]
        == "outer_hormone_hrpos_her2neg"
    ]
    if not timing_h.empty:
        r = timing_h.iloc[0]
        lines += [
            "",
            f"Treatment-start coverage among treated patients: "
            f"{100*r['treated_start_coverage']:.1f}%; "
            f"within 180 days: {int(r['treated_start_0_180'])}/"
            f"{int(r['treated'])}.",
        ]

    lines += [
        "",
        "## Paper B — multimodal causal AI",
        "",
    ]
    for _, r in data["paperB"].iterrows():
        lines.append(
            f"- {r['learner']}, {r['horizon_days']/365.25:.1f} y: "
            f"prognostic positive repeats "
            f"{int(r['prognostic_positive_repeats'])}/"
            f"{int(r['repeats'])}; prescriptive positive repeats "
            f"{int(r['prescriptive_positive_repeats'])}/"
            f"{int(r['repeats'])}; status "
            f"`{r['paperB_pilot_status']}`."
        )

    statuses_a = set(data["paperA_gate"]["paperA_feasibility_status"])
    statuses_b = set(data["paperB"]["paperB_pilot_status"])
    if any("FEASIBLE" in x for x in statuses_a):
        paper_a_decision = "Proceed to a full-pipeline Paper A estimator."
    elif any("DIRECTION_STABLE" in x for x in statuses_a):
        paper_a_decision = (
            "Proceed, but frame Paper A around robustness and precision limits."
        )
    else:
        paper_a_decision = (
            "Do not lock an efficacy claim; redesign Paper A as a reliability study."
        )

    if "PROMISING_PRESCRIPTIVE_SIGNAL" in statuses_b:
        paper_b_decision = (
            "Proceed to simulation-first modality-utility development."
        )
    elif "PROGNOSTIC_ONLY_SIGNAL" in statuses_b:
        paper_b_decision = (
            "Proceed with prognosis-versus-prescription as the central null finding."
        )
    else:
        paper_b_decision = (
            "Run power simulations before expanding beyond clinical versus RNA."
        )

    lines += [
        "",
        "## Provisional decisions",
        "",
        f"- **Paper A:** {paper_a_decision}",
        f"- **Paper B:** {paper_b_decision}",
        "",
        "Neither protocol should be marked final yet.",
    ]

    report_path = table_dir / "28_two_paper_feasibility_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 110)
    print("STAGE 28 — TWO-PAPER FEASIBILITY DECISION REPORT")
    print("=" * 110)
    print("\n".join(lines))
    print(f"\nSaved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
