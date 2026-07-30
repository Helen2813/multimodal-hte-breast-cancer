from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import PROJECT_ROOT, RESULTS_DIR, ensure_dirs, read_table


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    paper_a = read_table(
        table_dir / "34_paperA_candidate_summary.csv"
    )
    observed_b = read_table(
        table_dir / "35_paperB_repaired_observed_pilot.csv"
    )
    power_b = read_table(
        table_dir / "35_paperB_repaired_power_gate.csv"
    )

    require_columns(
        paper_a,
        {
            "primary_effect_days",
            "if_ci_low",
            "if_ci_high",
            "max_abs_smd",
            "status",
        },
        "Paper A summary",
    )
    require_columns(
        observed_b,
        {
            "learner",
            "mean_rloss_improvement",
            "paired_patient_p_value",
            "prescriptive_signal_detected",
        },
        "Paper B observed",
    )
    require_columns(
        power_b,
        {
            "learner",
            "null_detection_rate",
            "minimum_signal_sd_for_60pct_power",
            "power_status",
        },
        "Paper B power gate",
    )

    a = paper_a.iloc[0]
    lines = [
        "# Stage 10 decision report",
        "",
        "## Paper A",
        "",
        f"- Candidate primary effect: {a['primary_effect_days']:.1f} "
        f"RMST days.",
        f"- Diagnostic interval: {a['if_ci_low']:.1f} to "
        f"{a['if_ci_high']:.1f} days.",
        f"- Maximum weighted SMD: {a['max_abs_smd']:.3f}.",
        f"- Feasibility status: `{a['status']}`.",
        "",
        "Paper A should proceed as an AI-assisted reliability and robustness "
        "study of early treatment initiation. The primary classical nuisance "
        "specification is retained because boosted-AI nuisance models produced "
        "less stable censoring weights and wider uncertainty.",
        "",
        "## Paper B",
        "",
    ]

    for _, row in observed_b.iterrows():
        lines.append(
            f"- Observed {row['learner']}: mean paired R-loss improvement "
            f"{row['mean_rloss_improvement']:.1f}; p="
            f"{row['paired_patient_p_value']:.4f}; detected="
            f"{bool(row['prescriptive_signal_detected'])}."
        )

    for _, row in power_b.iterrows():
        threshold = row[
            "minimum_signal_sd_for_60pct_power"
        ]
        threshold_text = (
            f"{threshold:.0f} days"
            if pd.notna(threshold)
            else "not reached"
        )
        lines.append(
            f"- Power {row['learner']}: null detection "
            f"{row['null_detection_rate']:.2f}; minimum signal for "
            f"60% power {threshold_text}; "
            f"`{row['power_status']}`."
        )

    if (
        power_b["power_status"]
        == "ADEQUATE_FOR_100D_SIGNAL"
    ).any():
        b_decision = (
            "Proceed to the full simulation framework and then a "
            "prespecified clinical-versus-RNA analysis."
        )
    elif (
        power_b["power_status"]
        == "ONLY_LARGE_SIGNALS_DETECTABLE"
    ).any():
        b_decision = (
            "Proceed as a simulation-first methods paper, explicitly "
            "showing that the current TCGA application can detect only "
            "large prescriptive signals."
        )
    else:
        b_decision = (
            "Do not claim a prescriptive RNA null and do not expand to "
            "six modalities. Redesign the learner or add information "
            "before real-data modality attribution."
        )

    lines += [
        "",
        f"**Paper B decision:** {b_decision}",
        "",
        "## Protocol status",
        "",
        "- Paper A: candidate design ready for professor review; not yet locked.",
        "- Paper B: simulation framework still under methodological development.",
    ]

    report = table_dir / "36_stage10_decision.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 115)
    print("STAGE 36 — STAGE 10 DECISION")
    print("=" * 115)
    print("\n".join(lines))
    print(f"\nSaved: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
