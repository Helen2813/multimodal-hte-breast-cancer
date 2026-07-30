from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import RESULTS_DIR, ensure_dirs, read_table


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    required = {
        "landmarks": table_dir / "29_landmark_cohort_summary.csv",
        "balance": table_dir / "30_landmark_balance_summary.csv",
        "paperA": table_dir / "31_landmark_paperA_gate.csv",
        "paperA_results": table_dir / "31_landmark_aipw_results.csv",
        "paperB_observed": table_dir / "32_paperB_landmark_observed_pilot.csv",
        "paperB_power": table_dir / "32_paperB_power_gate.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    data = {name: read_table(path) for name, path in required.items()}

    required_columns = {
        "paperA": {
            "cohort",
            "landmark_day",
            "horizon_days_post_landmark",
            "feasibility_status",
            "strategy_and_truncation_spread_days",
            "maximum_ci_halfwidth_days",
        },
        "paperB_observed": {
            "learner",
            "mean_fold_rloss_improvement",
            "positive_folds",
            "folds",
        },
        "paperB_power": {
            "learner",
            "null_detection_rate",
            "moderate_detection_rate",
            "strong_detection_rate",
            "power_status",
        },
    }
    for name, columns in required_columns.items():
        missing_columns = columns - set(data[name].columns)
        if missing_columns:
            raise ValueError(
                f"Stage 33 input '{name}' is missing columns: "
                f"{sorted(missing_columns)}"
            )

    lines = [
        "# Stage 9 two-paper decision report",
        "",
        "## Status",
        "",
        "Landmark and power feasibility review. Protocol remains **DRAFT_NOT_LOCKED**.",
        "",
        "## Paper A — landmark causal survival",
        "",
    ]

    hormone_gate = data["paperA"][
        data["paperA"]["cohort"]
        == "outer_hormone_hrpos_her2neg"
    ].sort_values(
        [
            "feasibility_status",
            "horizon_days_post_landmark",
            "landmark_day",
        ]
    )
    for _, r in hormone_gate.iterrows():
        lines.append(
            f"- Landmark {int(r['landmark_day'])} d, post-landmark horizon "
            f"{int(r['horizon_days_post_landmark'])} d: "
            f"`{r['feasibility_status']}`; strategy spread "
            f"{r['strategy_and_truncation_spread_days']:.1f} d; "
            f"max CI half-width {r['maximum_ci_halfwidth_days']:.1f} d."
        )

    best = hormone_gate[
        hormone_gate["feasibility_status"]
        == "LANDMARK_PAPER_A_FEASIBLE"
    ]
    if not best.empty:
        best_row = best.sort_values(
            [
                "horizon_days_post_landmark",
                "landmark_day",
            ]
        ).iloc[0]
        paper_a_decision = (
            f"Proceed using landmark {int(best_row['landmark_day'])} days "
            f"and {int(best_row['horizon_days_post_landmark'])}-day "
            "post-landmark RMST as the candidate primary design."
        )
    elif (
        hormone_gate["feasibility_status"]
        == "DIRECTION_STABLE_BUT_SENSITIVE"
    ).any():
        paper_a_decision = (
            "Proceed as a robustness/reliability paper, not an efficacy paper."
        )
    else:
        paper_a_decision = (
            "Do not lock Paper A until exposure timing or censoring design is revised."
        )

    lines += [
        "",
        f"**Paper A provisional decision:** {paper_a_decision}",
        "",
        "## Paper B — landmark multimodal causal AI",
        "",
    ]
    for _, r in data["paperB_observed"].iterrows():
        lines.append(
            f"- Observed {r['learner']}: mean R-loss improvement "
            f"{r['mean_fold_rloss_improvement']:.3f}; "
            f"positive folds {int(r['positive_folds'])}/{int(r['folds'])}."
        )
    for _, r in data["paperB_power"].iterrows():
        lines.append(
            f"- Power gate {r['learner']}: null "
            f"{r['null_detection_rate']:.2f}, moderate "
            f"{r['moderate_detection_rate']:.2f}, strong "
            f"{r['strong_detection_rate']:.2f}; "
            f"`{r['power_status']}`."
        )

    if (
        data["paperB_power"]["power_status"]
        == "POWER_ADEQUATE_FOR_MODERATE_SIGNAL"
    ).any():
        paper_b_decision = (
            "Proceed to the full simulation framework and prespecified modality omnibus."
        )
    elif (
        data["paperB_power"]["power_status"]
        == "POWER_ONLY_FOR_STRONG_SIGNAL"
    ).any():
        paper_b_decision = (
            "Proceed, but frame real-data null results as limited sensitivity to moderate HTE."
        )
    else:
        paper_b_decision = (
            "Keep Paper B simulation-first; do not expand to six modalities yet."
        )

    lines += [
        "",
        f"**Paper B provisional decision:** {paper_b_decision}",
        "",
        "## Required next action",
        "",
        "Do not lock either protocol automatically. Review the landmark choice, "
        "IPCW tail behaviour, AI/classical stability, and the simulation type-I error.",
    ]

    report = table_dir / "33_stage9_two_paper_decision.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 115)
    print("STAGE 33 — TWO-PAPER LANDMARK DECISION")
    print("=" * 115)
    print("\n".join(lines))
    print(f"\nSaved: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
