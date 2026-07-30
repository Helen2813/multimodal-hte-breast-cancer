from __future__ import annotations

import pandas as pd
from _common import RESULTS_DIR, ensure_dirs, read_table


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    landmark = read_table(table_dir / "42_landmark_bootstrap_summary.csv").iloc[0]
    ccw = read_table(table_dir / "43_ccw_bootstrap_summary.csv").iloc[0]
    replication = read_table(table_dir / "41_landmark_replication_check.csv").iloc[0]
    ccw_point = read_table(table_dir / "41_ccw_point_estimate.csv").iloc[0]
    full = int(landmark["target_reps"]) >= 100 and int(ccw["target_reps"]) >= 100
    status = "FINAL_INFERENCE_COMPLETE" if full else "PILOT_INFERENCE_COMPLETE"
    lm_positive = landmark["percentile_ci_low_days"] > 0 if pd.notna(landmark["percentile_ci_low_days"]) else False
    ccw_positive = ccw["percentile_ci_low_days"] > 0 if pd.notna(ccw["percentile_ci_low_days"]) else False
    same_direction = replication["estimate_days"] * ccw_point["estimate_days"] > 0
    if lm_positive and ccw_positive and same_direction:
        interpretation = "Both analyses support a positive early-initiation signal, but observational data still do not establish efficacy."
    elif same_direction:
        interpretation = "Landmark and CCW analyses point in the same direction, but at least one interval includes zero; the paper remains a reliability/precision study."
    else:
        interpretation = "Landmark and CCW estimates disagree in direction; the paper must emphasize design sensitivity."
    lines = [
        "# Stage 12 inference report", "", f"**Status:** `{status}`", "",
        "## Primary landmark analysis", "",
        f"- Point estimate: {replication['estimate_days']:.1f} RMST days.",
        f"- Bootstrap repetitions: {int(landmark['successful_reps'])}/{int(landmark['target_reps'])}.",
        f"- Percentile interval: {landmark['percentile_ci_low_days']:.1f} to {landmark['percentile_ci_high_days']:.1f} days.",
        f"- Fraction positive: {landmark['fraction_positive']:.3f}.", "",
        "## Clone-censor-weight sensitivity", "",
        f"- Point estimate: {ccw_point['estimate_days']:.1f} RMST days through day 910 from diagnosis.",
        f"- Bootstrap repetitions: {int(ccw['successful_reps'])}/{int(ccw['target_reps'])}.",
        f"- Percentile interval: {ccw['percentile_ci_low_days']:.1f} to {ccw['percentile_ci_high_days']:.1f} days.",
        f"- Fraction positive: {ccw['fraction_positive']:.3f}.",
        f"- Median 99th-percentile clone weight: {ccw['weight_p99_median']:.2f}.", "",
        "## Interpretation gate", "", interpretation,
    ]
    report_path = table_dir / "44_stage12_inference_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = pd.DataFrame([{
        "status": status,
        "landmark_point_days": replication["estimate_days"],
        "landmark_ci_low": landmark["percentile_ci_low_days"],
        "landmark_ci_high": landmark["percentile_ci_high_days"],
        "ccw_point_days": ccw_point["estimate_days"],
        "ccw_ci_low": ccw["percentile_ci_low_days"],
        "ccw_ci_high": ccw["percentile_ci_high_days"],
        "same_direction": int(same_direction),
        "interpretation": interpretation,
    }])
    decision.to_csv(table_dir / "44_stage12_inference_decision.csv", index=False)
    print("=" * 118)
    print("STAGE 44 — STAGE 12 INFERENCE REPORT")
    print("=" * 118)
    print("\n".join(lines))
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
