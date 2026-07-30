from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from _common import RESULTS_DIR, ensure_dirs, read_table
from _stage11_utils import (
    LANDMARK,
    assemble_landmark_table,
    latex_escape,
    readable_feature,
    standardized_mean_difference,
    weighted_mean,
)


def table1(landmark: pd.DataFrame, W_cols: list[str]) -> pd.DataFrame:
    a = pd.to_numeric(landmark["analysis_treatment"], errors="raise").astype(int).to_numpy()
    w = pd.to_numeric(landmark["overlap_weight"], errors="raise").to_numpy(float)
    rows = []
    for col in W_cols:
        x = pd.to_numeric(landmark[col], errors="coerce").to_numpy(float)
        rows.append(
            {
                "feature": col,
                "feature_label": readable_feature(col),
                "treated_unweighted_mean": float(np.nanmean(x[a == 1])) if np.isfinite(x[a == 1]).any() else np.nan,
                "control_unweighted_mean": float(np.nanmean(x[a == 0])) if np.isfinite(x[a == 0]).any() else np.nan,
                "treated_weighted_mean": weighted_mean(x[a == 1], w[a == 1]),
                "control_weighted_mean": weighted_mean(x[a == 0], w[a == 0]),
                "smd_unweighted": standardized_mean_difference(x, a),
                "smd_overlap_weighted": standardized_mean_difference(x, a, w),
                "missing_fraction": float(np.mean(~np.isfinite(x))),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_smd_unweighted"] = out["smd_unweighted"].abs()
    out["abs_smd_weighted"] = out["smd_overlap_weighted"].abs()
    return out.sort_values("abs_smd_unweighted", ascending=False)


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline characteristics before and after overlap weighting in the 180-day landmark cohort.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Feature & T raw & C raw & T weighted & C weighted & SMD raw & SMD weighted \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{latex_escape(row['feature_label'])} & "
            f"{row['treated_unweighted_mean']:.2f} & "
            f"{row['control_unweighted_mean']:.2f} & "
            f"{row['treated_weighted_mean']:.2f} & "
            f"{row['control_weighted_mean']:.2f} & "
            f"{row['smd_unweighted']:.3f} & "
            f"{row['smd_overlap_weighted']:.3f} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def love_plot(frame: pd.DataFrame, png_path: Path, svg_path: Path) -> None:
    plot = frame.sort_values("abs_smd_unweighted", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(plot))))
    ax.scatter(plot["smd_unweighted"], y, marker="o", label="Before weighting")
    ax.scatter(plot["smd_overlap_weighted"], y, marker="x", label="After overlap weighting")
    ax.axvline(0, linewidth=1)
    ax.axvline(0.1, linestyle="--", linewidth=1)
    ax.axvline(-0.1, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["feature_label"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate balance in the 180-day landmark cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def later_love_plot(balance: pd.DataFrame, png_path: Path, svg_path: Path) -> None:
    plot = balance.sort_values("abs_smd", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(plot))))
    ax.scatter(plot["smd_later_vs_never"], y, marker="o")
    ax.axvline(0, linewidth=1)
    ax.axvline(0.1, linestyle="--", linewidth=1)
    ax.axvline(-0.1, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["feature_label"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Later initiators versus no recorded later initiation")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def flow_figure(flow: pd.DataFrame, png_path: Path, svg_path: Path) -> None:
    values = dict(zip(flow["node"], flow["n"]))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (5, 9, f"Verified HR+/HER2− source cohort\nn={values['source']}"),
        (3, 7, f"Excluded before day 180\ndeath/censoring n={values['excluded_before_landmark']}"),
        (7, 7, f"Excluded ambiguous treatment timing\nn={values['excluded_ambiguous_timing']}"),
        (5, 5.5, f"Eligible at 180-day landmark\nn={values['eligible_landmark']}"),
        (2.5, 3.5, f"Initiated by day 180\nn={values['treated_by_landmark']}\nevents={values['events_treated']}"),
        (7.5, 3.5, f"Not initiated by day 180\nn={values['control_strategy']}\nevents={values['events_control']}"),
        (6.5, 1.5, f"Initiated after day 180\nn={values['later_initiators']}"),
        (8.5, 1.5, f"No recorded later initiation\nn={values['never_recorded_later']}"),
    ]

    for x, y, label in boxes:
        patch = FancyBboxPatch(
            (x - 1.35, y - 0.55),
            2.7,
            1.1,
            boxstyle="round,pad=0.03",
            fill=False,
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    arrows = [
        ((5, 8.45), (3, 7.55)),
        ((5, 8.45), (7, 7.55)),
        ((5, 8.45), (5, 6.05)),
        ((5, 4.95), (2.5, 4.05)),
        ((5, 4.95), (7.5, 4.05)),
        ((7.5, 2.95), (6.5, 2.05)),
        ((7.5, 2.95), (8.5, 2.05)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1})

    ax.set_title("Landmark cohort flow")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    figure_dir = RESULTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    landmark, W_cols, metadata = assemble_landmark_table()
    table = table1(landmark, W_cols)
    table.to_csv(table_dir / "38_table1_landmark180.csv", index=False)
    write_latex_table(table, table_dir / "38_table1_landmark180.tex")

    treatment = landmark["analysis_treatment"].astype(int)
    control = landmark.loc[treatment.eq(0)]
    later_n = int(
        pd.to_numeric(control["later_initiator"], errors="coerce")
        .fillna(0)
        .astype(int)
        .sum()
    )
    never_n = len(control) - later_n
    source_summary = read_table(RESULTS_DIR / "tables" / "29_landmark_cohort_summary.csv")
    row = source_summary[
        (source_summary["cohort"] == metadata["cohort"])
        & (source_summary["landmark_day"] == LANDMARK)
    ].iloc[0]
    flow = pd.DataFrame(
        [
            {"node": "source", "n": int(row["source_n"])},
            {"node": "excluded_before_landmark", "n": int(row["excluded_dead_or_censored_before_landmark"])},
            {"node": "excluded_ambiguous_timing", "n": int(row["excluded_ambiguous_treatment_timing"])},
            {"node": "eligible_landmark", "n": int(row["eligible_landmark_n"])},
            {"node": "treated_by_landmark", "n": int(row["treated_by_landmark"])},
            {"node": "control_strategy", "n": int(row["not_treated_by_landmark"])},
            {"node": "later_initiators", "n": later_n},
            {"node": "never_recorded_later", "n": never_n},
            {"node": "events_treated", "n": int(row["events_treated"])},
            {"node": "events_control", "n": int(row["events_control"])},
        ]
    )
    flow.to_csv(table_dir / "38_landmark_flow_counts.csv", index=False)

    love_plot(
        table,
        figure_dir / "38_love_plot_landmark180.png",
        figure_dir / "38_love_plot_landmark180.svg",
    )
    later_balance = read_table(table_dir / "37_later_vs_never_balance.csv")
    later_love_plot(
        later_balance,
        figure_dir / "38_later_vs_never_love_plot.png",
        figure_dir / "38_later_vs_never_love_plot.svg",
    )
    flow_figure(
        flow,
        figure_dir / "38_landmark_flow.png",
        figure_dir / "38_landmark_flow.svg",
    )

    print("=" * 115)
    print("STAGE 38 — REPORTING ASSETS")
    print("=" * 115)
    print(pd.DataFrame([metadata]).to_string(index=False))
    print("\nTable 1")
    print(
        table[
            [
                "feature_label",
                "treated_unweighted_mean",
                "control_unweighted_mean",
                "smd_unweighted",
                "smd_overlap_weighted",
            ]
        ].to_string(index=False)
    )
    print("\nFlow counts")
    print(flow.to_string(index=False))
    print(f"\nFigures saved under: {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
