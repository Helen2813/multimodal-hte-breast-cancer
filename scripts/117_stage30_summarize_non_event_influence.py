from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage30_v10_non_event_influence_utils import (
    load_json,
    project_root,
    write_csv,
    write_json,
)


def summarize_group(
    label: str,
    frame: pd.DataFrame,
) -> dict:
    estimates = pd.to_numeric(
        frame["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    changes = pd.to_numeric(
        frame["difference_from_primary_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    return {
        "analysis_group": label,
        "deletions": len(frame),
        "minimum_estimate_days": float(np.min(estimates)),
        "median_estimate_days": float(np.median(estimates)),
        "maximum_estimate_days": float(np.max(estimates)),
        "maximum_absolute_change_days": float(
            np.max(np.abs(changes))
        ),
        "fraction_estimates_positive": float(
            np.mean(estimates > 0)
        ),
        "sign_reversals": int(np.sum(estimates <= 0)),
    }


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage30_v10_non_event_influence_config.json"
    )
    output = config["output"]
    primary = float(
        config["expected"]["primary_point_estimate_days"]
    )

    print("=" * 128)
    print("STAGE 117 - SUMMARIZE TARGETED NON-EVENT INFLUENCE")
    print("=" * 128)

    checkpoint = pd.read_csv(
        root / output["checkpoint_local"],
        low_memory=False,
    )
    non_event = (
        checkpoint[
            checkpoint["success"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]
        .sort_values("influence_rank")
        .drop_duplicates("influence_case_id", keep="last")
    )

    required = int(config["selection"]["top_k"])
    if len(non_event) != required:
        raise RuntimeError(
            f"Expected {required} successful non-event deletions, "
            f"found {len(non_event)}."
        )

    public_columns = [
        "influence_rank",
        "influence_case_id",
        "arm",
        "analysis_time",
        "original_influence",
        "original_absolute_influence",
        "original_normalized_contribution_days",
        "n",
        "treated",
        "control",
        "events",
        "treated_events",
        "control_events",
        "estimate_days",
        "difference_from_primary_days",
        "absolute_change_days",
        "relative_change_from_primary",
        "estimate_remains_positive",
        "diagnostic_if_se_days",
        "partition_sd_days",
        "partition_mcse_days",
        "partition_min_days",
        "partition_max_days",
        "partition_range_days",
        "minimum_raw_G",
        "median_pseudo_p99",
        "maximum_pseudo_max",
        "maximum_nuisance_retry",
    ]
    public = non_event[public_columns].copy()
    write_csv(public, root / output["public_results"])

    event_results = pd.read_csv(
        root / config["source"]["stage29_event_results"],
        low_memory=False,
    )
    if len(event_results) != int(
        config["expected"]["event_deletions_completed"]
    ):
        raise RuntimeError(
            "Stage 29 public event-influence table does not contain "
            "the expected 36 deletions."
        )

    event_results = event_results.copy()
    event_results["analysis_group"] = (
        "all_event_patients"
    )
    non_event_public = public.copy()
    non_event_public["analysis_group"] = (
        "top10_non_event_influence"
    )

    group_summary = pd.DataFrame([
        summarize_group(
            "all_event_patients",
            event_results,
        ),
        summarize_group(
            "top10_non_event_influence",
            non_event_public,
        ),
    ])

    combined_estimates = np.concatenate([
        pd.to_numeric(
            event_results["estimate_days"],
            errors="raise",
        ).to_numpy(dtype=float),
        pd.to_numeric(
            non_event_public["estimate_days"],
            errors="raise",
        ).to_numpy(dtype=float),
    ])
    combined_changes = combined_estimates - primary

    combined_row = {
        "analysis_group": "combined_targeted_deletions",
        "deletions": len(combined_estimates),
        "minimum_estimate_days": float(
            np.min(combined_estimates)
        ),
        "median_estimate_days": float(
            np.median(combined_estimates)
        ),
        "maximum_estimate_days": float(
            np.max(combined_estimates)
        ),
        "maximum_absolute_change_days": float(
            np.max(np.abs(combined_changes))
        ),
        "fraction_estimates_positive": float(
            np.mean(combined_estimates > 0)
        ),
        "sign_reversals": int(
            np.sum(combined_estimates <= 0)
        ),
    }
    group_summary = pd.concat(
        [
            group_summary,
            pd.DataFrame([combined_row]),
        ],
        ignore_index=True,
    )
    write_csv(
        group_summary,
        root / output["combined_summary_table"],
    )

    non_event_estimates = pd.to_numeric(
        public["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    non_event_changes = pd.to_numeric(
        public["difference_from_primary_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    relative_changes = pd.to_numeric(
        public["relative_change_from_primary"],
        errors="raise",
    ).to_numpy(dtype=float)

    final = {
        "status": "STAGE30_NON_EVENT_INFLUENCE_COMPLETE",
        "influence_id": load_json(
            root / output["calculation_manifest"]
        )["influence_id"],
        "primary_point_estimate_days": primary,
        "selected_non_event_patients": len(public),
        "minimum_non_event_leave_one_out_estimate_days": float(
            np.min(non_event_estimates)
        ),
        "median_non_event_leave_one_out_estimate_days": float(
            np.median(non_event_estimates)
        ),
        "maximum_non_event_leave_one_out_estimate_days": float(
            np.max(non_event_estimates)
        ),
        "all_non_event_estimates_positive": bool(
            (non_event_estimates > 0).all()
        ),
        "non_event_sign_reversals": int(
            np.sum(non_event_estimates <= 0)
        ),
        "maximum_non_event_absolute_change_days": float(
            np.max(np.abs(non_event_changes))
        ),
        "maximum_non_event_absolute_relative_change": float(
            np.max(np.abs(relative_changes))
        ),
        "non_event_deletions_over_20_days": int(
            np.sum(
                np.abs(non_event_changes)
                > float(
                    config["reporting"]["large_change_days"]
                )
            )
        ),
        "non_event_deletions_over_30_percent": int(
            np.sum(
                np.abs(relative_changes)
                > float(
                    config["reporting"][
                        "large_relative_change"
                    ]
                )
            )
        ),
        "combined_targeted_deletions": combined_row,
        "most_influential_non_event_public_record": (
            public.sort_values(
                "absolute_change_days",
                ascending=False,
            ).iloc[0].to_dict()
        ),
        "interpretation_boundary": config["boundary"],
    }
    write_json(final, root / output["final_summary"])

    figure_path = root / output["figure"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plot = public.sort_values("influence_rank")
    x = plot["influence_rank"].to_numpy(dtype=int)
    y = plot["estimate_days"].to_numpy(dtype=float)

    plt.figure(figsize=(8.0, 4.8))
    for arm, marker in [
        ("control", "o"),
        ("early_hormone", "s"),
    ]:
        mask = plot["arm"].eq(arm)
        plt.scatter(
            x[mask],
            y[mask],
            marker=marker,
            label=arm,
        )
    plt.axhline(
        primary,
        linestyle="--",
        linewidth=1.2,
        label="Primary estimate",
    )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Primary absolute-influence rank among non-events")
    plt.ylabel("Candidate V10 ATO RMST difference (days)")
    plt.title("Top-influence non-event leave-one-out analysis")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

    print("Top-influence non-event leave-one-out results")
    print(
        public[
            [
                "influence_rank",
                "influence_case_id",
                "arm",
                "original_normalized_contribution_days",
                "estimate_days",
                "difference_from_primary_days",
                "absolute_change_days",
                "estimate_remains_positive",
            ]
        ].to_string(index=False)
    )
    print("\nCombined targeted influence summary")
    print(group_summary.to_string(index=False))
    print("\nFinal summary")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 30 targeted non-event influence "
        "analysis summarized."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
