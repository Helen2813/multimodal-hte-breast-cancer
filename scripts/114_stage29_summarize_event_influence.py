from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage29_v10_event_influence_utils import (
    load_json,
    project_root,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage29_v10_event_influence_config.json"
    )
    output = config["output"]
    primary = float(
        config["expected"]["primary_point_estimate_days"]
    )

    print("=" * 128)
    print("STAGE 114 - SUMMARIZE EVENT-PATIENT INFLUENCE")
    print("=" * 128)

    checkpoint = pd.read_csv(
        root / output["checkpoint_local"],
        low_memory=False,
    )
    successful = checkpoint[
        checkpoint["success"].astype(str).str.lower().eq("true")
    ].copy()
    successful = (
        successful.sort_values("event_case_id")
        .drop_duplicates("event_case_id", keep="last")
    )

    required = int(
        config["expected"]["required_successful_deletions"]
    )
    if len(successful) != required:
        raise RuntimeError(
            f"Expected {required} successful deletions, "
            f"found {len(successful)}."
        )

    numeric_columns = [
        "estimate_days",
        "difference_from_primary_days",
        "absolute_change_days",
        "relative_change_from_primary",
        "partition_sd_days",
        "partition_mcse_days",
        "partition_min_days",
        "partition_max_days",
        "partition_range_days",
        "minimum_raw_G",
        "median_pseudo_p99",
        "maximum_pseudo_max",
    ]
    for column in numeric_columns:
        successful[column] = pd.to_numeric(
            successful[column],
            errors="raise",
        )

    public = successful[
        [
            "event_case_id",
            "omitted_arm",
            "omitted_analysis_time",
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
    ].copy()
    public = public.sort_values(
        "absolute_change_days",
        ascending=False,
    ).reset_index(drop=True)
    public.insert(
        0,
        "influence_rank",
        np.arange(1, len(public) + 1),
    )
    write_csv(public, root / output["public_results"])

    arm_rows = []
    for arm, group in public.groupby("omitted_arm"):
        estimates = group["estimate_days"].to_numpy(dtype=float)
        changes = group[
            "difference_from_primary_days"
        ].to_numpy(dtype=float)
        arm_rows.append({
            "omitted_arm": arm,
            "deletions": len(group),
            "minimum_estimate_days": float(np.min(estimates)),
            "median_estimate_days": float(np.median(estimates)),
            "maximum_estimate_days": float(np.max(estimates)),
            "minimum_change_days": float(np.min(changes)),
            "median_change_days": float(np.median(changes)),
            "maximum_change_days": float(np.max(changes)),
            "maximum_absolute_change_days": float(
                np.max(np.abs(changes))
            ),
            "fraction_estimates_positive": float(
                np.mean(estimates > 0)
            ),
            "sign_reversals": int(np.sum(estimates <= 0)),
        })
    arm_summary = pd.DataFrame(arm_rows)
    write_csv(arm_summary, root / output["arm_summary"])

    estimates = public["estimate_days"].to_numpy(dtype=float)
    changes = public[
        "difference_from_primary_days"
    ].to_numpy(dtype=float)

    large_change_days = float(
        config["reporting"]["large_change_days"]
    )
    large_relative_change = float(
        config["reporting"]["large_relative_change"]
    )

    final = {
        "status": "STAGE29_EVENT_INFLUENCE_COMPLETE",
        "influence_id": load_json(
            root / output["calculation_manifest"]
        )["influence_id"],
        "primary_point_estimate_days": primary,
        "event_deletions": len(public),
        "early_hormone_event_deletions": int(
            (public["omitted_arm"] == "early_hormone").sum()
        ),
        "control_event_deletions": int(
            (public["omitted_arm"] == "control").sum()
        ),
        "minimum_leave_one_event_out_estimate_days": float(
            np.min(estimates)
        ),
        "median_leave_one_event_out_estimate_days": float(
            np.median(estimates)
        ),
        "maximum_leave_one_event_out_estimate_days": float(
            np.max(estimates)
        ),
        "all_estimates_positive": bool((estimates > 0).all()),
        "sign_reversals": int(np.sum(estimates <= 0)),
        "maximum_absolute_change_days": float(
            np.max(np.abs(changes))
        ),
        "maximum_absolute_relative_change": float(
            np.max(
                np.abs(
                    public[
                        "relative_change_from_primary"
                    ].to_numpy(dtype=float)
                )
            )
        ),
        "deletions_with_absolute_change_over_20_days": int(
            np.sum(np.abs(changes) > large_change_days)
        ),
        "deletions_with_absolute_relative_change_over_30_percent": int(
            np.sum(
                np.abs(
                    public[
                        "relative_change_from_primary"
                    ].to_numpy(dtype=float)
                )
                > large_relative_change
            )
        ),
        "most_influential_public_record": public.iloc[0].to_dict(),
        "by_omitted_arm": arm_summary.to_dict("records"),
        "interpretation_boundary": config["boundary"],
    }
    write_json(final, root / output["final_summary"])

    figure_path = root / output["figure"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = public.sort_values(
        ["omitted_arm", "estimate_days"]
    ).reset_index(drop=True)
    x = np.arange(1, len(plot_data) + 1)
    plt.figure(figsize=(9.0, 5.0))

    for arm, marker in [
        ("control", "o"),
        ("early_hormone", "s"),
    ]:
        mask = plot_data["omitted_arm"].eq(arm)
        plt.scatter(
            x[mask],
            plot_data.loc[mask, "estimate_days"],
            marker=marker,
            label=arm,
        )

    plt.axhline(
        primary,
        linewidth=1.2,
        linestyle="--",
        label="Primary estimate",
    )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Event-patient deletion (anonymized order)")
    plt.ylabel("Candidate V10 ATO RMST difference (days)")
    plt.title("Leave-one-event-patient-out influence analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

    print("Leave-one-event-out results")
    print(
        public[
            [
                "influence_rank",
                "event_case_id",
                "omitted_arm",
                "estimate_days",
                "difference_from_primary_days",
                "absolute_change_days",
                "estimate_remains_positive",
            ]
        ].to_string(index=False)
    )
    print("\nSummary by omitted arm")
    print(arm_summary.to_string(index=False))
    print("\nFinal summary")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 29 event-patient influence analysis summarized."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
