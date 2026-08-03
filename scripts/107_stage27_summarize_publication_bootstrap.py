from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from _stage27_v10_bootstrap_utils import (
    dataframe_console,
    load_json,
    locked_bootstrap_settings,
    project_root,
    write_csv,
    write_json,
)


def interval_rows(
    values: np.ndarray,
    point: float,
) -> list[dict]:
    q_low = float(np.quantile(values, 0.025))
    q_high = float(np.quantile(values, 0.975))
    sd = float(np.std(values, ddof=1))
    return [
        {
            "interval": "percentile",
            "ci_low_days": q_low,
            "ci_high_days": q_high,
            "primary": True,
        },
        {
            "interval": "basic",
            "ci_low_days": 2.0 * point - q_high,
            "ci_high_days": 2.0 * point - q_low,
            "primary": False,
        },
        {
            "interval": "normal",
            "ci_low_days": point - 1.96 * sd,
            "ci_high_days": point + 1.96 * sd,
            "primary": False,
        },
    ]


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage27_v10_bootstrap_config.json"
    )
    output = config["output"]
    locked_bootstrap = locked_bootstrap_settings(
        root,
        config,
    )
    expected_reps = int(locked_bootstrap["repetitions"])
    point = float(
        config["expected"]["point_estimate_days"]
    )

    print("=" * 128)
    print("STAGE 107 - SUMMARIZE CANDIDATE V10 PUBLICATION BOOTSTRAP")
    print("=" * 128)

    checkpoint = pd.read_csv(
        root / output["checkpoint"],
        low_memory=False,
    )
    successful = checkpoint[
        checkpoint["success"].astype(str).str.lower()
        == "true"
    ].copy()
    successful["repetition"] = pd.to_numeric(
        successful["repetition"],
        errors="raise",
    ).astype(int)
    successful = (
        successful.sort_values("repetition")
        .drop_duplicates("repetition", keep="last")
    )

    if len(successful) != expected_reps:
        raise RuntimeError(
            f"Expected {expected_reps} successful bootstrap repetitions, "
            f"found {len(successful)}."
        )

    values = pd.to_numeric(
        successful["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    intervals = pd.DataFrame(
        interval_rows(values, point)
    )

    percentile = intervals[
        intervals["interval"] == "percentile"
    ].iloc[0]
    ci_low = float(percentile["ci_low_days"])
    ci_high = float(percentile["ci_high_days"])

    if ci_low > 0:
        direction_status = (
            "POSITIVE_DIRECTION_INTERVAL_EXCLUDES_ZERO"
        )
        wording = config["reporting"]["primary_claim_rule"][
            "ci_low_above_zero"
        ]
    elif ci_high < 0:
        direction_status = (
            "NEGATIVE_DIRECTION_INTERVAL_EXCLUDES_ZERO"
        )
        wording = config["reporting"]["primary_claim_rule"][
            "ci_high_below_zero"
        ]
    else:
        direction_status = "INTERVAL_INCLUDES_ZERO"
        wording = config["reporting"]["primary_claim_rule"][
            "ci_includes_zero"
        ]

    summary_rows = [
        {
            "protocol_id": config["expected"]["protocol_id"],
            "bootstrap_repetitions": len(values),
            "point_estimate_days": point,
            "bootstrap_mean_days": float(np.mean(values)),
            "bootstrap_median_days": float(np.median(values)),
            "bootstrap_sd_days": float(np.std(values, ddof=1)),
            "bootstrap_mean_mcse_days": float(
                np.std(values, ddof=1) / math.sqrt(len(values))
            ),
            "minimum_days": float(np.min(values)),
            "maximum_days": float(np.max(values)),
            "fraction_positive": float(np.mean(values > 0)),
            "fraction_negative": float(np.mean(values < 0)),
            "fraction_zero": float(np.mean(values == 0)),
            "percentile_ci_low_days": ci_low,
            "percentile_ci_high_days": ci_high,
            "direction_status": direction_status,
        }
    ]
    summary = pd.DataFrame(summary_rows)

    tail_low = float(config["reporting"]["tail_low"])
    tail_high = float(config["reporting"]["tail_high"])
    tail_threshold = float(
        config["reporting"][
            "design_tail_fraction_threshold"
        ]
    )
    balance_threshold = float(
        config["reporting"][
            "design_balance_threshold"
        ]
    )

    diagnostics = {
        "protocol_id": config["expected"]["protocol_id"],
        "repetitions": len(successful),
        "unique_source_patients": {
            "minimum": int(
                pd.to_numeric(
                    successful["unique_source_patients"],
                    errors="raise",
                ).min()
            ),
            "median": float(
                pd.to_numeric(
                    successful["unique_source_patients"],
                    errors="raise",
                ).median()
            ),
            "maximum": int(
                pd.to_numeric(
                    successful["unique_source_patients"],
                    errors="raise",
                ).max()
            ),
        },
        "events": {
            "minimum": int(
                pd.to_numeric(
                    successful["events"],
                    errors="raise",
                ).min()
            ),
            "median": float(
                pd.to_numeric(
                    successful["events"],
                    errors="raise",
                ).median()
            ),
            "maximum": int(
                pd.to_numeric(
                    successful["events"],
                    errors="raise",
                ).max()
            ),
            "minimum_treated_events": int(
                pd.to_numeric(
                    successful["treated_events"],
                    errors="raise",
                ).min()
            ),
            "minimum_control_events": int(
                pd.to_numeric(
                    successful["control_events"],
                    errors="raise",
                ).min()
            ),
        },
        "propensity": {
            "fraction_repetitions_with_tail_fraction_above_design_gate": float(
                np.mean(
                    (
                        pd.to_numeric(
                            successful[
                                "fraction_propensity_below_0_01"
                            ],
                            errors="raise",
                        )
                        > tail_threshold
                    )
                    |
                    (
                        pd.to_numeric(
                            successful[
                                "fraction_propensity_above_0_99"
                            ],
                            errors="raise",
                        )
                        > tail_threshold
                    )
                )
            ),
            "minimum_p01": float(
                pd.to_numeric(
                    successful["propensity_p01"],
                    errors="raise",
                ).min()
            ),
            "maximum_p99": float(
                pd.to_numeric(
                    successful["propensity_p99"],
                    errors="raise",
                ).max()
            ),
            "maximum_absolute_coefficient": float(
                pd.to_numeric(
                    successful[
                        "maximum_absolute_propensity_coefficient"
                    ],
                    errors="raise",
                ).max()
            ),
            "minimum_treated_ess_fraction": float(
                pd.to_numeric(
                    successful[
                        "ato_ess_fraction_treated"
                    ],
                    errors="raise",
                ).min()
            ),
            "minimum_control_ess_fraction": float(
                pd.to_numeric(
                    successful[
                        "ato_ess_fraction_control"
                    ],
                    errors="raise",
                ).min()
            ),
            "minimum_normalized_overlap_mass": float(
                pd.to_numeric(
                    successful[
                        "normalized_overlap_mass"
                    ],
                    errors="raise",
                ).min()
            ),
            "fraction_repetitions_balance_above_design_gate": float(
                np.mean(
                    pd.to_numeric(
                        successful[
                            "max_abs_ato_weighted_smd"
                        ],
                        errors="raise",
                    )
                    > balance_threshold
                )
            ),
        },
        "nuisance": {
            "minimum_G_min_raw": float(
                pd.to_numeric(
                    successful["minimum_G_min_raw"],
                    errors="raise",
                ).min()
            ),
            "maximum_pseudo_max": float(
                pd.to_numeric(
                    successful["maximum_pseudo_max"],
                    errors="raise",
                ).max()
            ),
            "maximum_nuisance_retry": int(
                pd.to_numeric(
                    successful["maximum_nuisance_retry"],
                    errors="raise",
                ).max()
            ),
            "maximum_partition_mcse_days": float(
                pd.to_numeric(
                    successful[
                        "partition_mcse_effect_days"
                    ],
                    errors="raise",
                ).max()
            ),
        },
        "bootstrap_distribution": {
            "skewness": float(
                stats.skew(values, bias=False)
            ),
            "excess_kurtosis": float(
                stats.kurtosis(
                    values,
                    fisher=True,
                    bias=False,
                )
            ),
        },
    }

    decision = {
        "status": (
            "CANDIDATE_V10_PUBLICATION_BOOTSTRAP_COMPLETE"
        ),
        "protocol_id": config["expected"]["protocol_id"],
        "point_estimate_days": point,
        "primary_percentile_interval_days": [
            ci_low,
            ci_high,
        ],
        "fraction_bootstrap_estimates_positive": float(
            np.mean(values > 0)
        ),
        "direction_status": direction_status,
        "authorized_wording": wording,
        "candidate_v10_role": (
            "Sequencing-aware amended primary analysis. "
            "Candidate V9 remains the immutable original full-cohort "
            "analysis and should be reported transparently as a "
            "sequencing-confounded secondary contrast."
        ),
        "interpretation_boundary": (
            "Observational overlap-population estimate among patients "
            "without documented chemotherapy initiation during the "
            "day-180 grace period. No randomized-treatment or individual "
            "treatment recommendation claim is authorized."
        ),
        "next_action": (
            "Review the numerical results, then prepare publication tables "
            "and manually write the manuscript. No additional estimator "
            "changes are authorized based on the bootstrap result."
        ),
    }

    write_csv(intervals, root / output["table_dir"] / "s27_107_interval_table.csv")
    write_csv(summary, root / output["summary"])
    write_json(diagnostics, root / output["diagnostics"])
    write_json(decision, root / output["decision"])

    print("Bootstrap summary")
    print(dataframe_console(summary))
    print("\nInterval table")
    print(dataframe_console(intervals))
    print("\nBootstrap diagnostics")
    print(json.dumps(diagnostics, indent=2))
    print("\nPublication decision")
    print(json.dumps(decision, indent=2))
    print(
        "\nPASS: Candidate V10 publication bootstrap summarized."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
