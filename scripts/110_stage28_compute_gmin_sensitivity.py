from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from _stage25c_v10_utils import fit_propensity
from _stage28_v10_gmin_utils import (
    dataframe_console,
    expected_count_checks,
    load_json,
    project_root,
    stage26_like_config,
    verify_stage28_inputs,
    write_csv,
    write_json,
)
from _stage26_v10_utils import (
    aggregate_partition_scores,
    assemble_v10_frame,
    fit_partition,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage28_v10_gmin_sensitivity_config.json"
    )
    output = config["output"]

    print("=" * 128)
    print("STAGE 110 - COMPUTE CANDIDATE V10 G-MIN SENSITIVITY")
    print("=" * 128)
    print("Point estimates only. No bootstrap is run.")

    verify_stage28_inputs(root, config)
    manifest = load_json(
        root / output["calculation_manifest"]
    )
    if manifest["status"] != "STAGE28_GMIN_SENSITIVITY_LOCKED":
        raise RuntimeError("Stage 28 sensitivity is not locked.")

    for key in (
        "partition_table",
        "summary_table",
        "paired_differences",
        "diagnostics",
    ):
        path = root / output[key]
        if path.exists():
            raise RuntimeError(
                f"Stage 28 output already exists and will not be overwritten: {path}"
            )

    frame, features, metadata = assemble_v10_frame(
        root,
        {
            "source": {
                "v10_cohort": config["source"]["v10_cohort"],
                "v10_compact": config["source"]["v10_compact"],
            }
        },
    )
    checks = expected_count_checks(
        frame,
        features,
        config,
    )
    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Frozen V10 count checks failed.\n"
            + dataframe_console(checks)
        )

    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()
    observed_time = pd.to_numeric(
        frame["analysis_time"],
        errors="coerce",
    ).to_numpy(dtype=float)

    stage25c_config = load_json(
        root / config["source"]["stage25c_config"]
    )
    propensity, _, propensity_fit = fit_propensity(
        frame,
        treatment,
        features,
        stage25c_config,
    )

    partition_rows = []
    summary_rows = []

    for g_min in config["sensitivity"]["g_min_values"]:
        sensitivity_config = stage26_like_config(
            config,
            float(g_min),
        )
        patient_rows = []
        local_partition_rows = []

        print(
            "\n" + "-" * 128
            + f"\nG-min = {float(g_min):.2f}\n"
            + "-" * 128
        )

        for partition_number, base_seed in enumerate(
            config["estimator"]["partition_base_seeds"],
            start=1,
        ):
            row, patient = fit_partition(
                frame,
                features,
                treatment,
                event,
                observed_time,
                propensity,
                partition_number,
                int(base_seed),
                sensitivity_config,
            )
            row["g_min"] = float(g_min)
            local_partition_rows.append(row)
            patient_rows.append(patient)

            print(
                f"gmin={float(g_min):.2f} "
                f"partition={partition_number:02d} "
                f"effect={row['estimate_days']:+.6f} "
                f"plugin={row['plugin_component_days']:+.6f} "
                f"augmentation={row['total_residual_augmentation_days']:+.6f} "
                f"pseudo_p99={row['pseudo_p99']:.6f} "
                f"pseudo_max={row['pseudo_max']:.6f}"
            )

        local = pd.DataFrame(local_partition_rows)
        scores = pd.concat(patient_rows, ignore_index=True)
        aggregate, _ = aggregate_partition_scores(scores)
        effects = pd.to_numeric(
            local["estimate_days"],
            errors="raise",
        ).to_numpy(dtype=float)

        summary_rows.append({
            "g_min": float(g_min),
            "role": (
                "primary_reproduction"
                if abs(
                    float(g_min)
                    - float(
                        config["sensitivity"]["primary_value"]
                    )
                ) < 1e-12
                else "post_hoc_sensitivity"
            ),
            "n": len(frame),
            "treated": int(treatment.sum()),
            "control": int((1 - treatment).sum()),
            "events": int(event.sum()),
            "estimate_days": aggregate["estimate_days"],
            "diagnostic_if_se_days": aggregate["if_se_days"],
            "diagnostic_if_ci_low_days": aggregate[
                "if_ci_low_days"
            ],
            "diagnostic_if_ci_high_days": aggregate[
                "if_ci_high_days"
            ],
            "plugin_component_days": aggregate[
                "plugin_component_days"
            ],
            "treated_residual_component_days": aggregate[
                "treated_residual_component_days"
            ],
            "control_residual_component_days": aggregate[
                "control_residual_component_days"
            ],
            "total_residual_augmentation_days": aggregate[
                "total_residual_augmentation_days"
            ],
            "partition_mean_days": float(np.mean(effects)),
            "partition_median_days": float(np.median(effects)),
            "partition_sd_days": float(
                np.std(effects, ddof=1)
            ),
            "partition_mcse_days": float(
                np.std(effects, ddof=1)
                / math.sqrt(len(effects))
            ),
            "partition_min_days": float(np.min(effects)),
            "partition_max_days": float(np.max(effects)),
            "partition_range_days": float(
                np.max(effects) - np.min(effects)
            ),
            "minimum_raw_G": float(
                pd.to_numeric(
                    local["G_min_raw"],
                    errors="raise",
                ).min()
            ),
            "median_pseudo_p99": float(
                pd.to_numeric(
                    local["pseudo_p99"],
                    errors="raise",
                ).median()
            ),
            "maximum_pseudo_max": float(
                pd.to_numeric(
                    local["pseudo_max"],
                    errors="raise",
                ).max()
            ),
            "maximum_nuisance_retry": int(
                pd.to_numeric(
                    local["nuisance_retry"],
                    errors="raise",
                ).max()
            ),
        })
        partition_rows.extend(local_partition_rows)

    partitions = pd.DataFrame(partition_rows)
    summary = pd.DataFrame(summary_rows).sort_values(
        "g_min"
    ).reset_index(drop=True)

    primary_g = float(
        config["sensitivity"]["primary_value"]
    )
    primary_summary = summary[
        np.isclose(summary["g_min"], primary_g)
    ].iloc[0]
    observed_primary = float(
        primary_summary["estimate_days"]
    )
    expected_primary = float(
        config["expected"]["primary_point_estimate_days"]
    )
    tolerance = float(
        config["expected"]["reproduction_tolerance_days"]
    )
    if abs(observed_primary - expected_primary) > tolerance:
        raise RuntimeError(
            "G-min=0.10 did not reproduce the locked Stage 26 "
            f"point estimate: {observed_primary} != {expected_primary}"
        )

    primary_partitions = partitions[
        np.isclose(partitions["g_min"], primary_g)
    ][["partition", "estimate_days"]].rename(
        columns={"estimate_days": "primary_estimate_days"}
    )
    paired_rows = []
    for g_min in config["sensitivity"]["post_hoc_values"]:
        sensitivity_partitions = partitions[
            np.isclose(partitions["g_min"], float(g_min))
        ][["partition", "estimate_days"]].rename(
            columns={"estimate_days": "sensitivity_estimate_days"}
        )
        paired = primary_partitions.merge(
            sensitivity_partitions,
            on="partition",
            how="inner",
            validate="one_to_one",
        )
        paired["g_min"] = float(g_min)
        paired["difference_from_primary_days"] = (
            paired["sensitivity_estimate_days"]
            - paired["primary_estimate_days"]
        )
        paired_rows.append(paired)

    paired_differences = pd.concat(
        paired_rows,
        ignore_index=True,
    )

    summary["difference_from_primary_days"] = (
        summary["estimate_days"] - observed_primary
    )
    summary["relative_change_from_primary"] = (
        summary["difference_from_primary_days"]
        / abs(observed_primary)
    )

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        checks,
        table_dir / "s28_110_preflight_checks.csv",
    )
    write_csv(
        partitions,
        root / output["partition_table"],
    )
    write_csv(
        summary,
        root / output["summary_table"],
    )
    write_csv(
        paired_differences,
        root / output["paired_differences"],
    )

    write_json(
        {
            "sensitivity_id": manifest["sensitivity_id"],
            "v10_protocol_id": config["expected"][
                "v10_protocol_id"
            ],
            "cohort_metadata": metadata,
            "propensity_fit": propensity_fit,
            "g_min_values": config["sensitivity"][
                "g_min_values"
            ],
            "primary_reproduced_exactly": True,
            "primary_point_estimate_days": observed_primary,
            "summary": summary.to_dict("records"),
            "boundary": config["boundary"],
        },
        root / output["diagnostics"],
    )

    print("\nG-min sensitivity summary")
    print(dataframe_console(summary))
    print(
        "\nPASS: primary G-min=0.10 reproduced exactly and "
        "post hoc G-min sensitivities completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
