from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from _stage26_v10_utils import (
    aggregate_partition_scores,
    assemble_v10_frame,
    compact_features,
    dataframe_console,
    fit_partition,
    fit_propensity,
    load_json,
    project_root,
    read_csv,
    stabilized_ato_components,
    verify_manifest,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage26_v10_point_estimate_config.json"
    )
    output = config["output"]
    expected = config["expected"]
    estimator = config["estimator"]

    print("=" * 128)
    print("STAGE 102 - COMPUTE LOCKED CANDIDATE V10 POINT ESTIMATE")
    print("=" * 128)
    print("No patient bootstrap is run.")

    calculation_manifest = load_json(
        root / output["calculation_manifest"]
    )
    if (
        calculation_manifest["status"]
        != "CANDIDATE_V10_POINT_ESTIMATE_CALCULATION_LOCKED"
    ):
        raise RuntimeError(
            "Stage 26 calculation is not locked."
        )
    verify_manifest(root, config)

    frame, features, metadata = assemble_v10_frame(
        root,
        config,
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

    checks = pd.DataFrame([
        {
            "check": "n",
            "observed": len(frame),
            "expected": expected["n"],
            "pass": len(frame) == int(expected["n"]),
        },
        {
            "check": "treated",
            "observed": int(treatment.sum()),
            "expected": expected["treated"],
            "pass": int(treatment.sum())
            == int(expected["treated"]),
        },
        {
            "check": "control",
            "observed": int((1 - treatment).sum()),
            "expected": expected["control"],
            "pass": int((1 - treatment).sum())
            == int(expected["control"]),
        },
        {
            "check": "events",
            "observed": int(event.sum()),
            "expected": expected["events"],
            "pass": int(event.sum())
            == int(expected["events"]),
        },
        {
            "check": "treated events",
            "observed": int(
                ((treatment == 1) & (event == 1)).sum()
            ),
            "expected": expected["treated_events"],
            "pass": int(
                ((treatment == 1) & (event == 1)).sum()
            )
            == int(expected["treated_events"]),
        },
        {
            "check": "control events",
            "observed": int(
                ((treatment == 0) & (event == 1)).sum()
            ),
            "expected": expected["control_events"],
            "pass": int(
                ((treatment == 0) & (event == 1)).sum()
            )
            == int(expected["control_events"]),
        },
        {
            "check": "features",
            "observed": len(features),
            "expected": expected["features"],
            "pass": len(features)
            == int(expected["features"]),
        },
        {
            "check": "partition seeds",
            "observed": len(
                estimator["partition_base_seeds"]
            ),
            "expected": expected["partitions"],
            "pass": len(
                estimator["partition_base_seeds"]
            )
            == int(expected["partitions"]),
        },
    ])
    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Candidate V10 point-estimate preflight failed.\n"
            + dataframe_console(checks)
        )

    stage25c_config = load_json(
        root / config["source"]["stage25c_config"]
    )
    propensity, propensity_coefficients, propensity_fit = (
        fit_propensity(
            frame,
            treatment,
            features,
            stage25c_config,
        )
    )

    partition_rows = []
    patient_rows = []
    for partition_number, base_seed in enumerate(
        estimator["partition_base_seeds"],
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
            config,
        )
        partition_rows.append(row)
        patient_rows.append(patient)

        print(
            f"partition={partition_number:02d} "
            f"effect={row['estimate_days']:+.6f} "
            f"IF_SE={row['if_se_days']:.6f} "
            f"G_min={row['G_min_raw']:.6f} "
            f"pseudo_p99={row['pseudo_p99']:.6f} "
            f"pseudo_max={row['pseudo_max']:.6f} "
            f"retry={row['nuisance_retry']}"
        )

    partitions = pd.DataFrame(partition_rows)
    scores = pd.concat(
        patient_rows,
        ignore_index=True,
    )
    aggregate, patients = aggregate_partition_scores(
        scores
    )

    partition_effects = pd.to_numeric(
        partitions["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    partition_sd = float(
        np.std(partition_effects, ddof=1)
    )
    partition_mcse = float(
        partition_sd / math.sqrt(len(partition_effects))
    )

    direct_effect_mean = float(
        pd.to_numeric(
            partitions["direct_ato_ipw_effect_days"],
            errors="raise",
        ).mean()
    )
    point_row = {
        "protocol_id": expected["protocol_id"],
        "calculation_id": calculation_manifest[
            "calculation_id"
        ],
        "n": len(frame),
        "treated": int(treatment.sum()),
        "control": int((1 - treatment).sum()),
        "events": int(event.sum()),
        "treated_events": int(
            ((treatment == 1) & (event == 1)).sum()
        ),
        "control_events": int(
            ((treatment == 0) & (event == 1)).sum()
        ),
        "features": len(features),
        "nuisance_partitions": len(partitions),
        "aipw_ato_rmst_difference_days": aggregate[
            "estimate_days"
        ],
        "diagnostic_if_se_days": aggregate[
            "if_se_days"
        ],
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
        "mean_partition_direct_ato_ipw_effect_days": (
            direct_effect_mean
        ),
        "partition_mean_effect_days": float(
            np.mean(partition_effects)
        ),
        "partition_median_effect_days": float(
            np.median(partition_effects)
        ),
        "partition_sd_effect_days": partition_sd,
        "partition_mcse_effect_days": partition_mcse,
        "partition_min_effect_days": float(
            np.min(partition_effects)
        ),
        "partition_max_effect_days": float(
            np.max(partition_effects)
        ),
        "partition_range_effect_days": float(
            np.max(partition_effects)
            - np.min(partition_effects)
        ),
        "minimum_G_min_raw": float(
            pd.to_numeric(
                partitions["G_min_raw"],
                errors="raise",
            ).min()
        ),
        "minimum_G_p01_raw": float(
            pd.to_numeric(
                partitions["G_p01_raw"],
                errors="raise",
            ).min()
        ),
        "median_pseudo_p99": float(
            pd.to_numeric(
                partitions["pseudo_p99"],
                errors="raise",
            ).median()
        ),
        "maximum_pseudo_max": float(
            pd.to_numeric(
                partitions["pseudo_max"],
                errors="raise",
            ).max()
        ),
        "maximum_nuisance_retry": int(
            pd.to_numeric(
                partitions["nuisance_retry"],
                errors="raise",
            ).max()
        ),
        "point_estimate_only": True,
        "primary_publication_interval_available": False,
    }

    table_dir = root / output["table_dir"]
    write_csv(checks, table_dir / "s26_102_preflight_checks.csv")
    write_csv(
        propensity_coefficients,
        table_dir / "s26_102_propensity_coefficients.csv",
    )
    write_csv(
        partitions,
        root / output["partition_table"],
    )
    write_csv(
        patients.sort_values(
            "absolute_influence",
            ascending=False,
        ),
        root / output["patient_scores"],
    )
    write_csv(
        pd.DataFrame([point_row]),
        root / output["point_estimate"],
    )
    write_json(
        {
            "protocol_id": expected["protocol_id"],
            "calculation_id": calculation_manifest[
                "calculation_id"
            ],
            "cohort_metadata": metadata,
            "propensity_fit": propensity_fit,
            "propensity_min": float(np.min(propensity)),
            "propensity_p01": float(
                np.quantile(propensity, 0.01)
            ),
            "propensity_p99": float(
                np.quantile(propensity, 0.99)
            ),
            "propensity_max": float(np.max(propensity)),
            "point_estimate": point_row,
            "top_absolute_influences": (
                patients.sort_values(
                    "absolute_influence",
                    ascending=False,
                )
                .head(10)[
                    [
                        "row_index",
                        "patient_id_normalized",
                        "influence",
                        "absolute_influence",
                    ]
                ]
                .to_dict("records")
            ),
            "boundary": config["boundary"],
        },
        root / output["diagnostics"],
    )

    print("\nCandidate V10 partition estimates")
    print(
        dataframe_console(
            partitions[
                [
                    "partition",
                    "estimate_days",
                    "if_se_days",
                    "plugin_component_days",
                    "treated_residual_component_days",
                    "control_residual_component_days",
                    "G_min_raw",
                    "pseudo_p99",
                    "pseudo_max",
                    "nuisance_retry",
                ]
            ],
            max_rows=30,
        )
    )
    print("\nCandidate V10 locked point estimate")
    print(
        dataframe_console(
            pd.DataFrame([point_row])
        )
    )
    print(
        "\nPASS: Candidate V10 point estimate computed. "
        "No patient bootstrap was run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
