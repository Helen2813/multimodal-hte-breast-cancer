from __future__ import annotations

import json

import pandas as pd

from _stage32_dgm_calibration_utils import (
    empirical_target_values,
    evaluate_parameter_set,
    load_json,
    optimize_outcome,
    optimize_treatment,
    project_root,
    sobol_baseline,
    treatment_metrics,
    write_csv,
    write_json,
)


def metric_rows(
    label: str,
    observed: dict[str, float],
    targets: dict[str, float],
    tolerances: dict[str, float],
) -> list[dict]:
    rows = []
    for metric, target in targets.items():
        if metric not in observed:
            continue
        value = float(observed[metric])
        tolerance = float(tolerances[metric])
        rows.append({
            "evaluation_set": label,
            "metric": metric,
            "target": float(target),
            "observed": value,
            "difference": value - float(target),
            "tolerance": tolerance,
            "standardized_residual": (
                value - float(target)
            ) / tolerance,
        })
    return rows


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage32_sequence_dgm_calibration_config.json"
    )
    output = config["outputs"]
    manifest = load_json(root / output["manifest"])

    print("=" * 128)
    print("STAGE 122 - RUN EMPIRICAL DGM CALIBRATION")
    print("=" * 128)

    if manifest["status"] != "STAGE32_DGM_CALIBRATION_LOCKED":
        raise RuntimeError("Stage 32 calibration is not locked.")

    for key in (
        "metrics_table",
        "optimization_runs",
        "recommended_dgm",
        "final_summary",
    ):
        path = root / output[key]
        if path.exists():
            raise RuntimeError(
                f"Stage 32 output already exists and will not be overwritten: {path}"
            )

    training = sobol_baseline(
        int(config["calibration"]["training_sobol_power"]),
        int(config["calibration"]["training_seed"]),
        config,
    )
    validation = sobol_baseline(
        int(config["calibration"]["validation_sobol_power"]),
        int(config["calibration"]["validation_seed"]),
        config,
    )

    print(
        f"Training Sobol sample: {len(training):,}; "
        f"validation sample: {len(validation):,}"
    )

    treatment_parameters, treatment_runs = optimize_treatment(
        training,
        config,
    )
    _, treatment_arrays = treatment_metrics(
        training,
        treatment_parameters["chemo_logit_intercept"],
        treatment_parameters["treatment_logit_intercept"],
        treatment_parameters["sequencing_strength"],
        config,
    )
    outcome_parameters, outcome_runs = optimize_outcome(
        training,
        treatment_arrays,
        config,
    )

    calibrated = {
        **treatment_parameters,
        **outcome_parameters,
    }
    defaults = dict(
        config["pilot_defaults_for_comparison"]
    )

    target = empirical_target_values(config)
    tolerances = config["target_tolerances"]

    evaluations = {
        "stage31_default_training": evaluate_parameter_set(
            training,
            defaults,
            config,
        ),
        "calibrated_training": evaluate_parameter_set(
            training,
            calibrated,
            config,
        ),
        "calibrated_validation": evaluate_parameter_set(
            validation,
            calibrated,
            config,
        ),
    }

    rows = []
    for label, observed in evaluations.items():
        rows.extend(
            metric_rows(
                label,
                observed,
                target,
                tolerances,
            )
        )
    metrics = pd.DataFrame(rows)
    write_csv(metrics, root / output["metrics_table"])

    optimization_runs = pd.concat(
        [treatment_runs, outcome_runs],
        ignore_index=True,
        sort=False,
    )
    write_csv(
        optimization_runs,
        root / output["optimization_runs"],
    )

    recommended = {
        "calibration_id": manifest["calibration_id"],
        "calibrated_parameters": calibrated,
        "fixed_covariate_parameters": config[
            "calibration"
        ]["fixed_covariate_parameters"],
        "censoring_parameters": config[
            "calibration"
        ]["censoring_parameters"],
        "intervals": config["calibration"]["intervals"],
        "interval_days": config["calibration"]["interval_days"],
        "empirical_targets": target,
        "training_metrics": evaluations[
            "calibrated_training"
        ],
        "validation_metrics": evaluations[
            "calibrated_validation"
        ],
        "confirmatory_scenarios": {
            "sample_sizes": config[
                "recommended_confirmatory_design"
            ]["sample_sizes"],
            "sequencing_strengths": {
                "none": 0.0,
                "half_empirical": (
                    calibrated["sequencing_strength"] / 2.0
                ),
                "empirical": calibrated[
                    "sequencing_strength"
                ],
            },
            "treatment_effect_regimes": {
                "null": 0.0,
                "empirically_calibrated_benefit": calibrated[
                    "true_treatment_log_hazard_effect"
                ],
            },
            "minimum_repetitions": config[
                "recommended_confirmatory_design"
            ]["minimum_repetitions"],
            "preferred_repetitions": config[
                "recommended_confirmatory_design"
            ]["preferred_repetitions"],
        },
        "required_additional_diagnostics": config[
            "recommended_confirmatory_design"
        ]["required_additional_diagnostics"],
        "boundary": config["boundary"],
    }
    write_json(recommended, root / output["recommended_dgm"])

    print("\nCalibrated parameters")
    print(json.dumps(calibrated, indent=2))
    print("\nTarget reproduction")
    print(metrics.to_string(index=False))
    print(
        "\nPASS: Stage 32 calibration completed. "
        "Stage 123 will apply locked decision rules."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
