from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from _stage31_simulation_utils import (
    ato_truth,
    estimate_method,
    load_json,
    project_root,
    simulate_dataset,
    write_csv,
)


def append_row(
    row: dict,
    path,
) -> None:
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
        mask = ~(
            (frame["scenario_id"].astype(str) == str(row["scenario_id"]))
            & (
                pd.to_numeric(
                    frame["repetition"],
                    errors="coerce",
                )
                == int(row["repetition"])
            )
            & (frame["method"].astype(str) == str(row["method"]))
        )
        frame = pd.concat(
            [frame[mask], pd.DataFrame([row])],
            ignore_index=True,
        )
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values(
        ["scenario_id", "repetition", "method"]
    ).reset_index(drop=True)
    write_csv(frame, path)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage31_sequencing_simulation_config.json"
    )
    output = config["outputs"]
    manifest = load_json(root / output["manifest"])

    print("=" * 128)
    print("STAGE 119 - RUN FOCUSED SEQUENCING SIMULATION PILOT")
    print("=" * 128)

    if manifest["status"] != "STAGE31_SIMULATION_PILOT_LOCKED":
        raise RuntimeError("Stage 31 simulation is not locked.")

    checkpoint_path = root / output["checkpoint"]
    errors_path = root / output["errors"]

    if checkpoint_path.exists():
        checkpoint = pd.read_csv(
            checkpoint_path,
            low_memory=False,
        )
        successful_checkpoint = checkpoint[
            checkpoint["success"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]
        completed = set(
            zip(
                successful_checkpoint["scenario_id"].astype(str),
                pd.to_numeric(
                    successful_checkpoint["repetition"],
                    errors="coerce",
                ).astype(int),
                successful_checkpoint["method"].astype(str),
            )
        )
    else:
        completed = set()

    errors = []
    if errors_path.exists():
        errors = pd.read_csv(
            errors_path,
            low_memory=False,
        ).to_dict("records")

    repetitions = int(
        config["simulation"]["repetitions_per_scenario"]
    )
    base_seed = int(config["simulation"]["base_seed"])
    estimator_specs = config["estimators"]

    scenario_counter = 0
    for sample_size in config["simulation"]["sample_sizes"]:
        for strength_name, strength_value in config[
            "simulation"
        ]["sequencing_strengths"].items():
            scenario_counter += 1
            scenario_id = (
                f"N{int(sample_size)}_{strength_name.upper()}"
            )
            print(
                "\n" + "-" * 128
                + f"\nScenario {scenario_id}: "
                f"sequencing_strength={strength_value}\n"
                + "-" * 128
            )

            for repetition in range(1, repetitions + 1):
                seed = (
                    base_seed
                    + scenario_counter * 100_000
                    + repetition
                )
                frame = simulate_dataset(
                    int(sample_size),
                    float(strength_value),
                    seed,
                    config,
                )
                full_truth = ato_truth(frame)
                no_chemo_frame = frame[
                    frame["chemo_by_day180"] == 0
                ].reset_index(drop=True)
                no_chemo_truth = ato_truth(no_chemo_frame)

                method_frames = {
                    "naive_full": frame,
                    "adjusted_full": frame,
                    "sequencing_aware": no_chemo_frame,
                }
                truths = {
                    "naive_full": full_truth,
                    "adjusted_full": full_truth,
                    "sequencing_aware": no_chemo_truth,
                }

                for method_name, method_spec in estimator_specs.items():
                    key = (
                        scenario_id,
                        repetition,
                        method_name,
                    )
                    if key in completed:
                        continue

                    try:
                        analysis_frame = method_frames[
                            method_name
                        ].copy()
                        method_seed_offset = {
                            "naive_full": 101,
                            "adjusted_full": 202,
                            "sequencing_aware": 303,
                        }[method_name]
                        result = estimate_method(
                            analysis_frame,
                            list(method_spec["covariates"]),
                            float(truths[method_name]),
                            seed + method_seed_offset,
                            config,
                        )
                        row = {
                            "scenario_id": scenario_id,
                            "sample_size": int(sample_size),
                            "sequencing_strength_name": strength_name,
                            "sequencing_strength": float(strength_value),
                            "repetition": repetition,
                            "seed": seed,
                            "method": method_name,
                            "success": True,
                            "error_type": "",
                            "error_message": "",
                            "chemo_prevalence": float(
                                frame["chemo_by_day180"].mean()
                            ),
                            "treated_fraction_full": float(
                                frame[
                                    "analysis_treatment"
                                ].mean()
                            ),
                            "treated_fraction_no_chemo": float(
                                no_chemo_frame[
                                    "analysis_treatment"
                                ].mean()
                            ),
                            **result,
                        }
                        append_row(row, checkpoint_path)
                        print(
                            f"{scenario_id} rep={repetition:03d} "
                            f"method={method_name:16s} "
                            f"truth={row['truth_days']:+.3f} "
                            f"estimate={row['estimate_days']:+.3f} "
                            f"bias={row['bias_days']:+.3f} "
                            f"events={row['events']:3d} "
                            f"maxSMD={row['max_abs_weighted_smd']:.3f}"
                        )
                    except Exception as error:
                        error_record = {
                            "scenario_id": scenario_id,
                            "sample_size": int(sample_size),
                            "sequencing_strength_name": strength_name,
                            "sequencing_strength": float(strength_value),
                            "repetition": repetition,
                            "seed": seed,
                            "method": method_name,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        errors = [
                            item
                            for item in errors
                            if not (
                                str(item["scenario_id"])
                                == scenario_id
                                and int(item["repetition"])
                                == repetition
                                and str(item["method"])
                                == method_name
                            )
                        ]
                        errors.append(error_record)
                        write_csv(
                            pd.DataFrame(errors).sort_values(
                                [
                                    "scenario_id",
                                    "repetition",
                                    "method",
                                ]
                            ),
                            errors_path,
                        )
                        row = {
                            "scenario_id": scenario_id,
                            "sample_size": int(sample_size),
                            "sequencing_strength_name": strength_name,
                            "sequencing_strength": float(strength_value),
                            "repetition": repetition,
                            "seed": seed,
                            "method": method_name,
                            "success": False,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                        append_row(row, checkpoint_path)
                        print(
                            f"{scenario_id} rep={repetition:03d} "
                            f"method={method_name} FAILED "
                            f"{type(error).__name__}: {error}"
                        )

    checkpoint = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )
    total_expected = (
        len(config["simulation"]["sample_sizes"])
        * len(
            config["simulation"][
                "sequencing_strengths"
            ]
        )
        * repetitions
        * len(estimator_specs)
    )
    total_success = int(
        checkpoint["success"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    print(
        f"\nSimulation pilot execution complete: "
        f"success={total_success}/{total_expected}"
    )
    print(
        "Stage 120 will summarize success fractions and "
        "will not treat pilot failures as silently missing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
