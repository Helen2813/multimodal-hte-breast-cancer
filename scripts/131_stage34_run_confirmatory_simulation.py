from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from _stage33_simulation_v2_utils import (
    estimate_method_v2,
    load_json,
    simulate_dataset_v2,
    truth_bundle,
    write_csv,
)


def append_row(row: dict, path: Path) -> None:
    if path.exists():
        frame = pd.read_csv(
            path,
            low_memory=False,
            keep_default_na=False,
        )
        mask = ~(
            (frame["scenario_id"].astype(str) == str(row["scenario_id"]))
            & (
                pd.to_numeric(
                    frame["repetition"], errors="coerce"
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
    root = Path.cwd().resolve()
    config = load_json(root / "stage34_confirmatory_simulation_config.json")
    output = config["outputs"]
    manifest = load_json(root / output["manifest"])

    print("=" * 128)
    print("STAGE 131 - RUN INDEPENDENT CONFIRMATORY SIMULATION")
    print("=" * 128)

    if manifest["status"] != "STAGE34_CONFIRMATORY_SIMULATION_LOCKED":
        raise RuntimeError("Stage 34 confirmatory simulation is not locked.")

    checkpoint_path = root / output["checkpoint"]
    errors_path = root / output["errors"]

    if checkpoint_path.exists():
        checkpoint = pd.read_csv(
            checkpoint_path,
            low_memory=False,
            keep_default_na=False,
        )
        completed_frame = checkpoint[
            checkpoint["success"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]
        completed = set(
            zip(
                completed_frame["scenario_id"].astype(str),
                pd.to_numeric(
                    completed_frame["repetition"],
                    errors="coerce",
                ).astype(int),
                completed_frame["method"].astype(str),
            )
        )
    else:
        completed = set()

    errors = []
    if errors_path.exists():
        errors = pd.read_csv(
            errors_path,
            low_memory=False,
            keep_default_na=False,
        ).to_dict("records")

    simulation = config["simulation"]
    methods = config["methods"]
    calibrated = manifest["calibrated_parameters"]
    fixed = manifest["fixed_covariate_parameters"]
    censor = manifest["censoring_parameters"]
    sequence_values = manifest["scenario_parameters"]["sequencing_strengths"]
    effect_values = manifest["scenario_parameters"]["treatment_effects"]

    scenario_number = 0
    for n in simulation["sample_sizes"]:
        for sequence_name in simulation["sequencing_levels"]:
            for effect_name in simulation["effect_regimes"]:
                scenario_number += 1
                sequence_value = float(sequence_values[sequence_name])
                effect_value = float(effect_values[effect_name])
                scenario_id = (
                    f"N{int(n)}_{sequence_name.upper()}_{effect_name.upper()}"
                )
                print(
                    "\n" + "-" * 128
                    + f"\n{scenario_id}: sequence={sequence_value:.6f}, "
                    f"effect={effect_value:.6f}\n"
                    + "-" * 128
                )

                for repetition in range(
                    1,
                    int(simulation["repetitions_per_scenario"]) + 1,
                ):
                    seed = (
                        int(simulation["base_seed"])
                        + scenario_number * 1_000_000
                        + repetition
                    )
                    full = simulate_dataset_v2(
                        int(n),
                        sequence_value,
                        effect_value,
                        seed,
                        calibrated,
                        fixed,
                        censor,
                        config,
                    )
                    strict = full[
                        full["strict_sequence_eligible"] == 1
                    ].reset_index(drop=True)
                    truths = truth_bundle(full)

                    frames = {
                        "naive_full": full,
                        "adjusted_full": full,
                        "sequencing_aware": strict,
                    }
                    primary_truths = {
                        "naive_full": truths["intended_full_ato_truth"],
                        "adjusted_full": truths["intended_full_ato_truth"],
                        "sequencing_aware": truths[
                            "strict_no_chemo_ato_truth"
                        ],
                    }
                    secondary_truths = {
                        "naive_full": truths["naive_implied_overlap_truth"],
                        "adjusted_full": None,
                        "sequencing_aware": None,
                    }

                    treatment = full[
                        "analysis_treatment"
                    ].to_numpy(dtype=int)
                    chemo = full[
                        "chemo_by_day180"
                    ].to_numpy(dtype=int)
                    common = {
                        "full_cohort_n": len(full),
                        "strict_population_n": len(strict),
                        "full_treated_fraction": float(treatment.mean()),
                        "strict_treated_fraction": float(
                            strict["analysis_treatment"].mean()
                        ),
                        "full_chemo_prevalence": float(chemo.mean()),
                        "full_chemo_fraction_treated": float(
                            chemo[treatment == 1].mean()
                        ),
                        "full_chemo_fraction_control": float(
                            chemo[treatment == 0].mean()
                        ),
                        "strict_treated_n": int(
                            strict["analysis_treatment"].sum()
                        ),
                        "strict_control_n": int(
                            (1 - strict["analysis_treatment"]).sum()
                        ),
                        "strict_events": int(
                            strict["analysis_event"].sum()
                        ),
                        "strict_treated_events": int(
                            (
                                (strict["analysis_treatment"] == 1)
                                & (strict["analysis_event"] == 1)
                            ).sum()
                        ),
                        "strict_control_events": int(
                            (
                                (strict["analysis_treatment"] == 0)
                                & (strict["analysis_event"] == 1)
                            ).sum()
                        ),
                        **truths,
                    }

                    for method_name, method_spec in methods.items():
                        key = (scenario_id, repetition, method_name)
                        if key in completed:
                            continue
                        try:
                            offset = {
                                "naive_full": 101,
                                "adjusted_full": 202,
                                "sequencing_aware": 303,
                            }[method_name]
                            result = estimate_method_v2(
                                frames[method_name],
                                list(method_spec["covariates"]),
                                float(primary_truths[method_name]),
                                seed + offset,
                                config,
                                secondary_truth=secondary_truths[method_name],
                            )
                            row = {
                                "scenario_id": scenario_id,
                                "sample_size": int(n),
                                "sequencing_level": sequence_name,
                                "sequencing_strength": sequence_value,
                                "effect_regime": effect_name,
                                "treatment_log_hazard_effect": effect_value,
                                "repetition": repetition,
                                "seed": seed,
                                "method": method_name,
                                "success": True,
                                "error_type": "",
                                "error_message": "",
                                **common,
                                **result,
                            }
                            append_row(row, checkpoint_path)
                            print(
                                f"{scenario_id} rep={repetition:04d} "
                                f"method={method_name:16s} "
                                f"estimate={row['estimate_days']:+.3f} "
                                f"bias={row['primary_bias_days']:+.3f}"
                            )
                        except Exception as error:
                            error_record = {
                                "scenario_id": scenario_id,
                                "sample_size": int(n),
                                "sequencing_level": sequence_name,
                                "effect_regime": effect_name,
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
                                    str(item["scenario_id"]) == scenario_id
                                    and int(item["repetition"]) == repetition
                                    and str(item["method"]) == method_name
                                )
                            ]
                            errors.append(error_record)
                            write_csv(
                                pd.DataFrame(errors).sort_values(
                                    ["scenario_id", "repetition", "method"]
                                ),
                                errors_path,
                            )
                            row = {
                                "scenario_id": scenario_id,
                                "sample_size": int(n),
                                "sequencing_level": sequence_name,
                                "sequencing_strength": sequence_value,
                                "effect_regime": effect_name,
                                "repetition": repetition,
                                "seed": seed,
                                "method": method_name,
                                "success": False,
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                            }
                            append_row(row, checkpoint_path)
                            print(
                                f"{scenario_id} rep={repetition:04d} "
                                f"method={method_name} FAILED: {error}"
                            )

    checkpoint = pd.read_csv(
        checkpoint_path,
        low_memory=False,
        keep_default_na=False,
    )
    success_count = int(
        checkpoint["success"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    print(
        f"\nConfirmatory execution complete: "
        f"success={success_count}/"
        f"{simulation['expected_method_runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
