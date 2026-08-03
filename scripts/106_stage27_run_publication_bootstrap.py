from __future__ import annotations

import numpy as np
import pandas as pd

from _stage27_v10_bootstrap_utils import (
    append_checkpoint,
    assemble_frame,
    error_row,
    load_json,
    locked_bootstrap_settings,
    project_root,
    run_resample,
    write_csv,
)


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
    repetitions = int(locked_bootstrap["repetitions"])
    base_seed = int(locked_bootstrap["base_seed"])
    checkpoint_path = root / output["checkpoint"]
    errors_path = root / output["errors"]

    print("=" * 128)
    print("STAGE 106 - RUN 300-REPETITION FULLY REFITTED PATIENT BOOTSTRAP")
    print("=" * 128)

    frame, features = assemble_frame(root, config)
    n = len(frame)

    if checkpoint_path.exists():
        checkpoint = pd.read_csv(
            checkpoint_path,
            low_memory=False,
        )
        completed = set(
            pd.to_numeric(
                checkpoint.loc[
                    checkpoint["success"].astype(str).str.lower()
                    == "true",
                    "repetition",
                ],
                errors="coerce",
            ).dropna().astype(int)
        )
    else:
        completed = set()

    print(
        f"Completed successful repetitions: "
        f"{len(completed)}/{repetitions}"
    )

    errors = []
    if errors_path.exists():
        errors = pd.read_csv(
            errors_path,
            low_memory=False,
        ).to_dict("records")

    for repetition in range(1, repetitions + 1):
        if repetition in completed:
            continue

        seed = base_seed + repetition
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, n)

        try:
            result = run_resample(
                frame,
                features,
                indices,
                repetition,
                config,
                bootstrap_seed=seed,
            )
            append_checkpoint(
                result,
                checkpoint_path,
            )
            if errors:
                errors = [
                    existing
                    for existing in errors
                    if int(existing["repetition"]) != repetition
                ]
                write_csv(
                    pd.DataFrame(errors),
                    errors_path,
                )
            print(
                f"bootstrap={repetition:03d}/{repetitions} "
                f"effect={result['estimate_days']:+.6f} "
                f"unique={result['unique_source_patients']:3d} "
                f"events={result['events']:2d} "
                f"treated_events={result['treated_events']:2d} "
                f"p01={result['propensity_p01']:.6f} "
                f"p99={result['propensity_p99']:.6f} "
                f"maxSMD={result['max_abs_ato_weighted_smd']:.3e} "
                f"Gmin={result['minimum_G_min_raw']:.6f} "
                f"retry={result['maximum_nuisance_retry']}"
            )
        except Exception as error:
            row = error_row(
                repetition,
                seed,
                error,
            )
            errors = [
                existing
                for existing in errors
                if int(existing["repetition"]) != repetition
            ]
            errors.append(row)
            write_csv(
                pd.DataFrame(errors).sort_values("repetition"),
                errors_path,
            )
            append_checkpoint(
                row,
                checkpoint_path,
            )
            print(
                f"bootstrap={repetition:03d}/{repetitions} "
                f"FAILED {type(error).__name__}: {error}"
            )

    checkpoint = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )
    success_count = int(
        checkpoint["success"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    failure_count = int(len(checkpoint) - success_count)

    print(
        f"\nBootstrap execution complete: "
        f"success={success_count}, failure={failure_count}"
    )

    if success_count < int(
        config["bootstrap"]["required_successful_repetitions"]
    ):
        raise RuntimeError(
            "Publication bootstrap is incomplete. "
            "Rerun the same command after reviewing the error table; "
            "completed successful repetitions will be skipped."
        )

    print("\nPASS: all locked bootstrap repetitions completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
