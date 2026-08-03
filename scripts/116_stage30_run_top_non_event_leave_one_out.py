from __future__ import annotations

import json

import pandas as pd

from _stage30_v10_non_event_influence_utils import (
    append_checkpoint,
    assemble_frame,
    load_json,
    project_root,
    run_point_estimate,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage30_v10_non_event_influence_config.json"
    )
    output = config["output"]
    manifest = load_json(
        root / output["calculation_manifest"]
    )

    print("=" * 128)
    print("STAGE 116 - RUN TOP-INFLUENCE NON-EVENT LEAVE-ONE-OUT")
    print("=" * 128)
    print("No bootstrap is run.")

    if (
        manifest["status"]
        != "STAGE30_NON_EVENT_INFLUENCE_LOCKED"
    ):
        raise RuntimeError("Stage 30 analysis is not locked.")

    frame, features = assemble_frame(root, config)
    primary = float(
        config["expected"]["primary_point_estimate_days"]
    )

    identity_path = root / output["identity_check"]
    if identity_path.exists():
        identity = load_json(identity_path)
        if not bool(identity["pass"]):
            raise RuntimeError(
                "Existing identity reproduction is marked failed."
            )
    else:
        result, _ = run_point_estimate(
            frame,
            features,
            config,
        )
        difference = float(
            result["estimate_days"] - primary
        )
        passed = abs(difference) <= float(
            config["expected"]["identity_tolerance_days"]
        )
        identity = {
            "expected_point_estimate_days": primary,
            "identity_estimate_days": result["estimate_days"],
            "difference_days": difference,
            "tolerance_days": config["expected"][
                "identity_tolerance_days"
            ],
            "pass": passed,
            "diagnostics": result,
        }
        write_json(identity, identity_path)
        if not passed:
            raise RuntimeError(
                "Stage 30 identity run did not reproduce Stage 26."
            )

    print(
        f"Identity estimate: "
        f"{identity['identity_estimate_days']:+.12f} days"
    )

    selected = pd.read_csv(
        root / manifest["selected_set_path_LOCAL_ONLY"],
        low_memory=False,
    )
    checkpoint_path = root / output["checkpoint_local"]

    if checkpoint_path.exists():
        checkpoint = pd.read_csv(
            checkpoint_path,
            low_memory=False,
        )
        completed = set(
            checkpoint.loc[
                checkpoint["success"]
                .astype(str)
                .str.lower()
                .eq("true"),
                "influence_case_id",
            ].astype(str)
        )
    else:
        completed = set()

    print(
        f"Completed targeted deletions: "
        f"{len(completed)}/{len(selected)}"
    )

    for _, selected_row in selected.iterrows():
        case_id = str(selected_row["influence_case_id"])
        if case_id in completed:
            continue

        patient_id = str(
            selected_row["patient_id_normalized"]
        )
        reduced = frame[
            frame["patient_id_normalized"].astype(str)
            != patient_id
        ].reset_index(drop=True)

        try:
            result, _ = run_point_estimate(
                reduced,
                features,
                config,
            )
            row = {
                "influence_rank": int(
                    selected_row["influence_rank"]
                ),
                "influence_case_id": case_id,
                "patient_id_normalized": patient_id,
                "arm": str(selected_row["arm"]),
                "analysis_time": float(
                    selected_row["analysis_time"]
                ),
                "original_influence": float(
                    selected_row["influence"]
                ),
                "original_absolute_influence": float(
                    selected_row["absolute_influence"]
                ),
                "original_normalized_contribution_days": float(
                    selected_row[
                        "normalized_contribution_days"
                    ]
                ),
                "success": True,
                "error_type": "",
                "error_message": "",
                **result,
                "difference_from_primary_days": float(
                    result["estimate_days"] - primary
                ),
                "absolute_change_days": float(
                    abs(result["estimate_days"] - primary)
                ),
                "relative_change_from_primary": float(
                    (result["estimate_days"] - primary)
                    / abs(primary)
                ),
                "estimate_remains_positive": bool(
                    result["estimate_days"] > 0
                ),
            }
            append_checkpoint(row, checkpoint_path)

            print(
                f"{case_id} rank={row['influence_rank']:02d} "
                f"arm={row['arm']:13s} "
                f"estimate={row['estimate_days']:+.6f} "
                f"change={row['difference_from_primary_days']:+.6f} "
                f"range={row['partition_range_days']:.6f} "
                f"Gmin={row['minimum_raw_G']:.6f}"
            )
        except Exception as error:
            row = {
                "influence_rank": int(
                    selected_row["influence_rank"]
                ),
                "influence_case_id": case_id,
                "patient_id_normalized": patient_id,
                "arm": str(selected_row["arm"]),
                "analysis_time": float(
                    selected_row["analysis_time"]
                ),
                "original_influence": float(
                    selected_row["influence"]
                ),
                "original_absolute_influence": float(
                    selected_row["absolute_influence"]
                ),
                "original_normalized_contribution_days": float(
                    selected_row[
                        "normalized_contribution_days"
                    ]
                ),
                "success": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            append_checkpoint(row, checkpoint_path)
            print(
                f"{case_id} FAILED "
                f"{type(error).__name__}: {error}"
            )

    checkpoint = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )
    successful = (
        checkpoint[
            checkpoint["success"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]
        .drop_duplicates("influence_case_id")
    )

    required = int(config["selection"]["top_k"])
    print(
        f"\nTargeted deletion execution complete: "
        f"success={len(successful)}, required={required}"
    )

    if len(successful) != required:
        raise RuntimeError(
            "Stage 30 is incomplete. Rerun the same command after "
            "reviewing failed rows; successful deletions will be skipped."
        )

    print(
        "\nPASS: all top-influence non-event leave-one-out "
        "estimates completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
