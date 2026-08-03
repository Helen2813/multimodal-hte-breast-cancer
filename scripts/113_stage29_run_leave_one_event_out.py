from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage29_v10_event_influence_utils import (
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
        root / "stage29_v10_event_influence_config.json"
    )
    output = config["output"]
    manifest = load_json(
        root / output["calculation_manifest"]
    )

    print("=" * 128)
    print("STAGE 113 - RUN LEAVE-ONE-EVENT-PATIENT-OUT ANALYSIS")
    print("=" * 128)
    print("No bootstrap is run.")

    if manifest["status"] != "STAGE29_EVENT_INFLUENCE_LOCKED":
        raise RuntimeError("Stage 29 analysis is not locked.")

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
        identity_result, _ = run_point_estimate(
            frame,
            features,
            config,
        )
        difference = float(
            identity_result["estimate_days"] - primary
        )
        passed = abs(difference) <= float(
            config["expected"]["identity_tolerance_days"]
        )
        identity = {
            "expected_point_estimate_days": primary,
            "identity_estimate_days": identity_result[
                "estimate_days"
            ],
            "difference_days": difference,
            "tolerance_days": config["expected"][
                "identity_tolerance_days"
            ],
            "pass": passed,
            "diagnostics": identity_result,
        }
        write_json(identity, identity_path)
        if not passed:
            raise RuntimeError(
                "Stage 29 identity run did not reproduce Stage 26."
            )

    print(
        f"Identity estimate: "
        f"{identity['identity_estimate_days']:+.12f} days"
    )

    omitted_set = pd.read_csv(
        root / manifest["omitted_set_path_LOCAL_ONLY"],
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
                checkpoint["success"].astype(str).str.lower().eq("true"),
                "event_case_id",
            ].astype(str)
        )
    else:
        completed = set()

    print(
        f"Completed event deletions: "
        f"{len(completed)}/{len(omitted_set)}"
    )

    for _, omitted in omitted_set.iterrows():
        event_case_id = str(omitted["event_case_id"])
        if event_case_id in completed:
            continue

        patient_id = str(omitted["patient_id_normalized"])
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
                "event_case_id": event_case_id,
                "patient_id_normalized": patient_id,
                "omitted_arm": str(omitted["omitted_arm"]),
                "omitted_analysis_time": float(
                    omitted["analysis_time"]
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
                f"{event_case_id} "
                f"arm={row['omitted_arm']:13s} "
                f"estimate={row['estimate_days']:+.6f} "
                f"change={row['difference_from_primary_days']:+.6f} "
                f"range={row['partition_range_days']:.6f} "
                f"Gmin={row['minimum_raw_G']:.6f}"
            )
        except Exception as error:
            row = {
                "event_case_id": event_case_id,
                "patient_id_normalized": patient_id,
                "omitted_arm": str(omitted["omitted_arm"]),
                "omitted_analysis_time": float(
                    omitted["analysis_time"]
                ),
                "success": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            append_checkpoint(row, checkpoint_path)
            print(
                f"{event_case_id} FAILED "
                f"{type(error).__name__}: {error}"
            )

    checkpoint = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )
    successful = checkpoint[
        checkpoint["success"].astype(str).str.lower().eq("true")
    ]
    success_count = len(
        successful.drop_duplicates("event_case_id")
    )

    print(
        f"\nEvent-deletion execution complete: "
        f"success={success_count}, "
        f"required={config['expected']['required_successful_deletions']}"
    )

    if success_count != int(
        config["expected"]["required_successful_deletions"]
    ):
        raise RuntimeError(
            "Stage 29 is incomplete. Rerun the same command after "
            "reviewing failed rows; successful deletions will be skipped."
        )

    print(
        "\nPASS: all 36 leave-one-event-patient-out estimates completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
