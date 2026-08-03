from __future__ import annotations

import json

import numpy as np

from _stage27_v10_bootstrap_utils import (
    assemble_frame,
    load_json,
    project_root,
    run_resample,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage27_v10_bootstrap_config.json"
    )
    output = config["output"]
    identity_path = root / output["identity_check"]

    print("=" * 128)
    print("STAGE 105 - IDENTITY RESAMPLE REPRODUCTION")
    print("=" * 128)

    if identity_path.exists():
        identity = load_json(identity_path)
        if not bool(identity["pass"]):
            raise RuntimeError(
                "Existing identity reproduction is marked failed."
            )
        print(json.dumps(identity, indent=2))
        print("\nPASS: existing identity reproduction verified.")
        return 0

    frame, features = assemble_frame(root, config)
    indices = np.arange(len(frame), dtype=int)
    result = run_resample(
        frame,
        features,
        indices,
        0,
        config,
    )
    expected = float(
        config["expected"]["point_estimate_days"]
    )
    difference = float(result["estimate_days"] - expected)
    passed = abs(difference) <= float(
        config["bootstrap"][
            "identity_reproduction_tolerance_days"
        ]
    )
    identity = {
        "expected_stage26_point_estimate_days": expected,
        "identity_resample_estimate_days": result[
            "estimate_days"
        ],
        "difference_days": difference,
        "tolerance_days": config["bootstrap"][
            "identity_reproduction_tolerance_days"
        ],
        "pass": passed,
        "identity_diagnostics": result,
    }
    write_json(identity, identity_path)
    print(json.dumps(identity, indent=2))

    if not passed:
        raise RuntimeError(
            "Stage 27 identity resample did not reproduce Stage 26."
        )

    print(
        "\nPASS: Stage 27 implementation exactly reproduces "
        "the locked Stage 26 point estimate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
