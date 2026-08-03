#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage25_v10_utils import (
    balance_table,
    compact_features,
    dataframe_console,
    load_config,
    propensity_diagnostics,
    project_root,
    read_csv,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_config(root)
    output = config["output"]
    gates = config["pre_effect_gates"]

    print("=" * 128)
    print("STAGE 95 - CANDIDATE V10 PRE-EFFECT OVERLAP AND BALANCE GATES")
    print("=" * 128)
    print("No treatment-effect model is fitted.")

    cohort = read_csv(root / output["cohort"])
    compact = read_csv(root / output["compact"])
    features = compact_features(compact)

    frame = (
        cohort.drop(
            columns=[
                column
                for column in features
                if column in cohort.columns
            ],
            errors="ignore",
        )
        .merge(
            compact[
                ["patient_id_normalized"] + features
            ],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()

    partitions, patient, propensity = (
        propensity_diagnostics(
            frame,
            features,
            config,
        )
    )
    balance = balance_table(
        frame,
        features,
        treatment,
        patient["propensity_repeated_mean"].to_numpy(
            dtype=float
        ),
    )

    treated_events = int(
        ((treatment == 1) & (event == 1)).sum()
    )
    control_events = int(
        ((treatment == 0) & (event == 1)).sum()
    )
    max_abs_weighted_smd = float(
        balance["abs_ato_weighted_smd"].max()
    )

    checks = pd.DataFrame([
        {
            "check": "minimum n",
            "observed": len(frame),
            "threshold": gates["minimum_n"],
            "pass": len(frame) >= int(gates["minimum_n"]),
        },
        {
            "check": "minimum treated",
            "observed": int(treatment.sum()),
            "threshold": gates["minimum_treated"],
            "pass": int(treatment.sum())
            >= int(gates["minimum_treated"]),
        },
        {
            "check": "minimum control",
            "observed": int((1 - treatment).sum()),
            "threshold": gates["minimum_control"],
            "pass": int((1 - treatment).sum())
            >= int(gates["minimum_control"]),
        },
        {
            "check": "minimum total events",
            "observed": int(event.sum()),
            "threshold": gates["minimum_total_events"],
            "pass": int(event.sum())
            >= int(gates["minimum_total_events"]),
        },
        {
            "check": "minimum treated events",
            "observed": treated_events,
            "threshold": gates["minimum_events_each_arm"],
            "pass": treated_events
            >= int(gates["minimum_events_each_arm"]),
        },
        {
            "check": "minimum control events",
            "observed": control_events,
            "threshold": gates["minimum_events_each_arm"],
            "pass": control_events
            >= int(gates["minimum_events_each_arm"]),
        },
        {
            "check": "feature count",
            "observed": len(features),
            "threshold": gates["required_feature_count"],
            "pass": len(features)
            == int(gates["required_feature_count"]),
        },
        {
            "check": "maximum absolute ATO-weighted SMD",
            "observed": max_abs_weighted_smd,
            "threshold": gates[
                "maximum_abs_ato_weighted_smd"
            ],
            "pass": max_abs_weighted_smd
            <= float(gates[
                "maximum_abs_ato_weighted_smd"
            ]),
        },
        {
            "check": "treated ATO ESS fraction",
            "observed": propensity[
                "ato_ess_fraction_treated"
            ],
            "threshold": gates[
                "minimum_ato_ess_fraction_each_arm"
            ],
            "pass": propensity[
                "ato_ess_fraction_treated"
            ]
            >= float(gates[
                "minimum_ato_ess_fraction_each_arm"
            ]),
        },
        {
            "check": "control ATO ESS fraction",
            "observed": propensity[
                "ato_ess_fraction_control"
            ],
            "threshold": gates[
                "minimum_ato_ess_fraction_each_arm"
            ],
            "pass": propensity[
                "ato_ess_fraction_control"
            ]
            >= float(gates[
                "minimum_ato_ess_fraction_each_arm"
            ]),
        },
        {
            "check": "fraction propensity below 0.05",
            "observed": propensity[
                "fraction_propensity_below_0_05"
            ],
            "threshold": gates[
                "maximum_fraction_propensity_below_0_05"
            ],
            "pass": propensity[
                "fraction_propensity_below_0_05"
            ]
            <= float(gates[
                "maximum_fraction_propensity_below_0_05"
            ]),
        },
        {
            "check": "fraction propensity above 0.95",
            "observed": propensity[
                "fraction_propensity_above_0_95"
            ],
            "threshold": gates[
                "maximum_fraction_propensity_above_0_95"
            ],
            "pass": propensity[
                "fraction_propensity_above_0_95"
            ]
            <= float(gates[
                "maximum_fraction_propensity_above_0_95"
            ]),
        },
    ])

    table_dir = root / output["table_dir"]
    write_csv(
        partitions,
        table_dir / "s25_95_propensity_partition_diagnostics.csv",
    )
    write_csv(
        patient,
        table_dir
        / "s25_95_propensity_patient_diagnostics_LOCAL_ONLY.csv",
    )
    write_csv(
        balance,
        table_dir / "s25_95_ato_balance.csv",
    )
    write_csv(
        checks,
        table_dir / "s25_95_pre_effect_gates.csv",
    )
    write_json(
        {
            "n": len(frame),
            "treated": int(treatment.sum()),
            "control": int((1 - treatment).sum()),
            "events": int(event.sum()),
            "treated_events": treated_events,
            "control_events": control_events,
            "features": len(features),
            "max_abs_ato_weighted_smd": (
                max_abs_weighted_smd
            ),
            **propensity,
            "all_pre_effect_gates_pass": bool(
                checks["pass"].all()
            ),
        },
        table_dir / "s25_95_pre_effect_summary.json",
    )

    print("Propensity partition diagnostics")
    print(
        dataframe_console(
            partitions,
            max_rows=30,
        )
    )
    print("\nATO balance")
    print(
        dataframe_console(
            balance[
                [
                    "feature",
                    "missing_fraction",
                    "unweighted_smd",
                    "ato_weighted_smd",
                    "abs_ato_weighted_smd",
                ]
            ],
            max_rows=100,
        )
    )
    print("\nPre-effect gates")
    print(dataframe_console(checks))

    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Candidate V10 failed one or more pre-effect gates. "
            "No protocol lock was created."
        )

    print(
        "\nPASS: Candidate V10 pre-effect overlap and balance "
        "gates passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
