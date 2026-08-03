from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage25b_v10_balance_utils import (
    balance_table,
    compact_features,
    dataframe_console,
    fit_unpenalized_propensity,
    load_json,
    project_root,
    propensity_summary,
    verify_stage25_inputs,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage25b_v10_balance_repair_config.json"
    )
    source = config["source"]
    output = config["output"]
    gates = config["gates"]

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 128)
    print("STAGE 97 - OUTCOME-BLIND CANDIDATE V10 PROPENSITY BALANCE REPAIR")
    print("=" * 128)
    print("No RMST effect or survival outcome model is fitted.")

    verification = verify_stage25_inputs(root, config)

    cohort = pd.read_csv(
        root / source["v10_cohort"],
        low_memory=False,
    )
    compact = pd.read_csv(
        root / source["v10_compact"],
        low_memory=False,
    )
    features = compact_features(compact)

    frame = cohort[
        [
            "patient_id_normalized",
            "analysis_treatment",
            "analysis_event",
        ]
    ].merge(
        compact[
            ["patient_id_normalized"] + features
        ],
        on="patient_id_normalized",
        how="inner",
        validate="one_to_one",
    )
    expected = config["expected"]
    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()

    count_checks = pd.DataFrame([
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
    ])
    if not bool(count_checks["pass"].all()):
        raise RuntimeError(
            "Frozen V10 count checks failed.\n"
            + dataframe_console(count_checks)
        )

    propensity, coefficients, fit_summary = (
        fit_unpenalized_propensity(
            frame,
            treatment,
            features,
            config,
        )
    )
    repaired_balance = balance_table(
        frame,
        features,
        treatment,
        propensity,
    )
    repaired_summary = propensity_summary(
        treatment,
        propensity,
    )
    repaired_summary.update({
        "method": config["design_repair"]["name"],
        "converged": fit_summary["converged"],
        "maximum_absolute_coefficient": fit_summary[
            "maximum_absolute_coefficient"
        ],
        "max_abs_ato_weighted_smd": float(
            repaired_balance[
                "abs_ato_weighted_smd"
            ].max()
        ),
    })

    original_balance = pd.read_csv(
        root / source["stage25_original_balance"],
        low_memory=False,
    )
    original_summary = load_json(
        root / source["stage25_original_summary"]
    )
    comparison = pd.DataFrame([
        {
            "method": "v9_crossfitted_regularized_propensity",
            "max_abs_ato_weighted_smd": float(
                original_balance[
                    "abs_ato_weighted_smd"
                ].max()
            ),
            "propensity_min": original_summary[
                "propensity_min"
            ],
            "propensity_max": original_summary[
                "propensity_max"
            ],
            "fraction_below_0_05": original_summary[
                "fraction_propensity_below_0_05"
            ],
            "fraction_above_0_95": original_summary[
                "fraction_propensity_above_0_95"
            ],
            "ato_ess_fraction_treated": original_summary[
                "ato_ess_fraction_treated"
            ],
            "ato_ess_fraction_control": original_summary[
                "ato_ess_fraction_control"
            ],
        },
        {
            "method": config["design_repair"]["name"],
            "max_abs_ato_weighted_smd": repaired_summary[
                "max_abs_ato_weighted_smd"
            ],
            "propensity_min": repaired_summary[
                "propensity_min"
            ],
            "propensity_max": repaired_summary[
                "propensity_max"
            ],
            "fraction_below_0_05": repaired_summary[
                "fraction_propensity_below_0_05"
            ],
            "fraction_above_0_95": repaired_summary[
                "fraction_propensity_above_0_95"
            ],
            "ato_ess_fraction_treated": repaired_summary[
                "ato_ess_fraction_treated"
            ],
            "ato_ess_fraction_control": repaired_summary[
                "ato_ess_fraction_control"
            ],
        },
    ])

    checks = pd.DataFrame([
        {
            "check": "GLM converged",
            "observed": repaired_summary["converged"],
            "threshold": True,
            "pass": bool(repaired_summary["converged"]),
        },
        {
            "check": "maximum absolute coefficient",
            "observed": repaired_summary[
                "maximum_absolute_coefficient"
            ],
            "threshold": gates[
                "maximum_absolute_coefficient"
            ],
            "pass": repaired_summary[
                "maximum_absolute_coefficient"
            ]
            <= float(gates[
                "maximum_absolute_coefficient"
            ]),
        },
        {
            "check": "maximum absolute ATO-weighted SMD",
            "observed": repaired_summary[
                "max_abs_ato_weighted_smd"
            ],
            "threshold": gates[
                "maximum_abs_ato_weighted_smd"
            ],
            "pass": repaired_summary[
                "max_abs_ato_weighted_smd"
            ]
            <= float(gates[
                "maximum_abs_ato_weighted_smd"
            ]),
        },
        {
            "check": "minimum propensity",
            "observed": repaired_summary[
                "propensity_min"
            ],
            "threshold": gates["minimum_propensity"],
            "pass": repaired_summary[
                "propensity_min"
            ]
            >= float(gates["minimum_propensity"]),
        },
        {
            "check": "maximum propensity",
            "observed": repaired_summary[
                "propensity_max"
            ],
            "threshold": gates["maximum_propensity"],
            "pass": repaired_summary[
                "propensity_max"
            ]
            <= float(gates["maximum_propensity"]),
        },
        {
            "check": "fraction propensity below 0.05",
            "observed": repaired_summary[
                "fraction_propensity_below_0_05"
            ],
            "threshold": gates[
                "maximum_fraction_propensity_below_0_05"
            ],
            "pass": repaired_summary[
                "fraction_propensity_below_0_05"
            ]
            <= float(gates[
                "maximum_fraction_propensity_below_0_05"
            ]),
        },
        {
            "check": "fraction propensity above 0.95",
            "observed": repaired_summary[
                "fraction_propensity_above_0_95"
            ],
            "threshold": gates[
                "maximum_fraction_propensity_above_0_95"
            ],
            "pass": repaired_summary[
                "fraction_propensity_above_0_95"
            ]
            <= float(gates[
                "maximum_fraction_propensity_above_0_95"
            ]),
        },
        {
            "check": "treated ATO ESS fraction",
            "observed": repaired_summary[
                "ato_ess_fraction_treated"
            ],
            "threshold": gates[
                "minimum_ato_ess_fraction_each_arm"
            ],
            "pass": repaired_summary[
                "ato_ess_fraction_treated"
            ]
            >= float(gates[
                "minimum_ato_ess_fraction_each_arm"
            ]),
        },
        {
            "check": "control ATO ESS fraction",
            "observed": repaired_summary[
                "ato_ess_fraction_control"
            ],
            "threshold": gates[
                "minimum_ato_ess_fraction_each_arm"
            ],
            "pass": repaired_summary[
                "ato_ess_fraction_control"
            ]
            >= float(gates[
                "minimum_ato_ess_fraction_each_arm"
            ]),
        },
    ])

    patient = pd.DataFrame({
        "patient_id_normalized": frame[
            "patient_id_normalized"
        ].astype(str),
        "analysis_treatment": treatment,
        "propensity_full_sample_mle": propensity,
        "ato_weight": np.where(
            treatment == 1,
            1.0 - propensity,
            propensity,
        ),
    })

    write_csv(
        count_checks,
        table_dir / "s25b_97_frozen_v10_count_checks.csv",
    )
    write_csv(
        coefficients,
        table_dir / "s25b_97_propensity_coefficients.csv",
    )
    write_csv(
        repaired_balance,
        table_dir / "s25b_97_repaired_ato_balance.csv",
    )
    write_csv(
        comparison,
        table_dir / "s25b_97_propensity_method_comparison.csv",
    )
    write_csv(
        checks,
        table_dir / "s25b_97_balance_repair_gates.csv",
    )
    write_csv(
        patient,
        table_dir
        / "s25b_97_repaired_propensity_LOCAL_ONLY.csv",
    )
    write_json(
        {
            "input_verification": verification,
            "fit_summary": fit_summary,
            "propensity_summary": repaired_summary,
            "all_balance_repair_gates_pass": bool(
                checks["pass"].all()
            ),
            "effect_estimated": False,
        },
        table_dir / "s25b_97_balance_repair_summary.json",
    )

    print("Frozen V10 checks")
    print(dataframe_console(count_checks))
    print("\nPropensity-method comparison")
    print(dataframe_console(comparison))
    print("\nRepaired ATO balance")
    print(
        dataframe_console(
            repaired_balance[
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
    print("\nBalance-repair gates")
    print(dataframe_console(checks))

    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "The outcome-blind propensity repair failed one or more gates. "
            "No Candidate V10 protocol lock is authorized."
        )

    print(
        "\nPASS: outcome-blind propensity balance repair passed. "
        "No treatment effect was estimated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
