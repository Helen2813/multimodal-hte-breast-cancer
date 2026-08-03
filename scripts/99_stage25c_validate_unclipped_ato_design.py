from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _stage25c_v10_utils import (
    balance_table,
    bootstrap_propensity_feasibility,
    compact_features,
    dataframe_console,
    fit_propensity,
    load_json,
    project_root,
    propensity_metrics,
    sha256_file,
    write_csv,
    write_json,
)


def main() -> int:
    root = project_root()
    config = load_json(
        root / "stage25c_v10_unclipped_ato_config.json"
    )
    source = config["source"]
    output = config["output"]
    expected = config["expected"]
    gates = config["gates"]

    table_dir = root / output["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 128)
    print("STAGE 99 - VALIDATE UNCLIPPED STABILIZED ATO DESIGN")
    print("=" * 128)
    print("No RMST effect, censoring model, or outcome model is fitted.")

    cohort_path = root / source["v10_cohort"]
    compact_path = root / source["v10_compact"]
    stage25_summary = load_json(
        root / source["stage25_cohort_summary"]
    )
    repair_summary = load_json(
        root / source["stage25b_repair_summary"]
    )
    repair_gates = pd.read_csv(
        root / source["stage25b_repair_gates"],
        low_memory=False,
    )

    if sha256_file(cohort_path) != stage25_summary["candidate_v10_cohort_sha256"]:
        raise RuntimeError("Frozen V10 cohort hash mismatch.")
    if sha256_file(compact_path) != stage25_summary["candidate_v10_compact_sha256"]:
        raise RuntimeError("Frozen V10 compact-table hash mismatch.")

    failed = repair_gates[
        repair_gates["pass"].astype(str).str.lower() != "true"
    ]
    expected_failed = {
        "minimum propensity",
        "maximum propensity",
    }
    if set(failed["check"].astype(str)) != expected_failed:
        raise RuntimeError(
            "Stage 25B did not fail only the expected pointwise propensity gates.\n"
            + dataframe_console(failed)
        )

    cohort = pd.read_csv(cohort_path, low_memory=False)
    compact = pd.read_csv(compact_path, low_memory=False)
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
    treatment = pd.to_numeric(
        frame["analysis_treatment"],
        errors="raise",
    ).astype(int).to_numpy()
    event = pd.to_numeric(
        frame["analysis_event"],
        errors="raise",
    ).astype(int).to_numpy()

    count_checks = pd.DataFrame([
        {"check": "n", "observed": len(frame), "expected": expected["n"], "pass": len(frame) == expected["n"]},
        {"check": "treated", "observed": int(treatment.sum()), "expected": expected["treated"], "pass": int(treatment.sum()) == expected["treated"]},
        {"check": "control", "observed": int((1-treatment).sum()), "expected": expected["control"], "pass": int((1-treatment).sum()) == expected["control"]},
        {"check": "events", "observed": int(event.sum()), "expected": expected["events"], "pass": int(event.sum()) == expected["events"]},
        {"check": "treated events", "observed": int(((treatment==1)&(event==1)).sum()), "expected": expected["treated_events"], "pass": int(((treatment==1)&(event==1)).sum()) == expected["treated_events"]},
        {"check": "control events", "observed": int(((treatment==0)&(event==1)).sum()), "expected": expected["control_events"], "pass": int(((treatment==0)&(event==1)).sum()) == expected["control_events"]},
        {"check": "features", "observed": len(features), "expected": expected["features"], "pass": len(features) == expected["features"]},
    ])
    if not bool(count_checks["pass"].all()):
        raise RuntimeError(
            "Frozen V10 count checks failed.\n"
            + dataframe_console(count_checks)
        )

    propensity, coefficients, fit_summary = fit_propensity(
        frame,
        treatment,
        features,
        config,
    )
    balance = balance_table(
        frame,
        features,
        treatment,
        propensity,
    )
    metrics = propensity_metrics(
        treatment,
        propensity,
    )
    max_abs_smd = float(
        balance["abs_ato_weighted_smd"].max()
    )

    bootstrap_rows, bootstrap_summary = (
        bootstrap_propensity_feasibility(
            frame,
            treatment,
            features,
            config,
        )
    )

    checks = pd.DataFrame([
        {
            "check": "full-sample GLM converged",
            "observed": fit_summary["converged"],
            "threshold": True,
            "pass": bool(fit_summary["converged"]),
        },
        {
            "check": "maximum absolute coefficient",
            "observed": fit_summary["maximum_absolute_coefficient"],
            "threshold": config["propensity"]["maximum_absolute_coefficient"],
            "pass": fit_summary["maximum_absolute_coefficient"]
            <= float(config["propensity"]["maximum_absolute_coefficient"]),
        },
        {
            "check": "maximum absolute ATO-weighted SMD",
            "observed": max_abs_smd,
            "threshold": gates["maximum_abs_ato_weighted_smd"],
            "pass": max_abs_smd
            <= float(gates["maximum_abs_ato_weighted_smd"]),
        },
        {
            "check": "treated ATO ESS fraction",
            "observed": metrics["ato_ess_fraction_treated"],
            "threshold": gates["minimum_ato_ess_fraction_each_arm"],
            "pass": metrics["ato_ess_fraction_treated"]
            >= float(gates["minimum_ato_ess_fraction_each_arm"]),
        },
        {
            "check": "control ATO ESS fraction",
            "observed": metrics["ato_ess_fraction_control"],
            "threshold": gates["minimum_ato_ess_fraction_each_arm"],
            "pass": metrics["ato_ess_fraction_control"]
            >= float(gates["minimum_ato_ess_fraction_each_arm"]),
        },
        {
            "check": "fraction propensity below 0.01",
            "observed": metrics["fraction_propensity_below_0_01"],
            "threshold": gates["maximum_fraction_propensity_below_0_01"],
            "pass": metrics["fraction_propensity_below_0_01"]
            <= float(gates["maximum_fraction_propensity_below_0_01"]),
        },
        {
            "check": "fraction propensity above 0.99",
            "observed": metrics["fraction_propensity_above_0_99"],
            "threshold": gates["maximum_fraction_propensity_above_0_99"],
            "pass": metrics["fraction_propensity_above_0_99"]
            <= float(gates["maximum_fraction_propensity_above_0_99"]),
        },
        {
            "check": "normalized overlap mass",
            "observed": metrics["normalized_overlap_mass"],
            "threshold": gates["minimum_normalized_overlap_mass"],
            "pass": metrics["normalized_overlap_mass"]
            >= float(gates["minimum_normalized_overlap_mass"]),
        },
        {
            "check": "propensity bootstrap success fraction",
            "observed": bootstrap_summary["success_fraction"],
            "threshold": gates["minimum_propensity_bootstrap_success_fraction"],
            "pass": bootstrap_summary["success_fraction"]
            >= float(gates["minimum_propensity_bootstrap_success_fraction"]),
        },
        {
            "check": "propensity bootstrap tail-gate failure fraction",
            "observed": bootstrap_summary["tail_gate_failure_fraction"],
            "threshold": gates["maximum_propensity_bootstrap_fraction_tail_gate_failures"],
            "pass": bootstrap_summary["tail_gate_failure_fraction"]
            <= float(gates["maximum_propensity_bootstrap_fraction_tail_gate_failures"]),
        },
        {
            "check": "propensity bootstrap balance-gate failure fraction",
            "observed": bootstrap_summary["balance_gate_failure_fraction"],
            "threshold": gates["maximum_propensity_bootstrap_fraction_balance_gate_failures"],
            "pass": bootstrap_summary["balance_gate_failure_fraction"]
            <= float(gates["maximum_propensity_bootstrap_fraction_balance_gate_failures"]),
        },
    ])

    patient = pd.DataFrame({
        "patient_id_normalized": frame["patient_id_normalized"].astype(str),
        "analysis_treatment": treatment,
        "propensity_unclipped": propensity,
        "overlap_treatment_weight": np.where(
            treatment == 1,
            1.0 - propensity,
            propensity,
        ),
        "h": propensity * (1.0 - propensity),
    })

    write_csv(
        count_checks,
        table_dir / "s25c_99_frozen_v10_count_checks.csv",
    )
    write_csv(
        coefficients,
        table_dir / "s25c_99_propensity_coefficients.csv",
    )
    write_csv(
        balance,
        table_dir / "s25c_99_unclipped_ato_balance.csv",
    )
    write_csv(
        patient,
        table_dir / "s25c_99_unclipped_propensity_LOCAL_ONLY.csv",
    )
    write_csv(
        bootstrap_rows,
        table_dir / "s25c_99_propensity_bootstrap_feasibility.csv",
    )
    write_csv(
        checks,
        table_dir / "s25c_99_design_gates.csv",
    )
    write_json(
        {
            "fit_summary": fit_summary,
            "propensity_metrics": metrics,
            "max_abs_ato_weighted_smd": max_abs_smd,
            "propensity_bootstrap_summary": bootstrap_summary,
            "all_design_gates_pass": bool(checks["pass"].all()),
            "effect_estimated": False,
            "ato_score": config["ato_score"],
        },
        table_dir / "s25c_99_design_summary.json",
    )

    print("Frozen V10 checks")
    print(dataframe_console(count_checks))
    print("\nFull-sample unclipped propensity metrics")
    print(json.dumps(metrics, indent=2))
    print("\nUnclipped overlap balance")
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
    print("\nPropensity-only bootstrap feasibility")
    print(json.dumps(bootstrap_summary, indent=2))
    print("\nStage 25C design gates")
    print(dataframe_console(checks))

    if not bool(checks["pass"].all()):
        raise RuntimeError(
            "Stage 25C design gates failed. No Candidate V10 lock was created."
        )

    print(
        "\nPASS: unclipped stabilized overlap design validated. "
        "No treatment effect was estimated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
