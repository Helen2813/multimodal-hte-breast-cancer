from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def root() -> Path:
    return Path.cwd().resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    project = root()
    config = load_json(project / "stage33c_pilot_decision_config.json")
    source = config["source"]
    output = config["output"]
    expected = config["expected"]

    print("=" * 128)
    print("STAGE 129 - AMEND STAGE 33 PILOT DECISION RULE")
    print("=" * 128)

    repaired = load_json(project / source["stage33b_final"])
    if repaired["stage33_simulation_id"] != expected["stage33_simulation_id"]:
        raise RuntimeError("Unexpected Stage 33 simulation ID.")
    if repaired["repair_id"] != expected["stage33b_repair_id"]:
        raise RuntimeError("Unexpected Stage 33B repair ID.")

    summary = pd.read_csv(
        project / source["stage33b_summary"],
        low_memory=False,
        keep_default_na=False,
    )
    if len(summary) != int(expected["summary_rows"]):
        raise RuntimeError(
            f"Expected {expected['summary_rows']} summary rows, found {len(summary)}."
        )

    anchors = pd.read_csv(
        project / source["stage33b_anchor_checks"],
        low_memory=False,
        keep_default_na=False,
    )
    anchors_pass = bool(
        anchors["pass"].astype(str).str.lower().eq("true").all()
    )

    valid_gates = config["valid_method_gates"]
    valid = summary[
        summary["method"].isin(expected["valid_methods"])
    ].copy()
    valid["success_gate"] = (
        pd.to_numeric(valid["success_fraction"], errors="raise")
        >= float(valid_gates["minimum_success_fraction"])
    )
    valid["bias_gate"] = (
        pd.to_numeric(valid["bias_days"], errors="raise").abs()
        <= float(valid_gates["maximum_absolute_bias_days"])
    )
    valid["coverage_gate"] = (
        pd.to_numeric(valid["primary_if_coverage"], errors="raise")
        >= float(valid_gates["minimum_if_coverage"])
    )
    valid["balance_gate"] = (
        pd.to_numeric(
            valid["mean_included_covariate_max_abs_weighted_smd"],
            errors="raise",
        )
        <= float(valid_gates["maximum_included_covariate_weighted_smd"])
    )
    valid["all_validity_gates"] = valid[
        ["success_gate", "bias_gate", "coverage_gate", "balance_gate"]
    ].all(axis=1)

    valid_out = valid[
        [
            "scenario_id",
            "sample_size",
            "sequencing_level",
            "effect_regime",
            "method",
            "bias_days",
            "primary_if_coverage",
            "success_fraction",
            "mean_included_covariate_max_abs_weighted_smd",
            "success_gate",
            "bias_gate",
            "coverage_gate",
            "balance_gate",
            "all_validity_gates",
        ]
    ].copy()
    write_csv(valid_out, project / output["valid_method_gates"])

    naive = summary[
        summary["method"] == expected["diagnostic_comparator"]
    ].copy()
    checks_cfg = config["diagnostic_comparator_checks"]
    checks: list[dict[str, Any]] = []

    checks.append({
        "check": "naive numerical success",
        "observed": float(
            pd.to_numeric(naive["success_fraction"], errors="raise").min()
        ),
        "criterion": (
            f">= {checks_cfg['minimum_success_fraction']}"
        ),
        "pass": bool(
            pd.to_numeric(naive["success_fraction"], errors="raise").min()
            >= float(checks_cfg["minimum_success_fraction"])
        ),
    })
    checks.append({
        "check": "naive included-variable balance",
        "observed": float(
            pd.to_numeric(
                naive["mean_included_covariate_max_abs_weighted_smd"],
                errors="raise",
            ).max()
        ),
        "criterion": (
            f"<= {checks_cfg['maximum_included_covariate_weighted_smd']}"
        ),
        "pass": bool(
            pd.to_numeric(
                naive["mean_included_covariate_max_abs_weighted_smd"],
                errors="raise",
            ).max()
            <= float(
                checks_cfg["maximum_included_covariate_weighted_smd"]
            )
        ),
    })

    no_sequence_null = naive[
        (naive["sequencing_level"] == "none")
        & (naive["effect_regime"] == "null")
    ]
    checks.append({
        "check": "naive null estimate without sequencing",
        "observed": float(
            pd.to_numeric(
                no_sequence_null["mean_estimate_days"], errors="raise"
            ).abs().max()
        ),
        "criterion": (
            "<= "
            + str(
                checks_cfg[
                    "no_sequence_null_maximum_absolute_mean_estimate_days"
                ]
            )
        ),
        "pass": bool(
            pd.to_numeric(
                no_sequence_null["mean_estimate_days"], errors="raise"
            ).abs().max()
            <= float(
                checks_cfg[
                    "no_sequence_null_maximum_absolute_mean_estimate_days"
                ]
            )
        ),
    })

    empirical_null = naive[
        (naive["sequencing_level"] == "empirical")
        & (naive["effect_regime"] == "null")
    ]
    checks.append({
        "check": "naive spurious positive null estimate with empirical sequencing",
        "observed": float(
            pd.to_numeric(
                empirical_null["mean_estimate_days"], errors="raise"
            ).min()
        ),
        "criterion": (
            ">= "
            + str(
                checks_cfg[
                    "empirical_sequence_null_minimum_positive_mean_estimate_days"
                ]
            )
        ),
        "pass": bool(
            pd.to_numeric(
                empirical_null["mean_estimate_days"], errors="raise"
            ).min()
            >= float(
                checks_cfg[
                    "empirical_sequence_null_minimum_positive_mean_estimate_days"
                ]
            )
        ),
    })

    empirical_all = naive[
        naive["sequencing_level"] == "empirical"
    ]
    checks.append({
        "check": "residual omitted-chemo imbalance under empirical sequencing",
        "observed": float(
            pd.to_numeric(
                empirical_all["mean_weighted_chemo_smd"], errors="raise"
            ).abs().min()
        ),
        "criterion": (
            ">= "
            + str(
                checks_cfg[
                    "empirical_sequence_minimum_absolute_weighted_chemo_smd"
                ]
            )
        ),
        "pass": bool(
            pd.to_numeric(
                empirical_all["mean_weighted_chemo_smd"], errors="raise"
            ).abs().min()
            >= float(
                checks_cfg[
                    "empirical_sequence_minimum_absolute_weighted_chemo_smd"
                ]
            )
        ),
    })

    empirical_benefit = naive[
        (naive["sequencing_level"] == "empirical")
        & (
            naive["effect_regime"]
            == "empirically_calibrated_benefit"
        )
    ]
    checks.append({
        "check": "naive positive bias in empirical benefit scenario",
        "observed": float(
            pd.to_numeric(
                empirical_benefit["bias_days"], errors="raise"
            ).min()
        ),
        "criterion": (
            ">= "
            + str(
                checks_cfg[
                    "empirical_sequence_benefit_minimum_positive_bias_days"
                ]
            )
        ),
        "pass": bool(
            pd.to_numeric(
                empirical_benefit["bias_days"], errors="raise"
            ).min()
            >= float(
                checks_cfg[
                    "empirical_sequence_benefit_minimum_positive_bias_days"
                ]
            )
        ),
    })

    diagnostic = pd.DataFrame(checks)
    write_csv(diagnostic, project / output["diagnostic_checks"])

    valid_pass = bool(valid_out["all_validity_gates"].all())
    diagnostic_pass = bool(diagnostic["pass"].all())
    ready = bool(valid_pass and diagnostic_pass and anchors_pass)

    locked_files = []
    for relative in [
        "stage33c_pilot_decision_config.json",
        "scripts/129_stage33c_amend_pilot_decision.py",
        "run_stage33c_pilot_decision_amendment.ps1",
        source["stage33b_manifest"],
        source["stage33b_summary"],
        source["stage33b_anchor_checks"],
        source["stage33b_final"],
    ]:
        path = project / relative
        locked_files.append({
            "path": relative.replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })

    manifest = {
        "status": "STAGE33C_PILOT_DECISION_AMENDMENT_LOCKED_AND_APPLIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage33_simulation_id": expected["stage33_simulation_id"],
        "stage33b_repair_id": expected["stage33b_repair_id"],
        "decision_principle": config["decision_principle"],
        "boundary": config["boundary"],
        "locked_files": locked_files,
    }
    manifest["decision_id"] = (
        "PAPER_A_STAGE33C_DECISION_"
        + hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:16].upper()
    )
    write_json(manifest, project / output["manifest"])

    final = {
        "status": (
            "STAGE33C_READY_FOR_INDEPENDENT_CONFIRMATORY_SIMULATION"
            if ready
            else "STAGE33C_REQUIRES_REVIEW"
        ),
        "decision_id": manifest["decision_id"],
        "ready_for_independent_confirmatory_simulation": ready,
        "valid_methods_pass": valid_pass,
        "diagnostic_comparator_checks_pass": diagnostic_pass,
        "empirical_anchor_checks_pass": anchors_pass,
        "valid_method_gates": valid_out.to_dict("records"),
        "diagnostic_comparator_checks": diagnostic.to_dict("records"),
        "decision_principle": config["decision_principle"],
        "boundary": config["boundary"],
    }
    write_json(final, project / output["final_json"])

    print("Valid-method gates")
    print(valid_out.to_string(index=False))
    print("\nNaive diagnostic-comparator checks")
    print(diagnostic.to_string(index=False))
    print("\nFinal decision")
    print(json.dumps(final, indent=2))
    print(
        "\nPASS: Stage 33 pilot decision rule documented without "
        "rerunning any simulation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
