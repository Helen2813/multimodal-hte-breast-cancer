from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path.cwd().resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(data),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
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


def canonical_sha256(data: object) -> str:
    raw = json.dumps(
        json_safe(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")



def restore_effect_regime_from_scenario_id(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repaired = frame.copy()

    if "scenario_id" not in repaired.columns:
        raise RuntimeError("scenario_id is missing from the checkpoint.")
    if "effect_regime" not in repaired.columns:
        raise RuntimeError("effect_regime is missing from the checkpoint.")

    scenario = repaired["scenario_id"].astype(str).str.strip()
    raw = repaired["effect_regime"].astype(str).str.strip()
    raw_lower = raw.str.lower()

    inferred = pd.Series("", index=repaired.index, dtype="object")
    inferred.loc[
        scenario.str.endswith("_NULL", na=False)
    ] = "null"
    inferred.loc[
        scenario.str.endswith(
            "_EMPIRICALLY_CALIBRATED_BENEFIT",
            na=False,
        )
    ] = "empirically_calibrated_benefit"

    unresolved = inferred.eq("")
    if bool(unresolved.any()):
        examples = scenario.loc[unresolved].drop_duplicates().head(10)
        raise RuntimeError(
            "Could not infer effect_regime from scenario_id for: "
            + ", ".join(examples.tolist())
        )

    blank_tokens = {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }
    # Literal "null" is valid only when scenario_id ends in _NULL.
    raw_is_blank_or_null_token = raw_lower.isin(blank_tokens)

    contradictory = (
        (~raw_is_blank_or_null_token)
        & raw_lower.ne(inferred)
    )
    if bool(contradictory.any()):
        sample = repaired.loc[
            contradictory,
            ["scenario_id", "effect_regime"],
        ].head(10)
        raise RuntimeError(
            "Checkpoint effect_regime contradicts scenario_id.\n"
            + sample.to_string(index=False)
        )

    audit = pd.DataFrame({
        "scenario_id": scenario,
        "raw_effect_regime": raw,
        "inferred_effect_regime": inferred,
        "cell_was_empty_or_na_token": raw_is_blank_or_null_token,
        "changed_by_repair": raw_lower.ne(inferred),
    })

    repaired["effect_regime"] = inferred
    return repaired, audit


def expected_scenario_ids(config: dict) -> set[str]:
    ids = set()
    for n in config["expected"]["sample_sizes"]:
        for sequencing in config["expected"]["sequencing_levels"]:
            for effect in config["expected"]["effect_regimes"]:
                ids.add(
                    f"N{int(n)}_{sequencing.upper()}_{effect.upper()}"
                )
    return ids


def summarize_checkpoint(
    checkpoint: pd.DataFrame,
) -> pd.DataFrame:
    checkpoint = checkpoint.copy()
    checkpoint["success_bool"] = as_bool(checkpoint["success"])

    group_columns = [
        "scenario_id",
        "sample_size",
        "sequencing_level",
        "sequencing_strength",
        "effect_regime",
        "treatment_log_hazard_effect",
        "method",
    ]
    rows: list[dict[str, Any]] = []

    for keys, group in checkpoint.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        successful = group[group["success_bool"]].copy()
        (
            scenario_id,
            sample_size,
            sequencing_level,
            sequencing_strength,
            effect_regime,
            treatment_effect,
            method,
        ) = keys

        row: dict[str, Any] = {
            "scenario_id": str(scenario_id),
            "sample_size": int(float(sample_size)),
            "sequencing_level": str(sequencing_level),
            "sequencing_strength": float(sequencing_strength),
            "effect_regime": str(effect_regime),
            "treatment_log_hazard_effect": float(treatment_effect),
            "method": str(method),
            "attempted_repetitions": int(len(group)),
            "successful_repetitions": int(len(successful)),
            "success_fraction": (
                float(len(successful) / len(group))
                if len(group)
                else np.nan
            ),
        }

        if len(successful):
            required_numeric = [
                "primary_truth_days",
                "estimate_days",
                "primary_bias_days",
                "primary_squared_error",
                "diagnostic_if_se_days",
                "primary_ci_low_days",
                "primary_ci_high_days",
                "included_covariate_max_abs_weighted_smd",
                "unweighted_chemo_smd",
                "weighted_chemo_smd",
                "ato_ess_fraction_treated",
                "ato_ess_fraction_control",
                "n_analysis",
                "treated",
                "control",
                "events",
                "treated_events",
                "control_events",
                "full_treated_fraction",
                "full_chemo_fraction_treated",
                "full_chemo_fraction_control",
                "strict_population_n",
                "strict_treated_n",
                "strict_control_n",
                "strict_treated_events",
                "strict_control_events",
                "secondary_truth_days",
                "secondary_bias_days",
                "target_drift_days",
                "residual_omitted_sequence_bias_days",
                "propensity_min",
                "propensity_p01",
                "propensity_p99",
                "propensity_max",
                "pseudo_p99",
                "pseudo_max",
            ]
            for column in required_numeric:
                if column in successful.columns:
                    successful[column] = pd.to_numeric(
                        successful[column],
                        errors="coerce",
                    )

            covered = as_bool(successful["primary_covered"])
            positive_exclusion = (
                successful["primary_ci_low_days"] > 0.0
            )
            negative_exclusion = (
                successful["primary_ci_high_days"] < 0.0
            )

            row.update({
                "mean_primary_truth_days": float(
                    successful["primary_truth_days"].mean()
                ),
                "mean_estimate_days": float(
                    successful["estimate_days"].mean()
                ),
                "bias_days": float(
                    successful["primary_bias_days"].mean()
                ),
                "empirical_sd_days": float(
                    successful["estimate_days"].std(ddof=1)
                ),
                "rmse_days": float(
                    math.sqrt(
                        successful["primary_squared_error"].mean()
                    )
                ),
                "mean_if_se_days": float(
                    successful["diagnostic_if_se_days"].mean()
                ),
                "primary_if_coverage": float(covered.mean()),
                "positive_ci_exclusion_rate": float(
                    positive_exclusion.mean()
                ),
                "negative_ci_exclusion_rate": float(
                    negative_exclusion.mean()
                ),
                "mean_included_covariate_max_abs_weighted_smd": float(
                    successful[
                        "included_covariate_max_abs_weighted_smd"
                    ].mean()
                ),
                "mean_unweighted_chemo_smd": float(
                    successful["unweighted_chemo_smd"].mean()
                ),
                "mean_weighted_chemo_smd": float(
                    successful["weighted_chemo_smd"].mean()
                ),
                "mean_ato_ess_fraction_treated": float(
                    successful["ato_ess_fraction_treated"].mean()
                ),
                "mean_ato_ess_fraction_control": float(
                    successful["ato_ess_fraction_control"].mean()
                ),
                "mean_n_analysis": float(
                    successful["n_analysis"].mean()
                ),
                "mean_events": float(
                    successful["events"].mean()
                ),
                "mean_treated_events": float(
                    successful["treated_events"].mean()
                ),
                "mean_control_events": float(
                    successful["control_events"].mean()
                ),
                "mean_full_treated_fraction": float(
                    successful["full_treated_fraction"].mean()
                ),
                "mean_full_chemo_fraction_treated": float(
                    successful["full_chemo_fraction_treated"].mean()
                ),
                "mean_full_chemo_fraction_control": float(
                    successful["full_chemo_fraction_control"].mean()
                ),
                "mean_strict_population_n": float(
                    successful["strict_population_n"].mean()
                ),
                "mean_strict_treated_n": float(
                    successful["strict_treated_n"].mean()
                ),
                "mean_strict_control_n": float(
                    successful["strict_control_n"].mean()
                ),
                "mean_strict_treated_events": float(
                    successful["strict_treated_events"].mean()
                ),
                "mean_strict_control_events": float(
                    successful["strict_control_events"].mean()
                ),
                "mean_secondary_truth_days": float(
                    successful["secondary_truth_days"].mean()
                )
                if successful["secondary_truth_days"].notna().any()
                else np.nan,
                "mean_secondary_bias_days": float(
                    successful["secondary_bias_days"].mean()
                )
                if successful["secondary_bias_days"].notna().any()
                else np.nan,
                "mean_target_drift_days": float(
                    successful["target_drift_days"].mean()
                )
                if successful["target_drift_days"].notna().any()
                else np.nan,
                "mean_residual_omitted_sequence_bias_days": float(
                    successful[
                        "residual_omitted_sequence_bias_days"
                    ].mean()
                )
                if successful[
                    "residual_omitted_sequence_bias_days"
                ].notna().any()
                else np.nan,
                "mean_propensity_min": float(
                    successful["propensity_min"].mean()
                ),
                "mean_propensity_p01": float(
                    successful["propensity_p01"].mean()
                ),
                "mean_propensity_p99": float(
                    successful["propensity_p99"].mean()
                ),
                "mean_propensity_max": float(
                    successful["propensity_max"].mean()
                ),
                "mean_pseudo_p99": float(
                    successful["pseudo_p99"].mean()
                ),
                "maximum_pseudo_max": float(
                    successful["pseudo_max"].max()
                ),
            })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "sample_size",
            "effect_regime",
            "sequencing_strength",
            "method",
        ]
    ).reset_index(drop=True)


def empirical_anchor_checks(
    summary: pd.DataFrame,
    stage33_config: dict,
) -> pd.DataFrame:
    expected = stage33_config["expected"]
    tolerances = stage33_config[
        "pilot_gates"
    ]["empirical_anchor_tolerances"]

    anchor = summary[
        (summary["sample_size"] == 559)
        & (summary["sequencing_level"] == "empirical")
        & (
            summary["effect_regime"]
            == "empirically_calibrated_benefit"
        )
        & (summary["method"] == "sequencing_aware")
    ]
    if len(anchor) != 1:
        raise RuntimeError(
            "Expected exactly one empirical-anchor sequencing-aware row."
        )
    row = anchor.iloc[0]

    checks = pd.DataFrame([
        {
            "metric": "full_treated_fraction",
            "observed": row["mean_full_treated_fraction"],
            "target": (
                expected["full_treated_anchor"]
                / expected["full_cohort_n_anchor"]
            ),
            "tolerance": tolerances["full_treated_fraction"],
        },
        {
            "metric": "chemo_fraction_treated",
            "observed": row[
                "mean_full_chemo_fraction_treated"
            ],
            "target": expected[
                "chemo_fraction_treated_anchor"
            ],
            "tolerance": tolerances[
                "chemo_fraction_treated"
            ],
        },
        {
            "metric": "chemo_fraction_control",
            "observed": row[
                "mean_full_chemo_fraction_control"
            ],
            "target": expected[
                "chemo_fraction_control_anchor"
            ],
            "tolerance": tolerances[
                "chemo_fraction_control"
            ],
        },
        {
            "metric": "strict_population_n",
            "observed": row["mean_strict_population_n"],
            "target": expected[
                "strict_sequence_eligible_n_anchor"
            ],
            "tolerance": tolerances["strict_population_n"],
        },
        {
            "metric": "strict_treated_n",
            "observed": row["mean_strict_treated_n"],
            "target": expected["strict_treated_anchor"],
            "tolerance": tolerances["strict_treated_n"],
        },
        {
            "metric": "strict_control_n",
            "observed": row["mean_strict_control_n"],
            "target": expected["strict_control_anchor"],
            "tolerance": tolerances["strict_control_n"],
        },
        {
            "metric": "strict_treated_events",
            "observed": row["mean_strict_treated_events"],
            "target": expected[
                "strict_treated_events_anchor"
            ],
            "tolerance": tolerances[
                "strict_treated_events"
            ],
        },
        {
            "metric": "strict_control_events",
            "observed": row["mean_strict_control_events"],
            "target": expected[
                "strict_control_events_anchor"
            ],
            "tolerance": tolerances[
                "strict_control_events"
            ],
        },
    ])
    checks["absolute_difference"] = (
        checks["observed"] - checks["target"]
    ).abs()
    checks["pass"] = (
        checks["absolute_difference"] <= checks["tolerance"]
    )
    return checks


def design_gates(
    summary: pd.DataFrame,
    stage33_config: dict,
) -> pd.DataFrame:
    gates = stage33_config["pilot_gates"]
    rows = []

    for _, row in summary.iterrows():
        method = row["method"]
        bias_limit = (
            gates["maximum_absolute_bias_adjusted_days"]
            if method == "adjusted_full"
            else gates[
                "maximum_absolute_bias_sequencing_aware_days"
            ]
            if method == "sequencing_aware"
            else None
        )
        rows.append({
            "scenario_id": row["scenario_id"],
            "effect_regime": row["effect_regime"],
            "method": method,
            "success_gate": bool(
                row["success_fraction"]
                >= gates["minimum_success_fraction"]
            ),
            "bias_gate_applicable": bias_limit is not None,
            "bias_gate": (
                bool(abs(row["bias_days"]) <= bias_limit)
                if bias_limit is not None
                else None
            ),
            "coverage_gate": bool(
                row["primary_if_coverage"]
                >= gates["minimum_primary_if_coverage"]
            ),
            "included_balance_gate": bool(
                row[
                    "mean_included_covariate_max_abs_weighted_smd"
                ]
                <= gates[
                    "maximum_included_covariate_weighted_smd"
                ]
            ),
        })
    return pd.DataFrame(rows)
