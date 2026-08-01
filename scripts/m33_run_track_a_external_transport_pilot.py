from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from _metabric_m6_resume_utils import (
    exact_tcga_stage_indicator,
    fast_harrell_c_index,
    resolve_tcga_id_pair,
    tcga_node_positive_indicator,
)
from _metabric_m6_utils import (
    binary_auc_at_horizon,
    fit_cox_risk,
    load_config,
    median_impute_scale_train_test,
    normalize_tcga_id,
    out_dir,
    print_table,
    project_root,
    rank_normalize_separately,
    write_csv,
)


def build_tcga_clinical(frame: pd.DataFrame) -> pd.DataFrame:
    age_column = next(
        (
            column for column in frame.columns
            if str(column).startswith("CLIN_age_at_index")
        ),
        None,
    )
    result = pd.DataFrame(index=frame.index)
    result["age"] = (
        pd.to_numeric(frame[age_column], errors="coerce")
        if age_column else np.nan
    )
    result["stage_II"] = exact_tcga_stage_indicator(frame, "II")
    result["stage_III"] = exact_tcga_stage_indicator(frame, "III")
    result["stage_IV"] = exact_tcga_stage_indicator(frame, "IV")
    result["node_positive"] = tcga_node_positive_indicator(frame)
    return result


def build_metabric_clinical(frame: pd.DataFrame) -> pd.DataFrame:
    stage = pd.to_numeric(frame["stage"], errors="coerce")
    nodes = pd.to_numeric(frame["positive_nodes"], errors="coerce")
    result = pd.DataFrame(index=frame.index)
    result["age"] = pd.to_numeric(frame["age_at_diagnosis"], errors="coerce")
    result["stage_II"] = (stage == 2).astype(float)
    result["stage_III"] = (stage == 3).astype(float)
    result["stage_IV"] = (stage == 4).astype(float)
    result["node_positive"] = (nodes > 0).astype(float)
    result.loc[nodes.isna(), "node_positive"] = np.nan
    return result


def bootstrap_all_models(
    time: np.ndarray,
    event: np.ndarray,
    risks: dict[str, np.ndarray],
    horizon: float,
    repetitions: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    n = len(time)
    metric_rows = []
    delta_rows = []
    model_names = list(risks)
    if "clinical" not in risks:
        raise RuntimeError("Clinical risk is required for paired bootstrap deltas.")

    for repetition in range(1, repetitions + 1):
        indices = rng.integers(0, n, n)
        metrics = {}
        for model_name in model_names:
            c_index = fast_harrell_c_index(
                time[indices],
                event[indices],
                risks[model_name][indices],
            )
            auc_5y, auc_n = binary_auc_at_horizon(
                time[indices],
                event[indices],
                risks[model_name][indices],
                horizon,
            )
            metrics[model_name] = {
                "c_index": c_index,
                "auc_5y": auc_5y,
            }
            metric_rows.append({
                "repetition": repetition,
                "model_set": model_name,
                "c_index": c_index,
                "auc_5y": auc_5y,
                "auc_5y_n": auc_n,
            })

        for model_name in model_names:
            if model_name == "clinical":
                continue
            delta_rows.append({
                "repetition": repetition,
                "model_set": model_name,
                "delta_c_index_vs_clinical": (
                    metrics[model_name]["c_index"]
                    - metrics["clinical"]["c_index"]
                ),
                "delta_auc_5y_vs_clinical": (
                    metrics[model_name]["auc_5y"]
                    - metrics["clinical"]["auc_5y"]
                ),
            })
    return metric_rows, delta_rows


def summarize_bootstrap(rows: list[dict], value_columns: list[str]) -> list[dict]:
    summary = []
    model_names = sorted({row["model_set"] for row in rows})
    for model_name in model_names:
        subset = [row for row in rows if row["model_set"] == model_name]
        for column in value_columns:
            values = np.asarray(
                [row[column] for row in subset],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            summary.append({
                "model_set": model_name,
                "metric": column,
                "bootstrap_repetitions": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "fraction_positive": float(np.mean(values > 0)),
            })
    return summary


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["track_a_pilot"]

    print("=" * 124)
    print("METABRIC M6.33R - ROBUST TRACK A FIXED-PANEL EXTERNAL TRANSPORT PILOT")
    print("=" * 124)

    tcga = pd.read_csv(root / cfg["tcga_canonical_table"], low_memory=False)
    outcome = pd.read_csv(root / cfg["tcga_outcome_file"], low_memory=False)

    tcga_id_column, outcome_id_column, id_pair_rows = resolve_tcga_id_pair(
        tcga,
        outcome,
        int(settings["minimum_tcga_alignment"]),
    )
    write_csv(out / "m33_tcga_id_alignment_candidates.csv", id_pair_rows)

    time_source = next(
        column for column in outcome.columns
        if str(column).lower() in {"os.time", "os_time"}
    )
    event_source = next(
        column for column in outcome.columns
        if str(column).lower() == "os"
    )

    tcga = tcga.copy()
    outcome = outcome.copy()
    tcga["__id"] = tcga[tcga_id_column].map(normalize_tcga_id)
    outcome["__id"] = outcome[outcome_id_column].map(normalize_tcga_id)

    tcga_duplicate_rows = int(tcga["__id"].duplicated(keep=False).sum())
    outcome_duplicate_rows = int(outcome["__id"].duplicated(keep=False).sum())
    if tcga_duplicate_rows or outcome_duplicate_rows:
        raise RuntimeError(
            "TCGA ID alignment is not one-to-one: "
            f"tcga duplicate rows={tcga_duplicate_rows}, "
            f"outcome duplicate rows={outcome_duplicate_rows}."
        )

    outcome_payload = outcome[
        ["__id", time_source, event_source]
    ].rename(
        columns={
            time_source: "__tcga_os_time",
            event_source: "__tcga_os_event",
        }
    )
    tcga = tcga.drop(
        columns=["__tcga_os_time", "__tcga_os_event"],
        errors="ignore",
    ).merge(
        outcome_payload,
        on="__id",
        how="inner",
        validate="one_to_one",
    )
    tcga["__tcga_os_time"] = pd.to_numeric(
        tcga["__tcga_os_time"],
        errors="coerce",
    )
    tcga["__tcga_os_event"] = pd.to_numeric(
        tcga["__tcga_os_event"],
        errors="coerce",
    )
    tcga = tcga[
        tcga["__tcga_os_time"].notna()
        & tcga["__tcga_os_event"].notna()
    ].reset_index(drop=True)

    metabric = pd.read_csv(
        root
        / cfg["metabric_m2_dir"]
        / "m06_metabric_clinical_master_LOCAL_ONLY.csv",
        low_memory=False,
    )
    metabric = metabric[
        pd.to_numeric(metabric["os_months"], errors="coerce").notna()
        & pd.to_numeric(metabric["os_event"], errors="coerce").notna()
    ].copy()

    fixed_rna = pd.read_csv(
        root / cfg["files"]["fixed_rna_matrix"],
        low_memory=False,
    )
    fixed_cna = pd.read_csv(
        root / cfg["files"]["fixed_cna_matrix"],
        low_memory=False,
    )
    primary_panel = pd.read_csv(
        root / cfg["files"]["primary_transport_panel"],
        dtype=str,
    )

    rna_ids = set(
        primary_panel.loc[
            primary_panel["modality"] == "rna",
            "ensembl_id",
        ].astype(str)
    )
    cna_ids = set(
        primary_panel.loc[
            primary_panel["modality"] == "cna",
            "ensembl_id",
        ].astype(str)
    )

    tcga_rna_columns = [
        column for column in tcga.columns
        if str(column).startswith("RNA_")
        and any(ensembl_id in str(column) for ensembl_id in rna_ids)
    ]
    tcga_cna_columns = [
        column for column in tcga.columns
        if str(column).startswith(("CNV_", "CNA_"))
        and any(ensembl_id in str(column) for ensembl_id in cna_ids)
    ]
    metabric_rna_columns = [
        column for column in fixed_rna.columns
        if str(column).startswith("TCGA_RNA_")
        and any(ensembl_id in str(column) for ensembl_id in rna_ids)
    ]
    metabric_cna_columns = [
        column for column in fixed_cna.columns
        if str(column).startswith("TCGA_CNA_")
        and any(ensembl_id in str(column) for ensembl_id in cna_ids)
    ]

    def ensembl_key(column: str) -> str:
        match = re.search(r"ENSG\d+", str(column))
        return match.group(0) if match else str(column)

    tcga_rna_map = {
        ensembl_key(column): column for column in tcga_rna_columns
    }
    metabric_rna_map = {
        ensembl_key(column): column for column in metabric_rna_columns
    }
    tcga_cna_map = {
        ensembl_key(column): column for column in tcga_cna_columns
    }
    metabric_cna_map = {
        ensembl_key(column): column for column in metabric_cna_columns
    }
    shared_rna = sorted(set(tcga_rna_map) & set(metabric_rna_map))
    shared_cna = sorted(set(tcga_cna_map) & set(metabric_cna_map))

    fixed_rna["sample_id"] = fixed_rna["sample_id"].astype(str)
    fixed_cna["sample_id"] = fixed_cna["sample_id"].astype(str)
    metabric["sample_id"] = metabric["sample_id"].astype(str)
    metabric = metabric.merge(
        fixed_rna[
            ["sample_id"]
            + [metabric_rna_map[identifier] for identifier in shared_rna]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    metabric = metabric.merge(
        fixed_cna[
            ["sample_id"]
            + [metabric_cna_map[identifier] for identifier in shared_cna]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if len(tcga) < int(settings["minimum_tcga_alignment"]):
        raise RuntimeError(
            f"TCGA aligned n={len(tcga)} is below the prespecified minimum."
        )
    if len(metabric) < int(settings["minimum_metabric_n"]):
        raise RuntimeError(
            f"METABRIC aligned n={len(metabric)} is below the prespecified minimum."
        )
    if len(shared_rna) == 0 or len(shared_cna) == 0:
        raise RuntimeError(
            f"Transport panel is empty: RNA={len(shared_rna)}, CNA={len(shared_cna)}."
        )

    tcga_clinical = build_tcga_clinical(tcga)
    metabric_clinical = build_metabric_clinical(metabric)

    tcga_rna = tcga[
        [tcga_rna_map[identifier] for identifier in shared_rna]
    ].copy()
    metabric_rna = metabric[
        [metabric_rna_map[identifier] for identifier in shared_rna]
    ].copy()
    tcga_rna.columns = shared_rna
    metabric_rna.columns = shared_rna

    tcga_cna = tcga[
        [tcga_cna_map[identifier] for identifier in shared_cna]
    ].copy()
    metabric_cna = metabric[
        [metabric_cna_map[identifier] for identifier in shared_cna]
    ].copy()
    tcga_cna.columns = shared_cna
    metabric_cna.columns = shared_cna

    tcga_rna = rank_normalize_separately(tcga_rna)
    metabric_rna = rank_normalize_separately(metabric_rna)
    tcga_cna, metabric_cna, cna_parameters = (
        median_impute_scale_train_test(
            tcga_cna,
            metabric_cna,
            scale=True,
        )
    )
    tcga_clinical, metabric_clinical, clinical_parameters = (
        median_impute_scale_train_test(
            tcga_clinical,
            metabric_clinical,
            scale=True,
        )
    )

    model_sets = {
        "clinical": (
            tcga_clinical,
            metabric_clinical,
        ),
        "clinical_rna": (
            pd.concat(
                [tcga_clinical, tcga_rna.add_prefix("RNA_")],
                axis=1,
            ),
            pd.concat(
                [metabric_clinical, metabric_rna.add_prefix("RNA_")],
                axis=1,
            ),
        ),
        "clinical_cna": (
            pd.concat(
                [tcga_clinical, tcga_cna.add_prefix("CNA_")],
                axis=1,
            ),
            pd.concat(
                [metabric_clinical, metabric_cna.add_prefix("CNA_")],
                axis=1,
            ),
        ),
        "clinical_rna_cna": (
            pd.concat(
                [
                    tcga_clinical,
                    tcga_rna.add_prefix("RNA_"),
                    tcga_cna.add_prefix("CNA_"),
                ],
                axis=1,
            ),
            pd.concat(
                [
                    metabric_clinical,
                    metabric_rna.add_prefix("RNA_"),
                    metabric_cna.add_prefix("CNA_"),
                ],
                axis=1,
            ),
        ),
    }

    tcga_time = tcga["__tcga_os_time"].to_numpy(dtype=float)
    tcga_event = tcga["__tcga_os_event"].to_numpy(dtype=int)
    metabric_time = pd.to_numeric(
        metabric["os_months"],
        errors="coerce",
    ).to_numpy(dtype=float)
    metabric_event = pd.to_numeric(
        metabric["os_event"],
        errors="coerce",
    ).to_numpy(dtype=int)

    result_rows = []
    coefficient_registry = {}
    risk_registry = {}
    for model_name in settings["model_sets"]:
        train_x, test_x = model_sets[model_name]
        risk, fit_information = fit_cox_risk(
            train_x,
            tcga_time,
            tcga_event,
            test_x,
            float(settings["cox_penalizer"]),
        )
        c_index = fast_harrell_c_index(
            metabric_time,
            metabric_event,
            risk,
        )
        auc_5y, auc_n = binary_auc_at_horizon(
            metabric_time,
            metabric_event,
            risk,
            float(settings["five_year_months"]),
        )
        result_rows.append({
            "model_set": model_name,
            "tcga_n": len(tcga),
            "metabric_n": len(metabric),
            "features": train_x.shape[1],
            "rna_features": len(shared_rna) if "rna" in model_name else 0,
            "cna_features": len(shared_cna) if "cna" in model_name else 0,
            "harrell_c_index": c_index,
            "auc_5y": auc_5y,
            "auc_5y_n": auc_n,
            "tcga_train_c_index": fit_information["concordance_train"],
        })
        coefficient_registry[model_name] = fit_information
        risk_registry[model_name] = risk
        print(
            f"{model_name:24s} features={train_x.shape[1]:3d} "
            f"TCGA train C={fit_information['concordance_train']:.4f} "
            f"METABRIC C={c_index:.4f} AUC5y={auc_5y:.4f}"
        )

    bootstrap_rows, delta_rows = bootstrap_all_models(
        metabric_time,
        metabric_event,
        risk_registry,
        float(settings["five_year_months"]),
        int(settings["bootstrap_repetitions"]),
        int(settings["bootstrap_seed"]),
    )
    bootstrap_summary = summarize_bootstrap(
        bootstrap_rows,
        ["c_index", "auc_5y"],
    )
    delta_summary = summarize_bootstrap(
        delta_rows,
        ["delta_c_index_vs_clinical", "delta_auc_5y_vs_clinical"],
    )

    write_csv(out / "m33_track_a_external_results.csv", result_rows)
    write_csv(out / "m33_track_a_bootstrap_pilot.csv", bootstrap_rows)
    write_csv(out / "m33_track_a_bootstrap_summary.csv", bootstrap_summary)
    write_csv(out / "m33_track_a_paired_bootstrap_deltas.csv", delta_rows)
    write_csv(out / "m33_track_a_paired_delta_summary.csv", delta_summary)
    (out / "m33_track_a_coefficients.json").write_text(
        json.dumps(coefficient_registry, indent=2),
        encoding="utf-8",
    )
    alignment = {
        "tcga_id_column": tcga_id_column,
        "outcome_id_column": outcome_id_column,
        "tcga_aligned_n": len(tcga),
        "metabric_os_n": len(metabric),
        "shared_primary_rna_features": shared_rna,
        "shared_primary_cna_features": shared_cna,
        "clinical_bridge_features": list(tcga_clinical.columns),
        "clinical_preprocessing": clinical_parameters,
        "cna_preprocessing": cna_parameters,
        "tcga_os_time_range": [
            float(np.min(tcga_time)),
            float(np.max(tcga_time)),
        ],
        "metabric_os_month_range": [
            float(np.min(metabric_time)),
            float(np.max(metabric_time)),
        ],
    }
    (out / "m33_track_a_alignment.json").write_text(
        json.dumps(alignment, indent=2),
        encoding="utf-8",
    )

    print("\nTop TCGA ID alignment candidates")
    print_table(
        id_pair_rows,
        [
            "left_column", "right_column", "overlap",
            "left_coverage", "right_coverage"
        ],
        max_rows=12,
    )

    print("\nTrack A external results")
    print_table(
        result_rows,
        [
            "model_set", "tcga_n", "metabric_n", "features",
            "rna_features", "cna_features", "tcga_train_c_index",
            "harrell_c_index", "auc_5y"
        ],
    )

    print("\nPilot bootstrap intervals")
    print_table(
        bootstrap_summary,
        [
            "model_set", "metric", "mean", "sd",
            "ci_low", "ci_high", "fraction_positive"
        ],
    )

    print("\nPaired bootstrap deltas versus clinical-only")
    print_table(
        delta_summary,
        [
            "model_set", "metric", "mean", "sd",
            "ci_low", "ci_high", "fraction_positive"
        ],
    )

    print("\nPASS: robust Track A pilot completed without rerunning M31/M32.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
