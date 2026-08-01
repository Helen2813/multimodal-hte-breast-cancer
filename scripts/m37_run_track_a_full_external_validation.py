from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from _metabric_m7_utils import (
    binary_auc_at_horizon,
    build_metabric_clinical,
    build_tcga_clinical,
    external_calibration_slope,
    fast_harrell_c_index,
    fit_external_cox,
    infer_time_unit_and_convert_to_months,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_dynamic_auc,
    km_survival_at,
    load_config,
    median_impute_scale_train_test,
    out_dir,
    print_table,
    project_root,
    rank_normalize_separately,
    read_rows,
    resolve_tcga_id_pair,
    uno_c_index,
    write_csv,
)


def bootstrap_all_models(
    time: np.ndarray,
    event: np.ndarray,
    risks: dict[str, np.ndarray],
    primary_horizon: float,
    secondary_horizon: float,
    repetitions: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    model_names = list(risks)
    rows = []
    deltas = []
    n = len(time)
    for repetition in range(1, repetitions + 1):
        indices = rng.integers(0, n, n)
        metrics = {}
        for model_name in model_names:
            c_index = fast_harrell_c_index(
                time[indices],
                event[indices],
                risks[model_name][indices],
            )
            auc_5y, auc_5y_n = binary_auc_at_horizon(
                time[indices],
                event[indices],
                risks[model_name][indices],
                primary_horizon,
            )
            auc_10y, auc_10y_n = binary_auc_at_horizon(
                time[indices],
                event[indices],
                risks[model_name][indices],
                secondary_horizon,
            )
            metrics[model_name] = {
                "c_index": c_index,
                "auc_5y": auc_5y,
                "auc_10y": auc_10y,
            }
            rows.append({
                "repetition": repetition,
                "model_set": model_name,
                "c_index": c_index,
                "auc_5y": auc_5y,
                "auc_5y_n": auc_5y_n,
                "auc_10y": auc_10y,
                "auc_10y_n": auc_10y_n,
            })
        for model_name in model_names:
            if model_name == "clinical":
                continue
            deltas.append({
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
                "delta_auc_10y_vs_clinical": (
                    metrics[model_name]["auc_10y"]
                    - metrics["clinical"]["auc_10y"]
                ),
            })
    return rows, deltas


def summarize(
    rows: list[dict],
    metrics: list[str],
) -> list[dict]:
    output = []
    for model_name in sorted({
        row["model_set"] for row in rows
    }):
        subset = [
            row for row in rows
            if row["model_set"] == model_name
        ]
        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in subset],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            output.append({
                "model_set": model_name,
                "metric": metric,
                "repetitions": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "fraction_positive": float(np.mean(values > 0)),
            })
    return output


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["track_a"]

    print("=" * 124)
    print("METABRIC M7.37 - FULL TRACK A FIXED-PANEL EXTERNAL VALIDATION")
    print("=" * 124)

    tcga = pd.read_csv(
        root / cfg["tcga_canonical_table"],
        low_memory=False,
    )
    outcome = pd.read_csv(
        root / cfg["tcga_outcome_file"],
        low_memory=False,
    )
    left_id, right_id, id_rows = resolve_tcga_id_pair(
        tcga,
        outcome,
        int(settings["minimum_tcga_alignment"]),
    )
    write_csv(out / "m37_tcga_id_alignment.csv", id_rows)

    time_source = next(
        column for column in outcome.columns
        if str(column).lower() in {"os.time", "os_time"}
    )
    event_source = next(
        column for column in outcome.columns
        if str(column).lower() == "os"
    )
    tcga["__id"] = tcga[left_id].map(
        lambda value: re.search(
            r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
            str(value).upper().replace("_", "-"),
        ).group(1)
        if re.search(
            r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
            str(value).upper().replace("_", "-"),
        )
        else str(value).upper()
    )
    outcome["__id"] = outcome[right_id].map(
        lambda value: re.search(
            r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
            str(value).upper().replace("_", "-"),
        ).group(1)
        if re.search(
            r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
            str(value).upper().replace("_", "-"),
        )
        else str(value).upper()
    )
    payload = outcome[
        ["__id", time_source, event_source]
    ].rename(columns={
        time_source: "__time_raw",
        event_source: "__event",
    })
    tcga = tcga.merge(
        payload,
        on="__id",
        how="inner",
        validate="one_to_one",
    )
    tcga["__time_raw"] = pd.to_numeric(
        tcga["__time_raw"],
        errors="coerce",
    )
    tcga["__event"] = pd.to_numeric(
        tcga["__event"],
        errors="coerce",
    )
    tcga = tcga[
        tcga["__time_raw"].notna()
        & tcga["__event"].notna()
    ].reset_index(drop=True)
    tcga_time, time_unit = infer_time_unit_and_convert_to_months(
        tcga["__time_raw"].to_numpy(dtype=float)
    )
    tcga_event = tcga["__event"].to_numpy(dtype=int)

    metabric = pd.read_csv(
        root / cfg["files"]["clinical_master"],
        low_memory=False,
    )
    metabric = metabric[
        pd.to_numeric(metabric["os_months"], errors="coerce").notna()
        & pd.to_numeric(metabric["os_event"], errors="coerce").notna()
    ].copy()
    metabric_time = pd.to_numeric(
        metabric["os_months"],
        errors="coerce",
    ).to_numpy(dtype=float)
    metabric_event = pd.to_numeric(
        metabric["os_event"],
        errors="coerce",
    ).to_numpy(dtype=int)

    fixed_rna = pd.read_csv(
        root / cfg["files"]["fixed_rna_matrix"],
        low_memory=False,
    )
    fixed_cna = pd.read_csv(
        root / cfg["files"]["fixed_cna_matrix"],
        low_memory=False,
    )
    panel = pd.read_csv(
        root / cfg["files"]["primary_transport_panel"],
        dtype=str,
    )
    rna_ids = set(
        panel.loc[
            panel["modality"] == "rna",
            "ensembl_id",
        ].astype(str)
    )
    cna_ids = set(
        panel.loc[
            panel["modality"] == "cna",
            "ensembl_id",
        ].astype(str)
    )

    def ensembl_key(column: str) -> str:
        match = re.search(r"ENSG\d+", str(column))
        return match.group(0) if match else str(column)

    tcga_rna_map = {
        ensembl_key(column): column
        for column in tcga.columns
        if str(column).startswith("RNA_")
        and any(identifier in str(column) for identifier in rna_ids)
    }
    tcga_cna_map = {
        ensembl_key(column): column
        for column in tcga.columns
        if str(column).startswith(("CNV_", "CNA_"))
        and any(identifier in str(column) for identifier in cna_ids)
    }
    meta_rna_map = {
        ensembl_key(column): column
        for column in fixed_rna.columns
        if str(column).startswith("TCGA_RNA_")
        and any(identifier in str(column) for identifier in rna_ids)
    }
    meta_cna_map = {
        ensembl_key(column): column
        for column in fixed_cna.columns
        if str(column).startswith("TCGA_CNA_")
        and any(identifier in str(column) for identifier in cna_ids)
    }
    shared_rna = sorted(
        set(tcga_rna_map) & set(meta_rna_map)
    )
    shared_cna = sorted(
        set(tcga_cna_map) & set(meta_cna_map)
    )

    metabric["sample_id"] = metabric["sample_id"].astype(str)
    fixed_rna["sample_id"] = fixed_rna["sample_id"].astype(str)
    fixed_cna["sample_id"] = fixed_cna["sample_id"].astype(str)
    metabric = metabric.merge(
        fixed_rna[
            ["sample_id"]
            + [meta_rna_map[key] for key in shared_rna]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    metabric = metabric.merge(
        fixed_cna[
            ["sample_id"]
            + [meta_cna_map[key] for key in shared_cna]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if len(tcga) < int(settings["minimum_tcga_alignment"]):
        raise RuntimeError("TCGA alignment below locked minimum.")
    if len(metabric) < int(settings["minimum_metabric_n"]):
        raise RuntimeError("METABRIC cohort below locked minimum.")

    tcga_clinical = build_tcga_clinical(tcga)
    meta_clinical = build_metabric_clinical(metabric)
    tcga_clinical, meta_clinical, clinical_parameters = (
        median_impute_scale_train_test(
            tcga_clinical,
            meta_clinical,
            scale=True,
        )
    )

    tcga_rna = tcga[
        [tcga_rna_map[key] for key in shared_rna]
    ].copy()
    meta_rna = metabric[
        [meta_rna_map[key] for key in shared_rna]
    ].copy()
    tcga_rna.columns = shared_rna
    meta_rna.columns = shared_rna
    tcga_rna = rank_normalize_separately(tcga_rna)
    meta_rna = rank_normalize_separately(meta_rna)

    tcga_cna = tcga[
        [tcga_cna_map[key] for key in shared_cna]
    ].copy()
    meta_cna = metabric[
        [meta_cna_map[key] for key in shared_cna]
    ].copy()
    tcga_cna.columns = shared_cna
    meta_cna.columns = shared_cna
    tcga_cna, meta_cna, cna_parameters = (
        median_impute_scale_train_test(
            tcga_cna,
            meta_cna,
            scale=True,
        )
    )

    model_sets = {
        "clinical": (
            tcga_clinical,
            meta_clinical,
        ),
        "clinical_rna": (
            pd.concat(
                [tcga_clinical, tcga_rna.add_prefix("RNA_")],
                axis=1,
            ),
            pd.concat(
                [meta_clinical, meta_rna.add_prefix("RNA_")],
                axis=1,
            ),
        ),
        "clinical_cna": (
            pd.concat(
                [tcga_clinical, tcga_cna.add_prefix("CNA_")],
                axis=1,
            ),
            pd.concat(
                [meta_clinical, meta_cna.add_prefix("CNA_")],
                axis=1,
            ),
        ),
        "clinical_rna_cna": (
            pd.concat([
                tcga_clinical,
                tcga_rna.add_prefix("RNA_"),
                tcga_cna.add_prefix("CNA_"),
            ], axis=1),
            pd.concat([
                meta_clinical,
                meta_rna.add_prefix("RNA_"),
                meta_cna.add_prefix("CNA_"),
            ], axis=1),
        ),
    }

    ibs_grid = np.linspace(
        float(settings["ibs_start_months"]),
        float(settings["ibs_stop_months"]),
        int(settings["ibs_grid_points"]),
    )
    prediction_times = np.unique(np.concatenate([
        ibs_grid,
        np.asarray([
            float(settings["primary_horizon_months"]),
            float(settings["secondary_horizon_months"]),
        ]),
    ]))

    results = []
    risks = {}
    coefficients = {}
    prediction_registry = {}
    for model_name in settings["model_sets"]:
        train_x, test_x = model_sets[model_name]
        risk, survival, fit_information = fit_external_cox(
            train_x,
            tcga_time,
            tcga_event,
            test_x,
            float(settings["cox_penalizer"]),
            prediction_times,
        )
        risks[model_name] = risk
        coefficients[model_name] = fit_information
        prediction_registry[model_name] = survival

        index_5 = int(np.where(
            np.isclose(
                prediction_times,
                float(settings["primary_horizon_months"]),
            )
        )[0][0])
        index_10 = int(np.where(
            np.isclose(
                prediction_times,
                float(settings["secondary_horizon_months"]),
            )
        )[0][0])
        survival_5 = survival[:, index_5]
        survival_10 = survival[:, index_10]
        ibs_indices = [
            int(np.where(np.isclose(prediction_times, value))[0][0])
            for value in ibs_grid
        ]
        survival_grid = survival[:, ibs_indices]

        observed_survival = km_survival_at(
            metabric_time,
            metabric_event,
            np.asarray([
                float(settings["primary_horizon_months"]),
                float(settings["secondary_horizon_months"]),
            ]),
        )
        auc_5, auc_5_n = binary_auc_at_horizon(
            metabric_time,
            metabric_event,
            risk,
            float(settings["primary_horizon_months"]),
        )
        auc_10, auc_10_n = binary_auc_at_horizon(
            metabric_time,
            metabric_event,
            risk,
            float(settings["secondary_horizon_months"]),
        )
        row = {
            "model_set": model_name,
            "tcga_n": len(tcga),
            "metabric_n": len(metabric),
            "features": train_x.shape[1],
            "rna_features": len(shared_rna) if "rna" in model_name else 0,
            "cna_features": len(shared_cna) if "cna" in model_name else 0,
            "tcga_train_c_index": fit_information["concordance_train"],
            "harrell_c_index": fast_harrell_c_index(
                metabric_time,
                metabric_event,
                risk,
            ),
            "uno_c_10y": uno_c_index(
                metabric_time,
                metabric_event,
                risk,
                float(settings["uno_tau_months"]),
            ),
            "binary_auc_5y": auc_5,
            "binary_auc_5y_n": auc_5_n,
            "binary_auc_10y": auc_10,
            "binary_auc_10y_n": auc_10_n,
            "ipcw_auc_5y": ipcw_dynamic_auc(
                metabric_time,
                metabric_event,
                risk,
                float(settings["primary_horizon_months"]),
            ),
            "ipcw_auc_10y": ipcw_dynamic_auc(
                metabric_time,
                metabric_event,
                risk,
                float(settings["secondary_horizon_months"]),
            ),
            "brier_5y": ipcw_brier_score(
                metabric_time,
                metabric_event,
                survival_5,
                float(settings["primary_horizon_months"]),
            ),
            "brier_10y": ipcw_brier_score(
                metabric_time,
                metabric_event,
                survival_10,
                float(settings["secondary_horizon_months"]),
            ),
            "integrated_brier_1_to_10y": integrated_brier_score(
                metabric_time,
                metabric_event,
                survival_grid,
                ibs_grid,
            ),
            "calibration_slope": external_calibration_slope(
                metabric_time,
                metabric_event,
                np.log(np.clip(risk, 1e-12, None)),
            ),
            "mean_predicted_survival_5y": float(np.mean(survival_5)),
            "observed_km_survival_5y": float(observed_survival[0]),
            "observed_minus_predicted_survival_5y": float(
                observed_survival[0] - np.mean(survival_5)
            ),
            "mean_predicted_survival_10y": float(np.mean(survival_10)),
            "observed_km_survival_10y": float(observed_survival[1]),
            "observed_minus_predicted_survival_10y": float(
                observed_survival[1] - np.mean(survival_10)
            ),
        }
        results.append(row)
        print(
            f"{model_name:24s} "
            f"C={row['harrell_c_index']:.4f} "
            f"UnoC10={row['uno_c_10y']:.4f} "
            f"AUC5={row['ipcw_auc_5y']:.4f} "
            f"IBS={row['integrated_brier_1_to_10y']:.4f} "
            f"slope={row['calibration_slope']:.4f}"
        )

    bootstrap_rows, delta_rows = bootstrap_all_models(
        metabric_time,
        metabric_event,
        risks,
        float(settings["primary_horizon_months"]),
        float(settings["secondary_horizon_months"]),
        int(settings["bootstrap_repetitions"]),
        int(settings["bootstrap_seed"]),
    )
    bootstrap_summary = summarize(
        bootstrap_rows,
        ["c_index", "auc_5y", "auc_10y"],
    )
    delta_summary = summarize(
        delta_rows,
        [
            "delta_c_index_vs_clinical",
            "delta_auc_5y_vs_clinical",
            "delta_auc_10y_vs_clinical",
        ],
    )

    pilot = read_rows(
        root / cfg["files"]["m6_track_a_paired_bootstrap"]
    )
    prefix = [
        row for row in delta_rows
        if int(row["repetition"])
        <= int(settings["pilot_prefix_repetitions"])
    ]
    keys = [
        "repetition",
        "model_set",
        "delta_c_index_vs_clinical",
        "delta_auc_5y_vs_clinical",
    ]
    pilot_lookup = {
        (
            int(float(row["repetition"])),
            row["model_set"],
        ): row
        for row in pilot
    }
    differences = []
    for row in prefix:
        key = (
            int(row["repetition"]),
            row["model_set"],
        )
        old = pilot_lookup.get(key)
        if old is None:
            differences.append(float("inf"))
            continue
        for metric in (
            "delta_c_index_vs_clinical",
            "delta_auc_5y_vs_clinical",
        ):
            differences.append(abs(
                float(row[metric]) - float(old[metric])
            ))
    maximum_prefix_difference = (
        float(np.max(differences))
        if differences else float("inf")
    )
    prefix_check = {
        "pilot_rows": len(pilot),
        "full_prefix_rows": len(prefix),
        "maximum_absolute_difference": maximum_prefix_difference,
        "pass": maximum_prefix_difference <= 1e-12,
    }

    write_csv(out / "m37_track_a_full_results.csv", results)
    write_csv(out / "m37_track_a_bootstrap_1000.csv", bootstrap_rows)
    write_csv(out / "m37_track_a_bootstrap_summary.csv", bootstrap_summary)
    write_csv(out / "m37_track_a_paired_deltas_1000.csv", delta_rows)
    write_csv(out / "m37_track_a_paired_delta_summary.csv", delta_summary)
    write_csv(out / "m37_pilot_prefix_verification.csv", [prefix_check])
    (out / "m37_track_a_model_registry.json").write_text(
        json.dumps(coefficients, indent=2),
        encoding="utf-8",
    )
    (out / "m37_track_a_alignment.json").write_text(
        json.dumps({
            "tcga_id_column": left_id,
            "outcome_id_column": right_id,
            "tcga_time_unit": time_unit,
            "tcga_n": len(tcga),
            "metabric_n": len(metabric),
            "shared_rna_features": shared_rna,
            "shared_cna_features": shared_cna,
            "clinical_preprocessing": clinical_parameters,
            "cna_preprocessing": cna_parameters,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nTrack A full results")
    print_table(
        results,
        [
            "model_set",
            "features",
            "harrell_c_index",
            "uno_c_10y",
            "ipcw_auc_5y",
            "ipcw_auc_10y",
            "brier_5y",
            "brier_10y",
            "integrated_brier_1_to_10y",
            "calibration_slope",
            "observed_minus_predicted_survival_5y",
            "observed_minus_predicted_survival_10y",
        ],
    )
    print("\nPaired 1000-bootstrap deltas")
    print_table(
        delta_summary,
        [
            "model_set",
            "metric",
            "mean",
            "sd",
            "ci_low",
            "ci_high",
            "fraction_positive",
        ],
    )
    print("\nPilot prefix verification")
    print(prefix_check)

    if not prefix_check["pass"]:
        raise RuntimeError(
            "The first 200 full-bootstrap repetitions do not reproduce "
            "the M6 pilot."
        )

    print("\nPASS: full Track A external validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
