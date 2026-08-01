from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _metabric_m7_utils import (
    atomic_write_csv,
    binary_auc_at_horizon,
    fast_harrell_c_index,
    iamb_select,
    jaccard,
    load_config,
    median_impute_scale_train_test,
    out_dir,
    overlap_coefficient,
    pad_selection,
    print_table,
    project_root,
    read_rows,
    top_signed_spearman_chunked,
    transform_for_ci,
    write_csv,
)


NONSYNONYMOUS_CLASSES = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Splice_Region",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonstop_Mutation",
    "Translation_Start_Site",
}


def load_rna(path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    dtype = {
        column: "float32"
        for column in header[1:]
    }
    frame = pd.read_csv(
        path,
        dtype=dtype,
        low_memory=False,
    )
    id_column = frame.columns[0]
    frame = frame.rename(columns={id_column: "sample_id"})
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.set_index("sample_id")
    frame.columns = [
        f"RNA__{column}" for column in frame.columns
    ]
    return frame


def load_cna(path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(
        path,
        sep="\t",
        comment="#",
        nrows=0,
    ).columns)
    sample_columns = [
        column for column in header
        if column not in {
            "Hugo_Symbol",
            "Entrez_Gene_Id",
        }
    ]
    dtype = {
        column: "float32"
        for column in sample_columns
    }
    dtype["Hugo_Symbol"] = "string"
    dtype["Entrez_Gene_Id"] = "string"
    raw = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=dtype,
        low_memory=False,
    )
    values = raw[sample_columns].astype("float32")
    values.insert(
        0,
        "Hugo_Symbol",
        raw["Hugo_Symbol"].astype(str).str.upper(),
    )
    gene_level = (
        values
        .groupby("Hugo_Symbol", sort=True)
        .median(numeric_only=True)
    )
    matrix = gene_level.T
    matrix.index = matrix.index.astype(str)
    matrix.index.name = "sample_id"
    matrix.columns = [
        f"CNA__{column}" for column in matrix.columns
    ]
    return matrix


def build_mutation_matrix(root: Path, cfg: dict) -> pd.DataFrame:
    genes = pd.read_csv(
        root / cfg["files"]["metabric_173_genes"],
        dtype=str,
    )["gene"].dropna().astype(str).str.upper()
    genes = sorted(set(genes))

    assignments = pd.read_csv(
        root / cfg["files"]["gene_panel_matrix"],
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
    )
    assigned = sorted(set(
        assignments.loc[
            assignments["mutations"].astype(str)
            == "METABRIC_173",
            "SAMPLE_ID",
        ].dropna().astype(str)
    ))

    calls = pd.read_csv(
        root / cfg["files"]["mutations"],
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
    )
    calls = calls[
        calls["Variant_Classification"].isin(
            NONSYNONYMOUS_CLASSES
        )
    ].copy()
    calls["Hugo_Symbol"] = (
        calls["Hugo_Symbol"].astype(str).str.upper()
    )
    calls = calls[
        calls["Hugo_Symbol"].isin(genes)
        & calls["Tumor_Sample_Barcode"].astype(str).isin(assigned)
    ]

    matrix = pd.DataFrame(
        0.0,
        index=pd.Index(assigned, name="sample_id"),
        columns=[f"MUT__{gene}" for gene in genes],
        dtype=np.float32,
    )
    for sample, group in calls.groupby(
        "Tumor_Sample_Barcode"
    ):
        columns = [
            f"MUT__{gene}"
            for gene in sorted(set(group["Hugo_Symbol"]))
            if gene in genes
        ]
        if columns:
            matrix.loc[str(sample), columns] = 1.0
    return matrix


def prefixed_clinical(
    clinical: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    result = clinical[features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    result.columns = [
        f"CLIN__{column}" for column in result.columns
    ]
    return result


def fit_cox(
    train_x: pd.DataFrame,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_x: pd.DataFrame,
    penalizer: float,
):
    from lifelines import CoxPHFitter
    frame = train_x.copy()
    frame["__time"] = train_time
    frame["__event"] = train_event
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(
        frame,
        duration_col="__time",
        event_col="__event",
        show_progress=False,
    )
    risk = (
        model.predict_partial_hazard(test_x)
        .to_numpy(dtype=float)
        .ravel()
    )
    return risk, float(model.concordance_index_)


def save_checkpoints(
    out: Path,
    fold_rows: list[dict],
    selected_rows: list[dict],
    candidate_rows: list[dict],
    prediction_rows: list[dict],
) -> None:
    atomic_write_csv(
        out / "m38_fold_checkpoint.csv",
        fold_rows,
    )
    atomic_write_csv(
        out / "m38_selected_features_checkpoint.csv",
        selected_rows,
    )
    atomic_write_csv(
        out / "m38_candidate_features_checkpoint.csv",
        candidate_rows,
    )
    atomic_write_csv(
        out / "m38_oof_predictions_LOCAL_ONLY.csv",
        prediction_rows,
    )


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["track_b"]

    print("=" * 124)
    print("METABRIC M7.38 - FULL 20x5 RECONSTRUCTED NESTED TRACK B")
    print("=" * 124)
    print(
        "This is explicitly a reconstructed dependency-aware replication, "
        "not a bitwise reproduction of the historical IAMB implementation."
    )

    clinical = pd.read_csv(
        root / cfg["files"]["clinical_master"],
        low_memory=False,
    )
    clinical["sample_id"] = clinical["sample_id"].astype(str)
    clinical = clinical.set_index("sample_id", drop=False)
    clinical = clinical[
        pd.to_numeric(clinical["os_months"], errors="coerce").notna()
        & pd.to_numeric(clinical["os_event"], errors="coerce").notna()
    ]

    print("Loading RNA matrix...")
    rna = load_rna(root / cfg["files"]["rna_cleaned"])
    print(f"RNA matrix: {rna.shape[0]} samples x {rna.shape[1]} features")
    print("Loading CNA matrix...")
    cna = load_cna(root / cfg["files"]["cna"])
    print(f"CNA matrix: {cna.shape[0]} samples x {cna.shape[1]} features")
    print("Building panel-aware mutation matrix...")
    mutation = build_mutation_matrix(root, cfg)
    print(
        f"Mutation matrix: {mutation.shape[0]} samples x "
        f"{mutation.shape[1]} panel genes"
    )

    shared_samples = sorted(
        set(clinical.index)
        & set(rna.index)
        & set(cna.index)
        & set(mutation.index)
    )
    if len(shared_samples) < int(settings["minimum_complete_n"]):
        raise RuntimeError("Track B complete cohort below locked minimum.")

    clinical = clinical.loc[shared_samples]
    rna = rna.loc[shared_samples]
    cna = cna.loc[shared_samples]
    mutation = mutation.loc[shared_samples]
    clinical_matrix = prefixed_clinical(
        clinical,
        settings["clinical_features"],
    )
    time = pd.to_numeric(
        clinical["os_months"],
        errors="coerce",
    ).to_numpy(dtype=float)
    event = pd.to_numeric(
        clinical["os_event"],
        errors="coerce",
    ).to_numpy(dtype=int)
    if int(event.sum()) < int(settings["minimum_events"]):
        raise RuntimeError("Track B event count below locked minimum.")

    fold_path = out / "m38_fold_checkpoint.csv"
    selected_path = out / "m38_selected_features_checkpoint.csv"
    candidate_path = out / "m38_candidate_features_checkpoint.csv"
    prediction_path = out / "m38_oof_predictions_LOCAL_ONLY.csv"

    fold_rows = read_rows(fold_path)
    selected_rows = read_rows(selected_path)
    candidate_rows = read_rows(candidate_path)
    prediction_rows = read_rows(prediction_path)
    completed = {
        (
            int(float(row["repeat"])),
            int(float(row["fold"])),
        )
        for row in fold_rows
    }

    seeds = list(range(
        int(settings["repeat_seed_start"]),
        int(settings["repeat_seed_start"])
        + int(settings["outer_repeats"]),
    ))
    total_folds = (
        int(settings["outer_repeats"])
        * int(settings["outer_folds"])
    )
    print(
        f"Already completed folds: {len(completed)}/{total_folds}"
    )

    for repeat_index, seed in enumerate(seeds, 1):
        splitter = StratifiedKFold(
            n_splits=int(settings["outer_folds"]),
            shuffle=True,
            random_state=seed,
        )
        for fold_index, (train_positions, test_positions) in enumerate(
            splitter.split(clinical, event),
            1,
        ):
            key = (repeat_index, fold_index)
            if key in completed:
                print(
                    f"Skipping completed repeat {repeat_index}, "
                    f"fold {fold_index}"
                )
                continue

            train_time = time[train_positions]
            selected_rna, rna_scores = top_signed_spearman_chunked(
                rna,
                train_positions,
                train_time,
                int(settings["rna_top_positive"]),
                int(settings["rna_top_negative"]),
            )
            selected_cna, cna_scores = top_signed_spearman_chunked(
                cna,
                train_positions,
                train_time,
                int(settings["cna_top_positive"]),
                int(settings["cna_top_negative"]),
            )
            mutation_train = mutation.iloc[train_positions]
            mutation_frequency = mutation_train.mean(axis=0)
            selected_mutation = list(
                mutation_frequency[
                    mutation_frequency
                    >= float(settings["mutation_frequency_threshold"])
                ]
                .sort_values(ascending=False)
                .index[: int(settings["maximum_mutation_candidates"])]
            )

            rho_lookup = {
                ("RNA", row["feature"]): row["spearman_rho"]
                for row in rna_scores
            }
            rho_lookup.update({
                ("CNV", row["feature"]): row["spearman_rho"]
                for row in cna_scores
            })
            for modality, features in (
                ("RNA", selected_rna),
                ("CNV", selected_cna),
                ("Mutation", selected_mutation),
            ):
                for rank, feature in enumerate(features, 1):
                    candidate_rows.append({
                        "repeat": repeat_index,
                        "seed": seed,
                        "fold": fold_index,
                        "modality": modality,
                        "rank": rank,
                        "feature": feature,
                        "screening_score": (
                            rho_lookup.get((modality, feature), "")
                            if modality != "Mutation"
                            else float(mutation_frequency[feature])
                        ),
                    })

            candidates = (
                list(clinical_matrix.columns)
                + selected_rna
                + selected_cna
                + selected_mutation
            )
            candidates = list(dict.fromkeys(candidates))

            train_raw = pd.concat([
                clinical_matrix.iloc[train_positions],
                rna.iloc[train_positions][selected_rna],
                cna.iloc[train_positions][selected_cna],
                mutation.iloc[train_positions][selected_mutation],
            ], axis=1)
            test_raw = pd.concat([
                clinical_matrix.iloc[test_positions],
                rna.iloc[test_positions][selected_rna],
                cna.iloc[test_positions][selected_cna],
                mutation.iloc[test_positions][selected_mutation],
            ], axis=1)
            train_x, test_x, _ = median_impute_scale_train_test(
                train_raw,
                test_raw,
                scale=True,
            )
            x_ci, y_ci = transform_for_ci(
                train_x.to_numpy(dtype=float),
                train_time,
                settings["engine"],
            )
            selected_indices = iamb_select(
                x_ci,
                y_ci,
                float(settings["iamb_alpha"]),
                int(settings["maximum_iamb_selected"]),
            )
            selected_features = [
                candidates[index]
                for index in selected_indices
            ]
            fallback = False
            if not selected_features:
                selected_indices = pad_selection(
                    [],
                    x_ci,
                    y_ci,
                    int(settings["fallback_features"]),
                )
                selected_features = [
                    candidates[index]
                    for index in selected_indices
                ]
                fallback = True

            model_risk, train_c = fit_cox(
                train_x[selected_features],
                time[train_positions],
                event[train_positions],
                test_x[selected_features],
                float(settings["cox_penalizer"]),
            )
            clinical_train, clinical_test, _ = (
                median_impute_scale_train_test(
                    clinical_matrix.iloc[train_positions],
                    clinical_matrix.iloc[test_positions],
                    scale=True,
                )
            )
            clinical_risk, clinical_train_c = fit_cox(
                clinical_train,
                time[train_positions],
                event[train_positions],
                clinical_test,
                float(settings["cox_penalizer"]),
            )

            model_c = fast_harrell_c_index(
                time[test_positions],
                event[test_positions],
                model_risk,
            )
            clinical_c = fast_harrell_c_index(
                time[test_positions],
                event[test_positions],
                clinical_risk,
            )
            model_auc, model_auc_n = binary_auc_at_horizon(
                time[test_positions],
                event[test_positions],
                model_risk,
                float(settings["five_year_months"]),
            )
            clinical_auc, _ = binary_auc_at_horizon(
                time[test_positions],
                event[test_positions],
                clinical_risk,
                float(settings["five_year_months"]),
            )

            fold_row = {
                "repeat": repeat_index,
                "seed": seed,
                "fold": fold_index,
                "train_n": len(train_positions),
                "test_n": len(test_positions),
                "train_events": int(event[train_positions].sum()),
                "test_events": int(event[test_positions].sum()),
                "combined_candidates": len(candidates),
                "rna_candidates": len(selected_rna),
                "cna_candidates": len(selected_cna),
                "mutation_candidates": len(selected_mutation),
                "clinical_candidates": len(clinical_matrix.columns),
                "selected_features": len(selected_features),
                "selected_clinical": sum(
                    feature.startswith("CLIN__")
                    for feature in selected_features
                ),
                "selected_rna": sum(
                    feature.startswith("RNA__")
                    for feature in selected_features
                ),
                "selected_cna": sum(
                    feature.startswith("CNA__")
                    for feature in selected_features
                ),
                "selected_mutation": sum(
                    feature.startswith("MUT__")
                    for feature in selected_features
                ),
                "selection_fallback": fallback,
                "train_c_index": train_c,
                "clinical_train_c_index": clinical_train_c,
                "harrell_c_index": model_c,
                "clinical_only_c_index": clinical_c,
                "delta_c_index_vs_clinical": model_c - clinical_c,
                "auc_5y": model_auc,
                "clinical_only_auc_5y": clinical_auc,
                "delta_auc_5y_vs_clinical": model_auc - clinical_auc,
                "auc_5y_n": model_auc_n,
                "engine": settings["engine"],
                "historical_engine_status": settings[
                    "historical_engine_status"
                ],
            }
            fold_rows.append(fold_row)

            for rank, feature in enumerate(selected_features, 1):
                if feature.startswith("CLIN__"):
                    modality = "Clinical"
                elif feature.startswith("RNA__"):
                    modality = "RNA"
                elif feature.startswith("CNA__"):
                    modality = "CNV"
                elif feature.startswith("MUT__"):
                    modality = "Mutation"
                else:
                    modality = "Unknown"
                selected_rows.append({
                    "repeat": repeat_index,
                    "seed": seed,
                    "fold": fold_index,
                    "rank": rank,
                    "feature": feature,
                    "modality": modality,
                })

            test_sample_ids = np.asarray(shared_samples)[test_positions]
            for local_index, sample_id in enumerate(test_sample_ids):
                prediction_rows.append({
                    "repeat": repeat_index,
                    "seed": seed,
                    "fold": fold_index,
                    "sample_id": sample_id,
                    "time_months": float(time[test_positions][local_index]),
                    "event": int(event[test_positions][local_index]),
                    "model_risk": float(model_risk[local_index]),
                    "clinical_risk": float(clinical_risk[local_index]),
                })

            save_checkpoints = (
                fold_rows,
                selected_rows,
                candidate_rows,
                prediction_rows,
            )
            atomic_write_csv(fold_path, save_checkpoints[0])
            atomic_write_csv(selected_path, save_checkpoints[1])
            atomic_write_csv(candidate_path, save_checkpoints[2])
            atomic_write_csv(prediction_path, save_checkpoints[3])
            completed.add(key)

            print(
                f"Completed {len(completed):3d}/{total_folds}: "
                f"repeat {repeat_index:02d}, fold {fold_index}, "
                f"selected={len(selected_features):3d}, "
                f"C={model_c:.4f} vs {clinical_c:.4f}, "
                f"AUC5={model_auc:.4f} vs {clinical_auc:.4f}"
            )

    if len(completed) != total_folds:
        raise RuntimeError(
            f"Only {len(completed)}/{total_folds} folds completed."
        )

    fold_rows = read_rows(fold_path)
    selected_rows = read_rows(selected_path)
    prediction_rows = read_rows(prediction_path)

    repeat_rows = []
    for repeat_index in range(1, int(settings["outer_repeats"]) + 1):
        subset = [
            row for row in prediction_rows
            if int(float(row["repeat"])) == repeat_index
        ]
        time_repeat = np.asarray(
            [float(row["time_months"]) for row in subset],
            dtype=float,
        )
        event_repeat = np.asarray(
            [int(float(row["event"])) for row in subset],
            dtype=int,
        )
        model_risk = np.asarray(
            [float(row["model_risk"]) for row in subset],
            dtype=float,
        )
        clinical_risk = np.asarray(
            [float(row["clinical_risk"]) for row in subset],
            dtype=float,
        )
        model_c = fast_harrell_c_index(
            time_repeat,
            event_repeat,
            model_risk,
        )
        clinical_c = fast_harrell_c_index(
            time_repeat,
            event_repeat,
            clinical_risk,
        )
        model_auc, _ = binary_auc_at_horizon(
            time_repeat,
            event_repeat,
            model_risk,
            float(settings["five_year_months"]),
        )
        clinical_auc, _ = binary_auc_at_horizon(
            time_repeat,
            event_repeat,
            clinical_risk,
            float(settings["five_year_months"]),
        )
        repeat_rows.append({
            "repeat": repeat_index,
            "seed": int(settings["repeat_seed_start"]) + repeat_index - 1,
            "n": len(subset),
            "events": int(event_repeat.sum()),
            "harrell_c_index": model_c,
            "clinical_only_c_index": clinical_c,
            "delta_c_index_vs_clinical": model_c - clinical_c,
            "auc_5y": model_auc,
            "clinical_only_auc_5y": clinical_auc,
            "delta_auc_5y_vs_clinical": model_auc - clinical_auc,
        })
    write_csv(out / "m38_repeat_level_oof_results.csv", repeat_rows)

    repeat_summary = []
    for metric in (
        "harrell_c_index",
        "clinical_only_c_index",
        "delta_c_index_vs_clinical",
        "auc_5y",
        "clinical_only_auc_5y",
        "delta_auc_5y_vs_clinical",
    ):
        values = np.asarray(
            [float(row[metric]) for row in repeat_rows],
            dtype=float,
        )
        repeat_summary.append({
            "metric": metric,
            "repeats": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "fraction_positive": float(np.mean(values > 0)),
            "interpretation": (
                "algorithmic repeated-split variability; not a sampling CI"
            ),
        })
    write_csv(out / "m38_repeat_level_summary.csv", repeat_summary)

    selection_frequency = {}
    modality_lookup = {}
    repeat_feature_counts = {}
    for row in selected_rows:
        feature = row["feature"]
        selection_frequency[feature] = (
            selection_frequency.get(feature, 0) + 1
        )
        modality_lookup[feature] = row["modality"]
        key = (
            int(float(row["repeat"])),
            feature,
        )
        repeat_feature_counts[key] = (
            repeat_feature_counts.get(key, 0) + 1
        )
    frequency_rows = [
        {
            "feature": feature,
            "modality": modality_lookup[feature],
            "selected_folds": count,
            "selection_frequency_100_folds": (
                count / total_folds
            ),
            "selected_repeats_at_least_once": sum(
                repeat_feature_counts.get((repeat_index, feature), 0) > 0
                for repeat_index in range(
                    1,
                    int(settings["outer_repeats"]) + 1,
                )
            ),
            "stable_repeats_3_of_5_folds": sum(
                repeat_feature_counts.get((repeat_index, feature), 0)
                >= int(settings["stable_within_repeat_folds"])
                for repeat_index in range(
                    1,
                    int(settings["outer_repeats"]) + 1,
                )
            ),
        }
        for feature, count in sorted(
            selection_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    write_csv(out / "m38_selection_frequency.csv", frequency_rows)

    within_stability = []
    repeat_stable_sets = {}
    for repeat_index in range(1, int(settings["outer_repeats"]) + 1):
        fold_sets = {}
        for fold_index in range(1, int(settings["outer_folds"]) + 1):
            fold_sets[fold_index] = {
                row["feature"]
                for row in selected_rows
                if int(float(row["repeat"])) == repeat_index
                and int(float(row["fold"])) == fold_index
            }
        for first, second in combinations(
            range(1, int(settings["outer_folds"]) + 1),
            2,
        ):
            within_stability.append({
                "repeat": repeat_index,
                "fold_a": first,
                "fold_b": second,
                "jaccard": jaccard(
                    fold_sets[first],
                    fold_sets[second],
                ),
                "overlap_coefficient": overlap_coefficient(
                    fold_sets[first],
                    fold_sets[second],
                ),
                "intersection": len(
                    fold_sets[first] & fold_sets[second]
                ),
                "union": len(
                    fold_sets[first] | fold_sets[second]
                ),
            })
        repeat_stable_sets[repeat_index] = {
            feature
            for (repeat, feature), count in repeat_feature_counts.items()
            if repeat == repeat_index
            and count >= int(settings["stable_within_repeat_folds"])
        }
    write_csv(out / "m38_within_repeat_stability.csv", within_stability)

    between_stability = []
    for first, second in combinations(
        range(1, int(settings["outer_repeats"]) + 1),
        2,
    ):
        between_stability.append({
            "repeat_a": first,
            "repeat_b": second,
            "stable_set_a": len(repeat_stable_sets[first]),
            "stable_set_b": len(repeat_stable_sets[second]),
            "jaccard": jaccard(
                repeat_stable_sets[first],
                repeat_stable_sets[second],
            ),
            "overlap_coefficient": overlap_coefficient(
                repeat_stable_sets[first],
                repeat_stable_sets[second],
            ),
        })
    write_csv(out / "m38_between_repeat_stability.csv", between_stability)

    composition_rows = []
    for modality in ("Clinical", "RNA", "CNV", "Mutation"):
        values = [
            sum(
                row["modality"] == modality
                and int(float(row["repeat"])) == repeat_index
                and int(float(row["fold"])) == fold_index
                for row in selected_rows
            )
            for repeat_index in range(
                1,
                int(settings["outer_repeats"]) + 1,
            )
            for fold_index in range(
                1,
                int(settings["outer_folds"]) + 1,
            )
        ]
        composition_rows.append({
            "modality": modality,
            "folds": len(values),
            "mean_selected": float(np.mean(values)),
            "sd_selected": float(np.std(values, ddof=1)),
            "median_selected": float(np.median(values)),
            "minimum_selected": int(np.min(values)),
            "maximum_selected": int(np.max(values)),
        })
    write_csv(out / "m38_modality_composition.csv", composition_rows)

    summary = {
        "status": "FULL_RECONSTRUCTED_TRACK_B_COMPLETE",
        "complete_case_n": len(shared_samples),
        "events": int(event.sum()),
        "outer_repeats": int(settings["outer_repeats"]),
        "outer_folds": int(settings["outer_folds"]),
        "completed_fold_fits": len(completed),
        "engine": settings["engine"],
        "historical_engine_status": settings[
            "historical_engine_status"
        ],
        "mean_repeat_delta_c_index": float(np.mean([
            row["delta_c_index_vs_clinical"]
            for row in repeat_rows
        ])),
        "sd_repeat_delta_c_index": float(np.std([
            row["delta_c_index_vs_clinical"]
            for row in repeat_rows
        ], ddof=1)),
        "mean_repeat_delta_auc_5y": float(np.mean([
            row["delta_auc_5y_vs_clinical"]
            for row in repeat_rows
        ])),
        "sd_repeat_delta_auc_5y": float(np.std([
            row["delta_auc_5y_vs_clinical"]
            for row in repeat_rows
        ], ddof=1)),
        "mean_within_repeat_jaccard": float(np.mean([
            row["jaccard"] for row in within_stability
        ])),
        "mean_between_repeat_stable_set_jaccard": float(np.mean([
            row["jaccard"] for row in between_stability
        ])),
        "stable_features_all_20_repeats_3_of_5": [
            row["feature"]
            for row in frequency_rows
            if int(row["stable_repeats_3_of_5_folds"])
            == int(settings["outer_repeats"])
        ],
        "interpretation_boundary": (
            "Repeated split distributions quantify algorithmic variability "
            "and are not confidence intervals for population performance."
        ),
    }
    (out / "m38_track_b_full_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nRepeat-level OOF results")
    print_table(
        repeat_rows,
        [
            "repeat",
            "harrell_c_index",
            "clinical_only_c_index",
            "delta_c_index_vs_clinical",
            "auc_5y",
            "clinical_only_auc_5y",
            "delta_auc_5y_vs_clinical",
        ],
    )
    print("\nRepeat-level algorithmic variability")
    print_table(
        repeat_summary,
        [
            "metric",
            "mean",
            "sd",
            "median",
            "q025",
            "q975",
            "minimum",
            "maximum",
            "fraction_positive",
        ],
    )
    print("\nModality composition")
    print_table(
        composition_rows,
        [
            "modality",
            "mean_selected",
            "sd_selected",
            "median_selected",
            "minimum_selected",
            "maximum_selected",
        ],
    )
    print("\nMost frequently selected features")
    print_table(
        frequency_rows,
        [
            "feature",
            "modality",
            "selected_folds",
            "selection_frequency_100_folds",
            "selected_repeats_at_least_once",
            "stable_repeats_3_of_5_folds",
        ],
        max_rows=150,
    )
    print("\nFull Track B summary")
    print(json.dumps(summary, indent=2))

    print("\nPASS: full 20x5 reconstructed Track B completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
