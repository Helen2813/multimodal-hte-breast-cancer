from __future__ import annotations

import gc
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _metabric_m8_utils import (
    atomic_write_csv,
    binary_auc_at_horizon,
    build_mutation_matrix,
    fast_harrell_c_index,
    fit_penalized_cox,
    iamb_select,
    jaccard,
    load_config,
    median_impute_scale_train_test,
    out_dir,
    overlap_coefficient,
    pad_selection,
    print_table,
    project_root,
    read_continuous_sample_matrix,
    read_gene_row_matrix,
    read_rows,
    strip_feature_prefix,
    top_signed_spearman_chunked,
    write_csv,
)


def load_modality(root, cfg, modality):
    if modality == "RNA":
        return read_continuous_sample_matrix(root / cfg["files"]["rna_cleaned"], "RNA")
    if modality == "CNV":
        return read_gene_row_matrix(root / cfg["files"]["cna"], "CNA")
    if modality == "Methylation":
        return read_gene_row_matrix(root / cfg["files"]["methylation"], "METH")
    if modality == "Mutation":
        return build_mutation_matrix(root, cfg)
    raise ValueError(modality)


def prefixed_clinical(clinical, features):
    frame = clinical[features].apply(pd.to_numeric, errors="coerce")
    frame.columns = [f"CLIN__{column}" for column in frame.columns]
    return frame


def save_checkpoints(out, fold_rows, selected_rows, candidate_rows, predictions):
    atomic_write_csv(out / "m41_fold_checkpoint.csv", fold_rows)
    atomic_write_csv(out / "m41_selected_features_checkpoint.csv", selected_rows)
    atomic_write_csv(out / "m41_candidate_features_checkpoint.csv", candidate_rows)
    atomic_write_csv(out / "m41_oof_predictions_LOCAL_ONLY.csv", predictions)


def summarize_repeats(repeat_rows, modalities):
    output = []
    metrics = (
        "clinical_c_index", "modality_c_index", "clinical_modality_c_index",
        "delta_c_index_vs_clinical", "clinical_auc_5y", "modality_auc_5y",
        "clinical_modality_auc_5y", "delta_auc_5y_vs_clinical",
    )
    for modality in modalities:
        subset = [row for row in repeat_rows if row["modality"] == modality]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            output.append({
                "modality": modality,
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
                "interpretation": "algorithmic repeated-split variability; not a sampling CI",
            })
    return output


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["modality_analysis"]
    print("=" * 124)
    print("METABRIC M8.41 - MODALITY-SPECIFIC REPEATED NESTED ANALYSIS")
    print("=" * 124)
    print("Every supervised step is confined to the outer training fold.")

    clinical = pd.read_csv(root / cfg["files"]["clinical_master"], low_memory=False)
    clinical["sample_id"] = clinical["sample_id"].astype(str)
    clinical = clinical.set_index("sample_id", drop=False)
    clinical = clinical[
        pd.to_numeric(clinical["os_months"], errors="coerce").notna()
        & pd.to_numeric(clinical["os_event"], errors="coerce").notna()
    ]

    fold_rows = read_rows(out / "m41_fold_checkpoint.csv")
    selected_rows = read_rows(out / "m41_selected_features_checkpoint.csv")
    candidate_rows = read_rows(out / "m41_candidate_features_checkpoint.csv")
    prediction_rows = read_rows(out / "m41_oof_predictions_LOCAL_ONLY.csv")
    completed = {
        (row["modality"], int(float(row["repeat"])), int(float(row["fold"])))
        for row in fold_rows
    }

    modalities = ["RNA", "CNV", "Methylation", "Mutation"]
    total_folds = len(modalities) * int(settings["outer_repeats"]) * int(settings["outer_folds"])
    print(f"Already completed folds: {len(completed)}/{total_folds}")
    universe_rows = []
    cohort_rows = []

    for modality in modalities:
        print(f"\nLoading {modality} matrix...")
        matrix = load_modality(root, cfg, modality)
        matrix.index = matrix.index.astype(str)
        shared_samples = sorted(set(clinical.index) & set(matrix.index))
        modality_clinical = clinical.loc[shared_samples]
        matrix = matrix.loc[shared_samples]
        time_values = pd.to_numeric(modality_clinical["os_months"], errors="coerce").to_numpy(dtype=float)
        event_values = pd.to_numeric(modality_clinical["os_event"], errors="coerce").to_numpy(dtype=int)
        if len(shared_samples) < int(settings["minimum_complete_n"][modality]):
            raise RuntimeError(f"{modality} n={len(shared_samples)} below locked minimum")
        if int(event_values.sum()) < int(settings["minimum_events"][modality]):
            raise RuntimeError(f"{modality} events={int(event_values.sum())} below locked minimum")
        clinical_matrix = prefixed_clinical(modality_clinical, settings["clinical_features"])
        cohort_rows.append({
            "modality": modality, "n": len(shared_samples),
            "events": int(event_values.sum()), "features": matrix.shape[1],
        })
        universe_rows.extend({
            "modality": modality,
            "feature": feature,
            "gene_or_probe": strip_feature_prefix(feature),
        } for feature in matrix.columns)

        for repeat_index, seed in enumerate(
            range(int(settings["repeat_seed_start"]), int(settings["repeat_seed_start"]) + int(settings["outer_repeats"])),
            1,
        ):
            splitter = StratifiedKFold(
                n_splits=int(settings["outer_folds"]),
                shuffle=True,
                random_state=seed,
            )
            for fold_index, (train_positions, test_positions) in enumerate(splitter.split(modality_clinical, event_values), 1):
                key = (modality, repeat_index, fold_index)
                if key in completed:
                    print(f"Skipping completed {modality} repeat {repeat_index}, fold {fold_index}")
                    continue
                training_time = time_values[train_positions]
                if modality == "Mutation":
                    frequencies = matrix.iloc[train_positions].mean(axis=0)
                    eligible = frequencies[frequencies >= float(settings["mutation_frequency_threshold"])]
                    if len(eligible) > int(settings["maximum_mutation_candidates"]):
                        candidate_features, score_lookup = top_signed_spearman_chunked(
                            matrix[list(eligible.index)], train_positions, training_time,
                            int(settings["top_positive"]), int(settings["top_negative"]),
                        )
                        screening_type = "signed_spearman_after_frequency_filter"
                    else:
                        candidate_features = list(eligible.sort_values(ascending=False).index)
                        score_lookup = {feature: float(frequencies[feature]) for feature in candidate_features}
                        screening_type = "mutation_frequency"
                else:
                    candidate_features, score_lookup = top_signed_spearman_chunked(
                        matrix, train_positions, training_time,
                        int(settings["top_positive"]), int(settings["top_negative"]),
                    )
                    screening_type = "signed_spearman"
                if not candidate_features:
                    raise RuntimeError(f"No candidates for {modality}, repeat {repeat_index}, fold {fold_index}")

                candidate_rows.extend({
                    "modality": modality, "repeat": repeat_index, "seed": seed,
                    "fold": fold_index, "rank": rank, "feature": feature,
                    "screening_type": screening_type,
                    "screening_score": score_lookup.get(feature, ""),
                } for rank, feature in enumerate(candidate_features, 1))

                train_x, test_x = median_impute_scale_train_test(
                    matrix.iloc[train_positions][candidate_features],
                    matrix.iloc[test_positions][candidate_features],
                )
                selected_indices = iamb_select(
                    train_x.to_numpy(dtype=float), training_time,
                    float(settings["historical_alpha"][modality]),
                    int(settings["maximum_iamb_selected"]),
                )
                selected_features = [candidate_features[index] for index in selected_indices]
                fallback = False
                if not selected_features:
                    selected_indices = pad_selection(
                        [], train_x.to_numpy(dtype=float), training_time,
                        int(settings["fallback_features"]),
                    )
                    selected_features = [candidate_features[index] for index in selected_indices]
                    fallback = True

                clinical_train, clinical_test = median_impute_scale_train_test(
                    clinical_matrix.iloc[train_positions], clinical_matrix.iloc[test_positions]
                )
                modality_train = train_x[selected_features]
                modality_test = test_x[selected_features]
                combined_train = pd.concat([clinical_train, modality_train], axis=1)
                combined_test = pd.concat([clinical_test, modality_test], axis=1)

                risks = {}
                train_c = {}
                penalizers = {}
                for model_name, train_features, test_features in (
                    ("clinical", clinical_train, clinical_test),
                    ("modality", modality_train, modality_test),
                    ("clinical_modality", combined_train, combined_test),
                ):
                    risk, train_score, penalizer = fit_penalized_cox(
                        train_features,
                        time_values[train_positions],
                        event_values[train_positions],
                        test_features,
                        settings["cox_penalizer_sequence"],
                    )
                    risks[model_name] = risk
                    train_c[model_name] = train_score
                    penalizers[model_name] = penalizer

                metrics = {}
                for model_name, risk in risks.items():
                    c_index = fast_harrell_c_index(time_values[test_positions], event_values[test_positions], risk)
                    auc_5y, auc_n = binary_auc_at_horizon(
                        time_values[test_positions], event_values[test_positions], risk,
                        float(settings["five_year_months"]),
                    )
                    metrics[model_name] = {"c_index": c_index, "auc_5y": auc_5y, "auc_n": auc_n}

                fold_rows.append({
                    "modality": modality, "repeat": repeat_index, "seed": seed, "fold": fold_index,
                    "train_n": len(train_positions), "test_n": len(test_positions),
                    "train_events": int(event_values[train_positions].sum()),
                    "test_events": int(event_values[test_positions].sum()),
                    "candidate_features": len(candidate_features),
                    "selected_features": len(selected_features),
                    "selection_fallback": fallback,
                    "clinical_c_index": metrics["clinical"]["c_index"],
                    "modality_c_index": metrics["modality"]["c_index"],
                    "clinical_modality_c_index": metrics["clinical_modality"]["c_index"],
                    "delta_c_index_vs_clinical": metrics["clinical_modality"]["c_index"] - metrics["clinical"]["c_index"],
                    "clinical_auc_5y": metrics["clinical"]["auc_5y"],
                    "modality_auc_5y": metrics["modality"]["auc_5y"],
                    "clinical_modality_auc_5y": metrics["clinical_modality"]["auc_5y"],
                    "delta_auc_5y_vs_clinical": metrics["clinical_modality"]["auc_5y"] - metrics["clinical"]["auc_5y"],
                    "auc_5y_n": metrics["clinical_modality"]["auc_n"],
                    "clinical_train_c": train_c["clinical"],
                    "modality_train_c": train_c["modality"],
                    "clinical_modality_train_c": train_c["clinical_modality"],
                    "clinical_penalizer": penalizers["clinical"],
                    "modality_penalizer": penalizers["modality"],
                    "clinical_modality_penalizer": penalizers["clinical_modality"],
                    "engine": settings["engine"],
                    "alpha": settings["historical_alpha"][modality],
                })
                selected_rows.extend({
                    "modality": modality, "repeat": repeat_index, "seed": seed,
                    "fold": fold_index, "rank": rank, "feature": feature,
                    "gene_or_probe": strip_feature_prefix(feature),
                } for rank, feature in enumerate(selected_features, 1))

                test_samples = np.asarray(shared_samples)[test_positions]
                for local_index, sample_id in enumerate(test_samples):
                    prediction_rows.append({
                        "modality": modality, "repeat": repeat_index, "seed": seed,
                        "fold": fold_index, "sample_id": sample_id,
                        "time_months": float(time_values[test_positions][local_index]),
                        "event": int(event_values[test_positions][local_index]),
                        "clinical_risk": float(risks["clinical"][local_index]),
                        "modality_risk": float(risks["modality"][local_index]),
                        "clinical_modality_risk": float(risks["clinical_modality"][local_index]),
                    })
                save_checkpoints(out, fold_rows, selected_rows, candidate_rows, prediction_rows)
                completed.add(key)
                print(
                    f"Completed {len(completed):3d}/{total_folds}: {modality:11s} "
                    f"repeat {repeat_index:02d}, fold {fold_index}, selected={len(selected_features):3d}, "
                    f"delta C={fold_rows[-1]['delta_c_index_vs_clinical']:+.4f}, "
                    f"delta AUC5={fold_rows[-1]['delta_auc_5y_vs_clinical']:+.4f}"
                )
        del matrix
        gc.collect()

    write_csv(out / "m41_modality_cohort_summary.csv", cohort_rows)
    write_csv(out / "m41_modality_feature_universe.csv", universe_rows)

    # Repeat-level pooled out-of-fold metrics.
    prediction_rows = read_rows(out / "m41_oof_predictions_LOCAL_ONLY.csv")
    repeat_rows = []
    for modality in modalities:
        for repeat_index in range(1, int(settings["outer_repeats"]) + 1):
            subset = [
                row for row in prediction_rows
                if row["modality"] == modality and int(float(row["repeat"])) == repeat_index
            ]
            time_values = np.asarray([float(row["time_months"]) for row in subset])
            event_values = np.asarray([int(float(row["event"])) for row in subset])
            risks = {
                "clinical": np.asarray([float(row["clinical_risk"]) for row in subset]),
                "modality": np.asarray([float(row["modality_risk"]) for row in subset]),
                "clinical_modality": np.asarray([float(row["clinical_modality_risk"]) for row in subset]),
            }
            metrics = {}
            for model_name, risk in risks.items():
                metrics[model_name] = {
                    "c": fast_harrell_c_index(time_values, event_values, risk),
                    "auc": binary_auc_at_horizon(time_values, event_values, risk, float(settings["five_year_months"]))[0],
                }
            repeat_rows.append({
                "modality": modality, "repeat": repeat_index, "n": len(subset),
                "events": int(event_values.sum()),
                "clinical_c_index": metrics["clinical"]["c"],
                "modality_c_index": metrics["modality"]["c"],
                "clinical_modality_c_index": metrics["clinical_modality"]["c"],
                "delta_c_index_vs_clinical": metrics["clinical_modality"]["c"] - metrics["clinical"]["c"],
                "clinical_auc_5y": metrics["clinical"]["auc"],
                "modality_auc_5y": metrics["modality"]["auc"],
                "clinical_modality_auc_5y": metrics["clinical_modality"]["auc"],
                "delta_auc_5y_vs_clinical": metrics["clinical_modality"]["auc"] - metrics["clinical"]["auc"],
            })
    write_csv(out / "m41_repeat_level_results.csv", repeat_rows)
    repeat_summary = summarize_repeats(repeat_rows, modalities)
    write_csv(out / "m41_repeat_level_summary.csv", repeat_summary)

    # Stability and core sets.
    selected_rows = read_rows(out / "m41_selected_features_checkpoint.csv")
    total_modality_folds = int(settings["outer_repeats"]) * int(settings["outer_folds"])
    frequency_rows = []
    stability_rows = []
    for modality in modalities:
        rows = [row for row in selected_rows if row["modality"] == modality]
        counts = {}
        repeat_counts = {}
        for row in rows:
            feature = row["feature"]
            counts[feature] = counts.get(feature, 0) + 1
            key = (int(float(row["repeat"])), feature)
            repeat_counts[key] = repeat_counts.get(key, 0) + 1
        for feature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            stable_repeats = sum(
                repeat_counts.get((repeat_index, feature), 0) >= int(settings["stable_folds_within_repeat"])
                for repeat_index in range(1, int(settings["outer_repeats"]) + 1)
            )
            frequency_rows.append({
                "modality": modality, "feature": feature,
                "gene_or_probe": strip_feature_prefix(feature),
                "selected_folds": count,
                "selection_frequency": count / total_modality_folds,
                "selected_repeats_at_least_once": sum(
                    repeat_counts.get((repeat_index, feature), 0) > 0
                    for repeat_index in range(1, int(settings["outer_repeats"]) + 1)
                ),
                "stable_repeats_3_of_5": stable_repeats,
                "core_by_frequency": count / total_modality_folds >= float(settings["core_selection_frequency"]),
                "recurrent_by_repeat": stable_repeats >= int(settings["minimum_stable_repeats"]),
            })
        for repeat_index in range(1, int(settings["outer_repeats"]) + 1):
            fold_sets = {
                fold_index: {
                    row["feature"] for row in rows
                    if int(float(row["repeat"])) == repeat_index and int(float(row["fold"])) == fold_index
                }
                for fold_index in range(1, int(settings["outer_folds"]) + 1)
            }
            for first, second in combinations(range(1, int(settings["outer_folds"]) + 1), 2):
                stability_rows.append({
                    "modality": modality, "repeat": repeat_index,
                    "fold_a": first, "fold_b": second,
                    "jaccard": jaccard(fold_sets[first], fold_sets[second]),
                    "overlap_coefficient": overlap_coefficient(fold_sets[first], fold_sets[second]),
                })
    write_csv(out / "m41_feature_selection_frequency.csv", frequency_rows)
    write_csv(out / "m41_within_repeat_stability.csv", stability_rows)

    modality_summary = []
    for modality in modalities:
        delta_c = next(row for row in repeat_summary if row["modality"] == modality and row["metric"] == "delta_c_index_vs_clinical")
        delta_auc = next(row for row in repeat_summary if row["modality"] == modality and row["metric"] == "delta_auc_5y_vs_clinical")
        frequency = [row for row in frequency_rows if row["modality"] == modality]
        stability = [float(row["jaccard"]) for row in stability_rows if row["modality"] == modality]
        modality_summary.append({
            "modality": modality,
            "mean_delta_c_index": delta_c["mean"],
            "sd_delta_c_index": delta_c["sd"],
            "fraction_repeats_delta_c_positive": delta_c["fraction_positive"],
            "mean_delta_auc_5y": delta_auc["mean"],
            "sd_delta_auc_5y": delta_auc["sd"],
            "fraction_repeats_delta_auc_positive": delta_auc["fraction_positive"],
            "mean_within_repeat_jaccard": float(np.mean(stability)),
            "core_features_frequency_ge_0_5": sum(str(row["core_by_frequency"]).lower() == "true" for row in frequency),
            "recurrent_features": sum(str(row["recurrent_by_repeat"]).lower() == "true" for row in frequency),
        })
    write_csv(out / "m41_modality_summary.csv", modality_summary)
    print("\nModality-specific repeated results")
    print_table(modality_summary, [
        "modality", "mean_delta_c_index", "sd_delta_c_index",
        "fraction_repeats_delta_c_positive", "mean_delta_auc_5y",
        "sd_delta_auc_5y", "fraction_repeats_delta_auc_positive",
        "mean_within_repeat_jaccard", "core_features_frequency_ge_0_5",
        "recurrent_features",
    ])
    print("\nPASS: all four modality-specific nested analyses completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
