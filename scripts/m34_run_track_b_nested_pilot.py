from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _metabric_m6_resume_utils import (
    fast_harrell_c_index,
    read_float32_csv,
    top_signed_spearman_chunked,
)
from _metabric_m6_utils import (
    binary_auc_at_horizon,
    fit_cox_risk,
    iamb_select,
    jaccard,
    load_config,
    median_impute_scale_train_test,
    out_dir,
    overlap_coefficient,
    pad_selection,
    print_table,
    project_root,
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


def load_cna_float32(path) -> pd.DataFrame:
    header = list(
        pd.read_csv(
            path,
            sep="\t",
            comment="#",
            nrows=0,
        ).columns
    )
    sample_columns = [
        column for column in header
        if column not in {"Hugo_Symbol", "Entrez_Gene_Id"}
    ]
    dtype = {column: "float32" for column in sample_columns}
    dtype["Hugo_Symbol"] = "string"
    dtype["Entrez_Gene_Id"] = "string"
    raw = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=dtype,
        low_memory=False,
    )
    symbols = raw["Hugo_Symbol"].astype(str).str.upper()
    values = raw[sample_columns].astype("float32")
    values.insert(0, "Hugo_Symbol", symbols)
    gene_level = (
        values
        .groupby("Hugo_Symbol", sort=True)
        .median(numeric_only=True)
    )
    matrix = gene_level.T
    matrix.index = matrix.index.astype(str)
    matrix.index.name = "sample_id"
    return matrix


def build_panel_aware_mutation_matrix(root, cfg) -> pd.DataFrame:
    panel_genes = pd.read_csv(
        root
        / cfg["metabric_m5_dir"]
        / "m27_metabric_173_gene_list.csv",
        dtype=str,
    )["gene"].dropna().astype(str).str.upper()
    panel_genes = sorted(set(panel_genes))

    panel_matrix = pd.read_csv(
        root
        / cfg["raw_dir"]
        / "data_gene_panel_matrix.txt",
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
    )
    assigned = sorted(set(
        panel_matrix.loc[
            panel_matrix["mutations"].astype(str) == "METABRIC_173",
            "SAMPLE_ID",
        ].dropna().astype(str)
    ))

    mutations = pd.read_csv(
        root
        / cfg["raw_dir"]
        / "data_mutations.txt",
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
    )
    mutations = mutations[
        mutations["Variant_Classification"].isin(NONSYNONYMOUS_CLASSES)
    ].copy()
    mutations["Hugo_Symbol"] = (
        mutations["Hugo_Symbol"].astype(str).str.upper()
    )
    mutations = mutations[
        mutations["Hugo_Symbol"].isin(panel_genes)
        & mutations["Tumor_Sample_Barcode"].astype(str).isin(assigned)
    ]

    matrix = pd.DataFrame(
        0,
        index=pd.Index(assigned, name="sample_id"),
        columns=[f"MUT__{gene}" for gene in panel_genes],
        dtype=np.float32,
    )
    for sample, group in mutations.groupby("Tumor_Sample_Barcode"):
        genes = sorted(set(group["Hugo_Symbol"].astype(str)))
        columns = [f"MUT__{gene}" for gene in genes if gene in panel_genes]
        if columns:
            matrix.loc[str(sample), columns] = 1.0
    return matrix


def prefixed_clinical(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame[features].apply(pd.to_numeric, errors="coerce").copy()
    result.columns = [f"CLIN__{column}" for column in result.columns]
    return result


def summarize_selection_stability(feature_sets: dict[int, set[str]]) -> list[dict]:
    rows = []
    for first, second in combinations(sorted(feature_sets), 2):
        rows.append({
            "fold_a": first,
            "fold_b": second,
            "jaccard": jaccard(feature_sets[first], feature_sets[second]),
            "overlap_coefficient": overlap_coefficient(
                feature_sets[first],
                feature_sets[second],
            ),
            "intersection": len(feature_sets[first] & feature_sets[second]),
            "union": len(feature_sets[first] | feature_sets[second]),
        })
    return rows


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["track_b_pilot"]

    print("=" * 124)
    print("METABRIC M6.34R - MEMORY-SAFE NESTED PAPER-1 REPLICATION PILOT")
    print("=" * 124)

    engine_decision = json.loads(
        (out / "m32_engine_decision.json").read_text(encoding="utf-8")
    )
    engine = engine_decision["selected_engine"]
    engine_reproduced = bool(
        engine_decision["historical_engine_reproduced"]
    )
    print(f"Selected CI engine: {engine}")
    print(f"Historical engine reproduced: {engine_reproduced}")
    print(
        "Track B is labelled reconstructed because the historical TCGA "
        "feature sets were not reproduced at the locked threshold."
    )
    print(
        "Mutation candidates use all 173 panel genes with panel-aware zeros "
        "and nonsynonymous calls only."
    )

    clinical = pd.read_csv(
        root
        / cfg["metabric_m2_dir"]
        / "m06_metabric_clinical_master_LOCAL_ONLY.csv",
        low_memory=False,
    )
    clinical["sample_id"] = clinical["sample_id"].astype(str)
    clinical = clinical.set_index("sample_id", drop=False)

    rna = read_float32_csv(
        root / cfg["raw_dir"] / cfg["files"]["metabric_rna_cleaned"]
    )
    rna_id_column = rna.columns[0]
    rna = rna.rename(columns={rna_id_column: "sample_id"})
    rna["sample_id"] = rna["sample_id"].astype(str)
    rna = rna.set_index("sample_id", drop=True)
    rna.columns = [f"RNA__{column}" for column in rna.columns]

    cna = load_cna_float32(
        root / cfg["raw_dir"] / cfg["files"]["metabric_cna"]
    )
    cna.columns = [f"CNA__{column}" for column in cna.columns]

    mutation = build_panel_aware_mutation_matrix(root, cfg)

    clinical_valid = clinical[
        pd.to_numeric(clinical["os_months"], errors="coerce").notna()
        & pd.to_numeric(clinical["os_event"], errors="coerce").notna()
    ]
    shared_samples = sorted(
        set(clinical_valid.index)
        & set(rna.index)
        & set(cna.index)
        & set(mutation.index)
    )
    if len(shared_samples) < int(settings["minimum_complete_n"]):
        raise RuntimeError(
            f"Track B complete n={len(shared_samples)} is below "
            f"the prespecified minimum {settings['minimum_complete_n']}."
        )

    clinical_valid = clinical_valid.loc[shared_samples]
    rna = rna.loc[shared_samples]
    cna = cna.loc[shared_samples]
    mutation = mutation.loc[shared_samples]

    time = pd.to_numeric(
        clinical_valid["os_months"],
        errors="coerce",
    ).to_numpy(dtype=float)
    event = pd.to_numeric(
        clinical_valid["os_event"],
        errors="coerce",
    ).to_numpy(dtype=int)
    events = int(event.sum())
    if events < int(settings["minimum_events"]):
        raise RuntimeError(
            f"Track B events={events} is below the prespecified minimum."
        )

    clinical_matrix = prefixed_clinical(
        clinical_valid,
        settings["clinical_features"],
    )

    splitter = StratifiedKFold(
        n_splits=int(settings["outer_folds"]),
        shuffle=True,
        random_state=int(settings["outer_seed"]),
    )

    fold_rows = []
    selected_rows = []
    screening_rows = []
    fold_feature_sets = {}

    for fold, (train_positions, test_positions) in enumerate(
        splitter.split(clinical_valid, event),
        1,
    ):
        train_time = time[train_positions]

        selected_rna, rna_screen = top_signed_spearman_chunked(
            rna,
            train_positions,
            train_time,
            int(settings["rna_top_positive"]),
            int(settings["rna_top_negative"]),
        )
        selected_cna, cna_screen = top_signed_spearman_chunked(
            cna,
            train_positions,
            train_time,
            int(settings["cna_top_positive"]),
            int(settings["cna_top_negative"]),
        )

        mutation_train = mutation.iloc[train_positions]
        mutation_frequency = mutation_train.mean(axis=0)
        mutation_candidates = list(
            mutation_frequency[
                mutation_frequency
                >= float(settings["mutation_frequency_threshold"])
            ]
            .sort_values(ascending=False)
            .index[: int(settings["maximum_mutation_candidates"])]
        )

        for row in rna_screen:
            screening_rows.append({
                "fold": fold,
                "modality": "RNA",
                **row,
                "selected_for_iamb": row["feature"] in selected_rna,
            })
        for row in cna_screen:
            screening_rows.append({
                "fold": fold,
                "modality": "CNV",
                **row,
                "selected_for_iamb": row["feature"] in selected_cna,
            })

        candidate_columns = (
            list(clinical_matrix.columns)
            + selected_rna
            + selected_cna
            + mutation_candidates
        )
        candidate_columns = list(dict.fromkeys(candidate_columns))

        train_raw = pd.concat(
            [
                clinical_matrix.iloc[train_positions],
                rna.iloc[train_positions][selected_rna],
                cna.iloc[train_positions][selected_cna],
                mutation.iloc[train_positions][mutation_candidates],
            ],
            axis=1,
        )
        test_raw = pd.concat(
            [
                clinical_matrix.iloc[test_positions],
                rna.iloc[test_positions][selected_rna],
                cna.iloc[test_positions][selected_cna],
                mutation.iloc[test_positions][mutation_candidates],
            ],
            axis=1,
        )
        train_x, test_x, _ = median_impute_scale_train_test(
            train_raw,
            test_raw,
            scale=True,
        )

        x_ci, y_ci = transform_for_ci(
            train_x.to_numpy(dtype=float),
            train_time,
            engine,
        )
        selected_indices = iamb_select(
            x_ci,
            y_ci,
            float(settings["iamb_alpha"]),
            max_selected=min(120, len(candidate_columns)),
        )
        selected_columns = [
            candidate_columns[index] for index in selected_indices
        ]
        selection_fallback = False
        if not selected_columns:
            selected_indices = pad_selection(
                [],
                x_ci,
                y_ci,
                min(20, len(candidate_columns)),
            )
            selected_columns = [
                candidate_columns[index] for index in selected_indices
            ]
            selection_fallback = True

        selected_train = train_x[selected_columns]
        selected_test = test_x[selected_columns]
        risk, fit_information = fit_cox_risk(
            selected_train,
            time[train_positions],
            event[train_positions],
            selected_test,
            float(settings["cox_penalizer"]),
        )
        c_index = fast_harrell_c_index(
            time[test_positions],
            event[test_positions],
            risk,
        )
        auc_5y, auc_n = binary_auc_at_horizon(
            time[test_positions],
            event[test_positions],
            risk,
            float(settings["five_year_months"]),
        )

        clinical_train, clinical_test, _ = (
            median_impute_scale_train_test(
                clinical_matrix.iloc[train_positions],
                clinical_matrix.iloc[test_positions],
                scale=True,
            )
        )
        clinical_risk, clinical_fit = fit_cox_risk(
            clinical_train,
            time[train_positions],
            event[train_positions],
            clinical_test,
            float(settings["cox_penalizer"]),
        )
        clinical_c_index = fast_harrell_c_index(
            time[test_positions],
            event[test_positions],
            clinical_risk,
        )
        clinical_auc_5y, _ = binary_auc_at_horizon(
            time[test_positions],
            event[test_positions],
            clinical_risk,
            float(settings["five_year_months"]),
        )

        fold_feature_sets[fold] = set(selected_columns)
        fold_rows.append({
            "fold": fold,
            "train_n": len(train_positions),
            "test_n": len(test_positions),
            "train_events": int(event[train_positions].sum()),
            "test_events": int(event[test_positions].sum()),
            "rna_candidates": len(selected_rna),
            "cna_candidates": len(selected_cna),
            "mutation_candidates": len(mutation_candidates),
            "clinical_candidates": len(clinical_matrix.columns),
            "combined_candidates": len(candidate_columns),
            "selected_features": len(selected_columns),
            "selected_clinical": sum(
                column.startswith("CLIN__")
                for column in selected_columns
            ),
            "selected_rna": sum(
                column.startswith("RNA__")
                for column in selected_columns
            ),
            "selected_cna": sum(
                column.startswith("CNA__")
                for column in selected_columns
            ),
            "selected_mutation": sum(
                column.startswith("MUT__")
                for column in selected_columns
            ),
            "selection_fallback": selection_fallback,
            "harrell_c_index": c_index,
            "auc_5y": auc_5y,
            "auc_5y_n": auc_n,
            "clinical_only_c_index": clinical_c_index,
            "clinical_only_auc_5y": clinical_auc_5y,
            "delta_c_index_vs_clinical": c_index - clinical_c_index,
            "delta_auc_5y_vs_clinical": auc_5y - clinical_auc_5y,
            "train_c_index": fit_information["concordance_train"],
            "clinical_train_c_index": clinical_fit["concordance_train"],
            "engine": engine,
            "engine_historical_reproduced": engine_reproduced,
        })

        for rank, column in enumerate(selected_columns, 1):
            if column.startswith("CLIN__"):
                modality = "Clinical"
            elif column.startswith("RNA__"):
                modality = "RNA"
            elif column.startswith("CNA__"):
                modality = "CNV"
            elif column.startswith("MUT__"):
                modality = "Mutation"
            else:
                modality = "Unknown"
            selected_rows.append({
                "fold": fold,
                "rank": rank,
                "feature": column,
                "modality": modality,
            })

        print(
            f"Fold {fold}: candidates={len(candidate_columns):3d}, "
            f"selected={len(selected_columns):3d}, "
            f"C={c_index:.4f} vs clinical {clinical_c_index:.4f}, "
            f"AUC5y={auc_5y:.4f} vs clinical {clinical_auc_5y:.4f}, "
            f"composition=Clin {fold_rows[-1]['selected_clinical']} / "
            f"RNA {fold_rows[-1]['selected_rna']} / "
            f"CNV {fold_rows[-1]['selected_cna']} / "
            f"MUT {fold_rows[-1]['selected_mutation']}"
        )

    stability_rows = summarize_selection_stability(fold_feature_sets)

    frequency = {}
    modality_lookup = {}
    for row in selected_rows:
        frequency[row["feature"]] = (
            frequency.get(row["feature"], 0) + 1
        )
        modality_lookup[row["feature"]] = row["modality"]
    frequency_rows = [
        {
            "feature": feature,
            "modality": modality_lookup[feature],
            "selected_folds": count,
            "selection_frequency": (
                count / int(settings["outer_folds"])
            ),
        }
        for feature, count in sorted(
            frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    write_csv(out / "m34_track_b_fold_results.csv", fold_rows)
    write_csv(out / "m34_track_b_selected_features.csv", selected_rows)
    write_csv(out / "m34_track_b_pairwise_stability.csv", stability_rows)
    write_csv(out / "m34_track_b_selection_frequency.csv", frequency_rows)
    write_csv(out / "m34_track_b_screening_scores.csv", screening_rows)

    summary = {
        "complete_case_n": len(shared_samples),
        "events": events,
        "outer_folds": int(settings["outer_folds"]),
        "engine": engine,
        "historical_engine_reproduced": engine_reproduced,
        "mutation_candidate_universe": 173,
        "mutation_rule": (
            "METABRIC_173 panel-aware zeros; nonsynonymous calls only"
        ),
        "methylation_role_in_this_pilot": (
            "Not included in the historical 525-feature combined benchmark; "
            "reserved for the full modality-specific replication."
        ),
        "mean_c_index": float(np.nanmean([
            row["harrell_c_index"] for row in fold_rows
        ])),
        "sd_c_index": float(np.nanstd([
            row["harrell_c_index"] for row in fold_rows
        ], ddof=1)),
        "mean_auc_5y": float(np.nanmean([
            row["auc_5y"] for row in fold_rows
        ])),
        "sd_auc_5y": float(np.nanstd([
            row["auc_5y"] for row in fold_rows
        ], ddof=1)),
        "mean_clinical_only_c_index": float(np.nanmean([
            row["clinical_only_c_index"] for row in fold_rows
        ])),
        "mean_clinical_only_auc_5y": float(np.nanmean([
            row["clinical_only_auc_5y"] for row in fold_rows
        ])),
        "mean_delta_c_index_vs_clinical": float(np.nanmean([
            row["delta_c_index_vs_clinical"] for row in fold_rows
        ])),
        "mean_delta_auc_5y_vs_clinical": float(np.nanmean([
            row["delta_auc_5y_vs_clinical"] for row in fold_rows
        ])),
        "mean_selected_features": float(np.mean([
            row["selected_features"] for row in fold_rows
        ])),
        "mean_pairwise_jaccard": float(np.mean([
            row["jaccard"] for row in stability_rows
        ])),
        "mean_overlap_coefficient": float(np.mean([
            row["overlap_coefficient"] for row in stability_rows
        ])),
        "historical_external_benchmark": cfg[
            "historical_external_benchmark"
        ],
        "status": (
            "NESTED_RECONSTRUCTED_PAPER1_PILOT_COMPLETE"
            if engine_reproduced else
            "NESTED_PILOT_COMPLETE_ENGINE_NOT_BITWISE_HISTORICAL"
        ),
    }
    (out / "m34_track_b_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTrack B fold results")
    print_table(
        fold_rows,
        [
            "fold", "train_n", "test_n", "combined_candidates",
            "selected_features", "selected_clinical", "selected_rna",
            "selected_cna", "selected_mutation", "harrell_c_index",
            "clinical_only_c_index", "delta_c_index_vs_clinical",
            "auc_5y", "clinical_only_auc_5y",
            "delta_auc_5y_vs_clinical"
        ],
    )

    print("\nPairwise selection stability")
    print_table(
        stability_rows,
        [
            "fold_a", "fold_b", "jaccard",
            "overlap_coefficient", "intersection", "union"
        ],
    )

    print("\nSelection frequency")
    print_table(
        frequency_rows,
        [
            "feature", "modality", "selected_folds",
            "selection_frequency"
        ],
        max_rows=120,
    )

    print("\nTrack B summary")
    print(json.dumps(summary, indent=2))

    print(
        "\nPASS: memory-safe nested Track B pilot completed. "
        "All supervised steps were confined to training folds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
