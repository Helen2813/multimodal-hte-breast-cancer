from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import DERIVED_DIR, RESULTS_DIR, ensure_dirs, read_table


warnings.filterwarnings("ignore", category=ConvergenceWarning)

POST_BASELINE_TOKENS = (
    "treatment",
    "therapy",
    "therapeutic",
    "drug",
    "pharmaceutical",
    "radiation",
    "days_to_death",
    "days_to_last_follow",
    "vital_status",
    "survival",
    "follow_up",
    "followup",
    "created",
    "updated",
)
ADMIN_TOKENS = (
    "staging_system",
    "edition",
    "submitter",
    "project",
    "sample",
    "case_id",
)
EXPLICIT_EXCLUDE = {
    "ER_status",
    "PR_status",
    "HER2_status",
    "OS",
    "OS.time",
    "OS_time",
    "Y",
    "Y_died_5yr",
    "analysis_treatment",
    "analysis_event",
    "analysis_time",
}


def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    return float(mean), float(var)


def weighted_smd(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = weighted_stats(x[t == 1], w[t == 1])
    m0, v0 = weighted_stats(x[t == 0], w[t == 0])
    pooled = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return np.nan
    return float((m1 - m0) / pooled)


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return (
        float((w.sum() ** 2) / np.sum(w**2))
        if np.sum(w**2) > 0
        else np.nan
    )


def full_clinical_columns(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rows = []
    included = []

    for col in map(str, df.columns):
        low = col.lower()
        reason = ""
        candidate = (
            col.startswith("CLIN_")
            or col.startswith("pathology_details.")
        )
        if not candidate:
            continue

        if col in EXPLICIT_EXCLUDE:
            reason = "explicit_outcome_treatment_or_receptor_exclusion"
        elif any(token in low for token in POST_BASELINE_TOKENS):
            reason = "possible_post_baseline_or_outcome_information"
        elif any(token in low for token in ADMIN_TOKENS):
            reason = "administrative_or_staging_metadata"
        else:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() < max(20, int(0.10 * len(df))):
                reason = "too_few_numeric_values"
            elif numeric.nunique(dropna=True) <= 1:
                reason = "constant"
            else:
                reason = "included_baseline_clinical_sensitivity"
                included.append(col)

        rows.append(
            {
                "column": col,
                "decision": "include" if col in included else "exclude",
                "reason": reason,
            }
        )

    return included, pd.DataFrame(rows)


def add_diagnosis_year(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    year_path = (
        DERIVED_DIR / "manifests" / "20_patient_diagnosis_year.csv"
    )
    if not year_path.exists():
        return df.copy(), False
    years = read_table(year_path)[
        ["patient_id_normalized", "diagnosis_year"]
    ]
    out = df.merge(
        years,
        on="patient_id_normalized",
        how="left",
        validate="one_to_one",
    )
    return out, True


def cross_fitted_elastic_net(
    X: pd.DataFrame, t: np.ndarray, seed: int = 221
) -> tuple[np.ndarray, pd.DataFrame]:
    counts = np.bincount(t, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 3:
        raise ValueError(f"Too few patients per treatment class: {counts}")

    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    ps = np.full(len(t), np.nan)
    tuning_rows = []

    for fold, (train, test) in enumerate(
        splitter.split(X, t), start=1
    ):
        inner = max(
            2,
            int(min(4, np.bincount(t[train], minlength=2).min())),
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegressionCV(
                        Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                        cv=inner,
                        scoring="neg_log_loss",
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
                        max_iter=12000,
                        n_jobs=-1,
                        random_state=seed + fold,
                        refit=True,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train], t[train])
        ps[test] = model.predict_proba(X.iloc[test])[:, 1]
        fitted = model.named_steps["logit"]
        tuning_rows.append(
            {
                "fold": fold,
                "chosen_C": float(fitted.C_[0]),
                "chosen_l1_ratio": float(fitted.l1_ratio_[0]),
                "nonzero_coefficients": int(
                    np.count_nonzero(np.abs(fitted.coef_) > 1e-10)
                ),
            }
        )

    return np.clip(ps, 0.01, 0.99), pd.DataFrame(tuning_rows)


def balance_table(
    df: pd.DataFrame,
    features: list[str],
    t: np.ndarray,
    weights: np.ndarray,
    cohort: str,
    strategy: str,
) -> pd.DataFrame:
    rows = []
    for col in features:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        rows.append(
            {
                "cohort": cohort,
                "strategy": strategy,
                "feature": col,
                "smd_unweighted": weighted_smd(
                    x, t, np.ones(len(t))
                ),
                "smd_weighted": weighted_smd(x, t, weights),
                "missing_fraction": float(np.mean(~np.isfinite(x))),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_smd_unweighted"] = out["smd_unweighted"].abs()
    out["abs_smd_weighted"] = out["smd_weighted"].abs()
    return out.sort_values("abs_smd_weighted", ascending=False)


def main() -> int:
    ensure_dirs()
    table_dir = RESULTS_DIR / "tables"
    cohort_dir = DERIVED_DIR / "verified_cohorts"
    compact_dir = DERIVED_DIR / "verified_compact_adjustment"

    print("=" * 110)
    print("STAGE 21 — COMPACT VS FULL ELASTIC-NET PROPENSITY")
    print("=" * 110)

    cohort_paths = sorted(cohort_dir.glob("*_verified.csv"))
    if not cohort_paths:
        raise FileNotFoundError("Verified cohorts were not found.")

    summary_rows = []
    manifest_frames = []

    for cohort_path in cohort_paths:
        cohort = cohort_path.stem.replace("_verified", "")
        print("\n" + "=" * 110)
        print(f"COHORT: {cohort}")
        print("=" * 110)

        df = read_table(cohort_path)
        df, year_added = add_diagnosis_year(df)
        t = pd.to_numeric(
            df["analysis_treatment"], errors="raise"
        ).astype(int).to_numpy()

        compact_path = (
            compact_dir / f"{cohort}_compact_verified.csv"
        )
        compact_weight_path = (
            table_dir / f"19_verified_weights_{cohort}.csv"
        )
        if not compact_path.exists() or not compact_weight_path.exists():
            raise FileNotFoundError(
                f"Missing Stage 19 compact files for {cohort}"
            )

        compact = read_table(compact_path)
        compact_weights = read_table(compact_weight_path)
        compact_merged = compact.merge(
            compact_weights[
                ["patient_id_normalized", "overlap_weight"]
            ],
            on="patient_id_normalized",
            how="inner",
            validate="one_to_one",
        )
        compact_features = [
            c for c in compact_merged.columns if c.startswith("W_")
        ]
        if year_added:
            compact_merged = compact_merged.merge(
                df[["patient_id_normalized", "diagnosis_year"]],
                on="patient_id_normalized",
                how="left",
                validate="one_to_one",
            )
            if (
                compact_merged["diagnosis_year"].notna().sum()
                >= max(20, int(0.5 * len(compact_merged)))
                and compact_merged["diagnosis_year"].nunique(dropna=True) > 1
            ):
                compact_features.append("diagnosis_year")

        compact_t = pd.to_numeric(
            compact_merged["analysis_treatment"], errors="raise"
        ).astype(int).to_numpy()
        compact_w = pd.to_numeric(
            compact_merged["overlap_weight"], errors="raise"
        ).to_numpy(float)
        compact_balance = balance_table(
            compact_merged,
            compact_features,
            compact_t,
            compact_w,
            cohort,
            "compact_overlap",
        )
        compact_balance.to_csv(
            table_dir / f"21_balance_compact_{cohort}.csv",
            index=False,
        )

        full_features, feature_manifest = full_clinical_columns(df)
        if year_added and "diagnosis_year" in df.columns:
            numeric_year = pd.to_numeric(
                df["diagnosis_year"], errors="coerce"
            )
            if (
                numeric_year.notna().sum() >= max(20, int(0.5 * len(df)))
                and numeric_year.nunique(dropna=True) > 1
            ):
                full_features.append("diagnosis_year")
                feature_manifest = pd.concat(
                    [
                        feature_manifest,
                        pd.DataFrame(
                            [
                                {
                                    "column": "diagnosis_year",
                                    "decision": "include",
                                    "reason": "verified_calendar_era_sensitivity",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        if len(full_features) < 3:
            raise ValueError(
                f"{cohort}: too few full clinical sensitivity features."
            )

        ps, tuning = cross_fitted_elastic_net(
            df[full_features].apply(pd.to_numeric, errors="coerce"),
            t,
        )
        full_w = np.where(t == 1, 1 - ps, ps)
        full_balance = balance_table(
            df, full_features, t, full_w, cohort, "full_elastic_net_overlap"
        )
        full_balance.to_csv(
            table_dir / f"21_balance_full_{cohort}.csv",
            index=False,
        )
        tuning["cohort"] = cohort
        tuning.to_csv(
            table_dir / f"21_tuning_full_{cohort}.csv", index=False
        )

        pd.DataFrame(
            {
                "patient_id_normalized": df["patient_id_normalized"],
                "analysis_treatment": t,
                "propensity_score_oof_full_elastic_net": ps,
                "overlap_weight_full_elastic_net": full_w,
            }
        ).to_csv(
            table_dir / f"21_full_weights_{cohort}.csv",
            index=False,
        )

        feature_manifest["cohort"] = cohort
        manifest_frames.append(feature_manifest)

        print(
            f"n={len(df)}; treated={int(t.sum())}; "
            f"controls={int((t == 0).sum())}; "
            f"events={int(pd.to_numeric(df['analysis_event']).sum())}"
        )
        print(
            f"Diagnosis year included: {year_added and 'diagnosis_year' in full_features}"
        )
        print(
            f"Compact features assessed: {len(compact_features)}; "
            f"full elastic-net candidate features: {len(full_features)}"
        )

        print("\nElastic-net tuning by outer fold")
        print(tuning.to_string(index=False))

        print("\nTop compact-set residual imbalances")
        print(
            compact_balance[
                ["feature", "smd_unweighted", "smd_weighted"]
            ].head(12).to_string(index=False)
        )

        print("\nTop full-set residual imbalances")
        print(
            full_balance[
                ["feature", "smd_unweighted", "smd_weighted"]
            ].head(15).to_string(index=False)
        )

        compact_max = float(
            compact_balance["abs_smd_weighted"].max()
        )
        full_max = float(full_balance["abs_smd_weighted"].max())
        compact_ess_t = ess(compact_w[compact_t == 1])
        compact_ess_c = ess(compact_w[compact_t == 0])
        full_ess_t = ess(full_w[t == 1])
        full_ess_c = ess(full_w[t == 0])

        if (
            compact_max <= 0.10
            and full_max <= 0.10
            and min(compact_ess_c, full_ess_c) >= 50
        ):
            status = "ROBUST_PRIMARY_READY"
        elif (
            compact_max <= 0.10
            and full_max <= 0.15
            and min(compact_ess_c, full_ess_c) >= 30
        ):
            status = "PRIMARY_COMPACT_FULL_SENSITIVITY_ACCEPTABLE"
        elif (
            min(compact_max, full_max) <= 0.15
            and max(compact_ess_c, full_ess_c) >= 25
        ):
            status = "EXPLORATORY_ONLY"
        else:
            status = "NOT_CAUSAL_READY"

        row = {
            "cohort": cohort,
            "n": len(df),
            "treated": int(t.sum()),
            "control": int((t == 0).sum()),
            "events": int(pd.to_numeric(df["analysis_event"]).sum()),
            "diagnosis_year_included": int(
                year_added and "diagnosis_year" in full_features
            ),
            "n_compact_features_evaluated": len(compact_features),
            "n_full_elastic_net_features": len(full_features),
            "compact_max_abs_smd": compact_max,
            "full_max_abs_smd": full_max,
            "compact_mean_abs_smd": float(
                compact_balance["abs_smd_weighted"].mean()
            ),
            "full_mean_abs_smd": float(
                full_balance["abs_smd_weighted"].mean()
            ),
            "compact_ess_treated": compact_ess_t,
            "compact_ess_control": compact_ess_c,
            "full_ess_treated": full_ess_t,
            "full_ess_control": full_ess_c,
            "full_ps_min": float(ps.min()),
            "full_ps_p05": float(np.quantile(ps, 0.05)),
            "full_ps_median": float(np.median(ps)),
            "full_ps_p95": float(np.quantile(ps, 0.95)),
            "full_ps_max": float(ps.max()),
            "decision_status": status,
        }
        summary_rows.append(row)
        print("\nCohort decision")
        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(summary_rows).sort_values("cohort")
    summary.to_csv(
        table_dir / "21_propensity_strategy_summary.csv",
        index=False,
    )
    pd.concat(manifest_frames, ignore_index=True).to_csv(
        table_dir / "21_full_clinical_feature_manifest.csv",
        index=False,
    )

    print("\n" + "=" * 110)
    print("FINAL STAGE 21 PROPENSITY STRATEGY SUMMARY")
    print("=" * 110)
    print(summary.to_string(index=False))
    print(f"\nOutputs saved under: {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
