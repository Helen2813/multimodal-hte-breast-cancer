#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _stage17_utils import (
    dataframe_console,
    effect_and_patient_components,
    ensure_stage17_dirs,
    exact_landmark_payload,
    load_stage17_config,
    markdown_table,
    original_fold_effects,
    patient_censoring_diagnostics,
    project_root,
    write_csv,
    write_text,
)


def pooled_smd(x: np.ndarray, group: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    group = np.asarray(group, dtype=bool)
    x1 = x[group & np.isfinite(x)]
    x0 = x[(~group) & np.isfinite(x)]
    if len(x1) < 2 or len(x0) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0 if np.isclose(np.mean(x1), np.mean(x0)) else float("nan")
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def main() -> int:
    root = project_root()
    ensure_stage17_dirs(root)
    cfg = load_stage17_config(root)
    tables = root / "results/tables"
    local = root / "data/derived/stage17"

    payload = exact_landmark_payload()
    horizon = float(payload["horizon"])
    g_min = float(cfg["repeated_crossfit"]["primary_g_min"])

    y = np.asarray(payload["y"], dtype=float)
    a = np.asarray(payload["a"], dtype=int)
    event = np.asarray(payload["event"], dtype=int)
    observed_time = np.asarray(payload["observed_time"], dtype=float)
    e = np.asarray(payload["e"], dtype=float)
    fold = np.asarray(payload["fold"], dtype=int)
    mu0 = np.clip(np.asarray(payload["mu0"], dtype=float), 0.0, horizon)
    mu1 = np.clip(np.asarray(payload["mu1"], dtype=float), 0.0, horizon)

    summary, components = effect_and_patient_components(y, a, e, mu0, mu1)
    censor_diag = patient_censoring_diagnostics(
        observed_time,
        np.asarray(payload["G"], dtype=float),
        np.asarray(payload["starts"], dtype=float),
        g_min,
    )

    patient = pd.DataFrame(
        {
            "local_row_index": np.arange(len(a)),
            "original_fold": fold,
            "treatment": a,
            "event": event,
            "observed_time": observed_time,
            "propensity": e,
            "overlap_h": e * (1.0 - e),
            "ipcw_rmst_pseudo": y,
            "bounded_mu0": mu0,
            "bounded_mu1": mu1,
        }
    )
    if "diagnosis_year" in payload["frame"].columns:
        patient["diagnosis_year"] = pd.to_numeric(
            payload["frame"]["diagnosis_year"], errors="coerce"
        ).to_numpy(float)
    patient = pd.concat(
        [
            patient.reset_index(drop=True),
            censor_diag.reset_index(drop=True),
            components.reset_index(drop=True),
        ],
        axis=1,
    )
    patient["absolute_influence"] = patient["influence"].abs()

    fold_effects, loo = original_fold_effects(y, a, e, mu0, mu1, fold)
    fold_rows: list[dict[str, float | int]] = []
    top25_indices = set(patient.nlargest(25, "absolute_influence")["local_row_index"].astype(int))
    for f in sorted(np.unique(fold)):
        mask = fold == f
        p = patient.loc[mask]
        effect = float(
            fold_effects.loc[fold_effects["fold"] == f, "fold_effect_days"].iloc[0]
        )
        row: dict[str, float | int] = {
            "fold": int(f),
            "n": int(mask.sum()),
            "treated": int(a[mask].sum()),
            "control": int((1 - a[mask]).sum()),
            "events": int(event[mask].sum()),
            "fold_effect_days": effect,
            "observed_time_median": float(np.median(observed_time[mask])),
            "observed_time_p10": float(np.quantile(observed_time[mask], 0.10)),
            "observed_time_p90": float(np.quantile(observed_time[mask], 0.90)),
            "propensity_mean": float(np.mean(e[mask])),
            "propensity_min": float(np.min(e[mask])),
            "propensity_max": float(np.max(e[mask])),
            "pseudo_mean": float(np.mean(y[mask])),
            "pseudo_sd": float(np.std(y[mask], ddof=1)),
            "pseudo_p95": float(np.quantile(y[mask], 0.95)),
            "pseudo_p99": float(np.quantile(y[mask], 0.99)),
            "pseudo_max": float(np.max(y[mask])),
            "g_min_at_risk_min": float(p["g_min_at_risk_raw"].min()),
            "g_min_at_risk_median": float(p["g_min_at_risk_raw"].median()),
            "max_inverse_g_raw": float(p["max_inverse_g_raw"].max()),
            "sum_absolute_influence": float(p["absolute_influence"].sum()),
            "max_absolute_influence": float(p["absolute_influence"].max()),
            "top25_influence_count": int(p["local_row_index"].isin(top25_indices).sum()),
        }
        if "diagnosis_year" in p.columns:
            row["diagnosis_year_mean"] = float(p["diagnosis_year"].mean())
            row["diagnosis_year_missing"] = int(p["diagnosis_year"].isna().sum())
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows)

    fold2 = fold == 2
    comparison_vars = [
        "treatment",
        "event",
        "observed_time",
        "propensity",
        "ipcw_rmst_pseudo",
        "g_min_at_risk_raw",
        "max_inverse_g_raw",
        "absolute_influence",
    ]
    if "diagnosis_year" in patient.columns:
        comparison_vars.append("diagnosis_year")
    comparison_rows = []
    for col in comparison_vars:
        x = pd.to_numeric(patient[col], errors="coerce").to_numpy(float)
        x2 = x[fold2 & np.isfinite(x)]
        xo = x[(~fold2) & np.isfinite(x)]
        comparison_rows.append(
            {
                "variable": col,
                "fold2_n_nonmissing": int(len(x2)),
                "other_folds_n_nonmissing": int(len(xo)),
                "fold2_mean": float(np.mean(x2)) if len(x2) else np.nan,
                "other_folds_mean": float(np.mean(xo)) if len(xo) else np.nan,
                "fold2_median": float(np.median(x2)) if len(x2) else np.nan,
                "other_folds_median": float(np.median(xo)) if len(xo) else np.nan,
                "fold2_p95": float(np.quantile(x2, 0.95)) if len(x2) else np.nan,
                "other_folds_p95": float(np.quantile(xo, 0.95)) if len(xo) else np.nan,
                "fold2_vs_others_smd": pooled_smd(x, fold2),
            }
        )
    fold2_comparison = pd.DataFrame(comparison_rows)
    fold2_comparison["absolute_smd"] = fold2_comparison["fold2_vs_others_smd"].abs()
    fold2_comparison = fold2_comparison.sort_values("absolute_smd", ascending=False)

    top = patient.nlargest(25, "absolute_influence").copy()
    public_cols = [
        "local_row_index",
        "original_fold",
        "treatment",
        "event",
        "observed_time",
        "propensity",
        "ipcw_rmst_pseudo",
        "g_min_at_risk_raw",
        "max_inverse_g_raw",
        "bounded_mu0",
        "bounded_mu1",
        "normalized_contribution_days",
        "influence",
        "absolute_influence",
    ]
    if "diagnosis_year" in top.columns:
        public_cols.insert(5, "diagnosis_year")
    top_public = top[public_cols]

    overall = pd.DataFrame(
        [
            {
                "model": "arm_ridge_bounded",
                **summary,
                "n": len(patient),
                "treated": int(a.sum()),
                "control": int((1 - a).sum()),
                "events": int(event.sum()),
                "pseudo_mean": float(np.mean(y)),
                "pseudo_sd": float(np.std(y, ddof=1)),
                "pseudo_p99": float(np.quantile(y, 0.99)),
                "pseudo_max": float(np.max(y)),
                "g_min_at_risk_min": float(patient["g_min_at_risk_raw"].min()),
                "max_inverse_g_raw": float(patient["max_inverse_g_raw"].max()),
                "fold_effect_spread_days": float(
                    fold_effects["fold_effect_days"].max()
                    - fold_effects["fold_effect_days"].min()
                ),
                "loo_effect_spread_days": float(
                    loo["leave_one_fold_out_effect_days"].max()
                    - loo["leave_one_fold_out_effect_days"].min()
                ),
            }
        ]
    )

    write_csv(overall, tables / "67_original_split_influence_summary.csv")
    write_csv(fold_summary, tables / "67_original_fold_forensics.csv")
    write_csv(fold2_comparison, tables / "67_fold2_vs_other_folds.csv")
    write_csv(fold_effects, tables / "67_original_fold_effects_bounded.csv")
    write_csv(loo, tables / "67_original_leave_one_fold_out_bounded.csv")
    write_csv(top_public, tables / "67_top25_influence_rows_deidentified.csv")
    write_csv(patient, local / "67_patient_influence_forensics_LOCAL_ONLY.csv")

    write_text(
        f"""# Stage 17 original-split influence and fold-2 forensic audit

## Overall bounded-ridge estimator

{markdown_table(overall)}

## Fold-level diagnostics

{markdown_table(fold_summary)}

## Fold 2 versus all other folds

{markdown_table(fold2_comparison)}

## Top 25 deidentified influence rows

{markdown_table(top_public, max_rows=25)}

The full patient-level diagnostic table is local-only and contains no patient identifiers in the
public results table. Fold-specific estimates are diagnostics, not independent causal estimates.
""",
        tables / "67_influence_and_fold2_forensics.md",
    )

    print("=" * 124)
    print("STAGE 67 — PATIENT INFLUENCE AND ORIGINAL FOLD-2 FORENSIC AUDIT")
    print("=" * 124)
    print("Overall bounded-ridge estimator")
    print(dataframe_console(overall))
    print("\nFold-level forensic summary")
    print(dataframe_console(fold_summary))
    print("\nFold 2 versus all other folds")
    print(dataframe_console(fold2_comparison))
    print("\nOriginal fold-specific effects")
    print(dataframe_console(fold_effects))
    print("\nOriginal leave-one-fold-out effects")
    print(dataframe_console(loo))
    print("\nTop 25 deidentified influence rows")
    print(dataframe_console(top_public))
    print("\nPatient identifiers are not printed to the terminal or written to public Stage 17 tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
