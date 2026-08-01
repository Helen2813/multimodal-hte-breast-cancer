from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _stage22_utils import (
    as_float,
    assert_close,
    ensure_dirs,
    find_root,
    first_existing,
    load_config,
    pick_col,
    print_frame,
    read_csv,
    read_one_row,
    verify_manifest,
)


def main() -> None:
    root = find_root(Path.cwd())
    config = load_config(root)
    dirs = ensure_dirs(root, config)
    tolerance = float(config.get("recompute_tolerance", 1e-6))

    paths = {
        "point": first_existing(root, ["results/tables/79_candidate_v9_final_point_estimate.csv"]),
        "repetitions": first_existing(root, ["results/tables/82_publication_bootstrap_repetitions_checkpoint.csv"]),
        "partitions": first_existing(root, ["results/tables/82_publication_bootstrap_partitions_checkpoint.csv"]),
        "errors": first_existing(root, ["results/tables/82_publication_bootstrap_errors.csv"]),
        "summary": first_existing(root, ["results/tables/83_publication_bootstrap_summary.csv"]),
        "decision": first_existing(root, ["results/tables/84_publication_bootstrap_decision.csv"]),
        "analysis_plan": first_existing(root, ["paper_A_treatment_effects/analysis_plan_FINAL.md"]),
        "manifest": first_existing(root, ["data/derived/manifests/80_candidate_v9_protocol_lock_manifest.json"]),
    }

    point_row = read_one_row(paths["point"])
    summary_row = read_one_row(paths["summary"])
    decision_row = read_one_row(paths["decision"])
    reps = read_csv(paths["repetitions"])
    partitions = read_csv(paths["partitions"])
    errors = read_csv(paths["errors"])

    effect_col = pick_col(reps, ["aggregated_effect_days", "estimate_days", "effect_days"], "bootstrap effect")
    rep_col = pick_col(reps, ["bootstrap_repetition", "repetition", "rep"], "bootstrap repetition")
    point = as_float(point_row.get("estimate_days", point_row.get("locked_point_estimate_days")))
    expected_reps = int(config["expected_bootstrap_repetitions"])
    expected_partitions = int(config["expected_partitions_per_bootstrap"])

    clean_effects = pd.to_numeric(reps[effect_col], errors="coerce").dropna()
    recomputed = {
        "mean": float(clean_effects.mean()),
        "median": float(clean_effects.median()),
        "sd": float(clean_effects.std(ddof=1)),
        "q025": float(clean_effects.quantile(0.025)),
        "q975": float(clean_effects.quantile(0.975)),
        "fraction_positive": float((clean_effects > 0).mean()),
    }

    checks: list[dict[str, object]] = []

    def add(name: str, observed: object, expected: object, passed: bool) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "pass": bool(passed)})

    add("bootstrap_repetition_rows", len(reps), expected_reps, len(reps) == expected_reps)
    add("unique_bootstrap_repetitions", reps[rep_col].nunique(), expected_reps, reps[rep_col].nunique() == expected_reps)
    add("finite_bootstrap_effects", len(clean_effects), expected_reps, len(clean_effects) == expected_reps)
    add(
        "partition_fit_rows",
        len(partitions),
        expected_reps * expected_partitions,
        len(partitions) == expected_reps * expected_partitions,
    )
    add("persistent_error_rows", len(errors), 0, len(errors) == 0)

    status = str(decision_row.get("protocol_status", ""))
    decision = str(decision_row.get("stage21_decision", ""))
    add("protocol_status", status, "PAPER_A_CANDIDATE_V9_ANALYSIS_COMPLETE", status == "PAPER_A_CANDIDATE_V9_ANALYSIS_COMPLETE")
    add("stage21_decision", decision, "FULL_BOOTSTRAP_COMPLETE_DIRECTION_IMPRECISE", decision == "FULL_BOOTSTRAP_COMPLETE_DIRECTION_IMPRECISE")

    mappings = [
        ("bootstrap_mean_days", recomputed["mean"]),
        ("bootstrap_median_days", recomputed["median"]),
        ("bootstrap_sd_days", recomputed["sd"]),
        ("percentile_ci_low_days", recomputed["q025"]),
        ("percentile_ci_high_days", recomputed["q975"]),
        ("fraction_positive", recomputed["fraction_positive"]),
    ]
    for column, observed in mappings:
        expected = as_float(summary_row[column])
        try:
            assert_close(observed, expected, tolerance, column)
            passed = True
        except RuntimeError:
            passed = False
        add(f"recomputed_{column}", observed, expected, passed)

    summary_point = as_float(summary_row["locked_point_estimate_days"])
    add("locked_point_matches_summary", point, summary_point, abs(point - summary_point) <= tolerance)

    manifest_df = verify_manifest(root)
    manifest_pass = bool(len(manifest_df) > 0 and manifest_df["match"].fillna(False).all())
    add("locked_manifest_hashes", int(manifest_df["match"].fillna(False).sum()), len(manifest_df), manifest_pass)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(dirs["audit"] / "85_stage22_preflight_checks.csv", index=False)
    manifest_df.to_csv(dirs["audit"] / "85_locked_manifest_verification.csv", index=False)

    print_frame("STAGE 85 - PUBLICATION-ASSET PREFLIGHT", checks_df)
    print("\nCandidate V9 locked result")
    print(f"  Point estimate: {point:.6f} days")
    print(f"  Percentile 95% CI: [{recomputed['q025']:.6f}, {recomputed['q975']:.6f}] days")
    print(f"  Fraction positive: {recomputed['fraction_positive']:.6f}")
    print(f"  Bootstrap repetitions: {len(reps)}/{expected_reps}")
    print(f"  Partition fits: {len(partitions)}/{expected_reps * expected_partitions}")

    if not checks_df["pass"].all():
        failed = checks_df.loc[~checks_df["pass"], "check"].tolist()
        raise SystemExit("Stage 22 preflight failed: " + ", ".join(failed))

    print("\nPASS: Stage 21 results are complete, recomputable, and safe for publication-asset generation.")


if __name__ == "__main__":
    main()
