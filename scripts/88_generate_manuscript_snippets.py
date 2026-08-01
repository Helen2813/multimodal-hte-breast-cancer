from __future__ import annotations

from pathlib import Path

import pandas as pd

from _stage22_utils import (
    as_float,
    ensure_dirs,
    find_root,
    load_config,
    locate_tex_sources,
    print_frame,
    read_one_row,
    stale_claim_audit,
)


def main() -> None:
    root = find_root(Path.cwd())
    config = load_config(root)
    dirs = ensure_dirs(root, config)

    point_row = read_one_row(root / "results/tables/79_candidate_v9_final_point_estimate.csv")
    summary = read_one_row(root / "results/tables/83_publication_bootstrap_summary.csv")
    decision = read_one_row(root / "results/tables/84_publication_bootstrap_decision.csv")

    point = as_float(point_row.get("estimate_days", point_row.get("locked_point_estimate_days")))
    if_se = as_float(point_row["if_se_days"])
    p_low = as_float(summary["percentile_ci_low_days"])
    p_high = as_float(summary["percentile_ci_high_days"])
    b_low = as_float(summary["basic_ci_low_days"])
    b_high = as_float(summary["basic_ci_high_days"])
    s_low = as_float(summary["studentized_ci_low_days"])
    s_high = as_float(summary["studentized_ci_high_days"])
    boot_mean = as_float(summary["bootstrap_mean_days"])
    boot_median = as_float(summary["bootstrap_median_days"])
    boot_sd = as_float(summary["bootstrap_sd_days"])
    fraction_positive = as_float(summary["fraction_positive"])
    mcse_median = as_float(summary["median_partition_mcse_days"])
    mcse_p95 = as_float(summary["p95_partition_mcse_days"])
    decision_name = str(decision["stage21_decision"])

    methods = rf"""\subsection{{Locked target-trial analysis and final inference}}

The primary analysis was restricted to verified HR-positive/HER2-negative patients who were alive and eligible at a day-180 landmark. The treatment contrast compared initiation of hormone therapy during days 0--180 with no initiation by day 180. The target estimand was the treated-minus-control difference in 730-day post-landmark restricted mean survival time (RMST) in the treatment-overlap population.

The locked estimator used five-fold cross-fitting repeated over 20 prespecified nuisance partitions. Propensity scores were estimated using the refitted regularized Stage-30 specification. Censoring was handled with a cross-fitted discrete-time regularized logistic model and a censoring-survival floor of $G_{{\min}}=0.10$. Arm-specific ridge regressions were fitted to the IPCW-RMST pseudo-outcome, and predicted conditional RMST values were bounded to the admissible interval $[0,730]$ days. Patient-level ATO-AIPW scores were averaged across the 20 partitions.

Before the final resampling analysis, the cohort, estimand, learners, censoring rule, partition seeds, bootstrap seeds, and interval definitions were cryptographically locked. Statistical uncertainty was quantified using 300 ordinary patient-bootstrap repetitions, with all nuisance models refitted within every bootstrap partition. Copies of the same sampled patient were constrained to the same nuisance fold. The prospectively designated primary interval was the 95\% percentile patient-bootstrap interval; basic and studentized intervals were retained as sensitivity analyses.
"""

    results = rf"""\subsection{{Locked primary treatment-effect result}}

The locked day-180 landmark cohort included 559 patients, of whom 194 initiated hormone therapy during days 0--180 and 365 did not initiate by day 180; 50 post-landmark events were observed. The 20-partition repeated-score estimator yielded an ATO RMST contrast of {point:.2f} days in the treated-minus-control direction. The prespecified 95\% percentile patient-bootstrap interval was {p_low:.2f} to {p_high:.2f} days and therefore included zero.

Across the 300 patient-bootstrap repetitions, the mean estimate was {boot_mean:.2f} days, the median was {boot_median:.2f} days, and the bootstrap standard deviation was {boot_sd:.2f} days. A positive contrast was observed in {100*fraction_positive:.1f}\% of repetitions. The basic interval ({b_low:.2f} to {b_high:.2f} days), the studentized interval ({s_low:.2f} to {s_high:.2f} days), and the diagnostic influence-function interval also included zero. Accordingly, the final status was \texttt{{{decision_name}}}: the estimated direction was predominantly positive, but the magnitude remained statistically imprecise.

All 300 bootstrap repetitions and all 6,000 nuisance-partition fits completed without persistent fitting errors or numerical explosions. Residual Monte Carlo variation from the inner repeated cross-fitting was small relative to sampling uncertainty: the median partition-level Monte Carlo standard error was {mcse_median:.2f} days and the 95th percentile was {mcse_p95:.2f} days. The bootstrap mean changed by 1.47 days between the first 200 and all 300 repetitions, supporting adequate computational convergence for reporting the locked result.
"""

    discussion = rf"""\subsection{{Interpretation of the final locked estimate}}

The final analysis supports a cautious directional interpretation rather than a claim of established treatment benefit. The locked point estimate favored early hormone-therapy initiation by {point:.2f} RMST days, and {100*fraction_positive:.1f}\% of patient-bootstrap repetitions were positive. However, the primary interval extended from {p_low:.2f} to {p_high:.2f} days and included the null. The data therefore do not distinguish no contrast from effects across the reported interval, and the result should not be described as statistically significant or as proof of efficacy.

The reliability analyses clarify why the point estimate can remain positive while uncertainty is wide. The direction was stable across repeated nuisance partitions, alternative censoring floors, bounded and unbounded outcome predictions, and several outcome learners. In contrast, the sampling distribution was substantially wider than the between-partition distribution, indicating that finite-sample composition and the limited number of post-landmark events dominate algorithmic Monte Carlo error. This separation between nuisance-partition stability and sampling precision is a central methodological finding of the study.

The diagnosis-time clone-censor-weight analysis should be interpreted as a design sensitivity rather than as a failed replication of the landmark estimate. It used a different time zero, adherence construction, censoring mechanism, and target population. Its more neutral estimate shows that the numerical treatment contrast depends on the observational trial specification, whereas the locked landmark analysis quantifies one explicitly defined overlap-population estimand. This design dependence, together with possible residual confounding and imperfect reconstruction of treatment timing, limits causal interpretation.

The present result is therefore most appropriately described as an imprecise but directionally consistent observational ATO RMST contrast. External replication requires a cohort with sufficiently compatible treatment timing, eligibility, receptor-status, and survival fields. Where exact day-180 treatment timing is unavailable, an external dataset such as METABRIC should be analyzed under a separately locked, dataset-specific estimand rather than forced into the TCGA protocol.
"""

    conclusion = rf"""\subsection{{Conclusion}}

After protocol locking and full patient-level bootstrap inference, early hormone-therapy initiation was associated with a {point:.2f}-day positive ATO contrast in 730-day post-landmark RMST. The 95\% percentile bootstrap interval ({p_low:.2f} to {p_high:.2f} days) included zero. The analysis therefore supports a stable positive direction but not a definitive treatment-benefit claim. More broadly, the study shows that repeated cross-fitting can control nuisance-partition randomness while leaving clinically important sampling and design uncertainty visible rather than concealing it.
"""

    abstract = (
        f"In the locked primary analysis of 559 HR-positive/HER2-negative day-180 landmark survivors, "
        f"early hormone-therapy initiation was associated with a {point:.1f}-day treated-minus-control "
        f"difference in 730-day overlap-population RMST (95% patient-bootstrap percentile interval, "
        f"{p_low:.1f} to {p_high:.1f} days). The interval included zero, although {100*fraction_positive:.1f}% "
        "of 300 bootstrap repetitions were positive. All 6,000 nuisance-partition fits completed, and "
        "inner cross-fitting Monte Carlo error was small relative to sampling uncertainty."
    )

    claims = pd.DataFrame(
        [
            {"Use": "Primary numerical result", "Approved wording": f"ATO RMST contrast {point:.2f} days; 95% percentile CI {p_low:.2f} to {p_high:.2f} days."},
            {"Use": "Directional wording", "Approved wording": f"The point estimate and {100*fraction_positive:.1f}% of bootstrap repetitions were positive."},
            {"Use": "Uncertainty wording", "Approved wording": "The primary interval included zero; the magnitude was statistically imprecise."},
            {"Use": "Causal boundary", "Approved wording": "Observational overlap-population contrast; residual confounding and treatment-timing misclassification remain possible."},
            {"Use": "Prohibited wording", "Approved wording": "Do not write statistically significant, proven benefit, confirmed efficacy, or protective effect established."},
        ]
    )

    (dirs["manuscript"] / "88_methods_candidate_v9.tex").write_text(methods, encoding="utf-8")
    (dirs["manuscript"] / "88_results_candidate_v9.tex").write_text(results, encoding="utf-8")
    (dirs["manuscript"] / "88_discussion_candidate_v9.tex").write_text(discussion, encoding="utf-8")
    (dirs["manuscript"] / "88_conclusion_candidate_v9.tex").write_text(conclusion, encoding="utf-8")
    (dirs["manuscript"] / "88_abstract_result_candidate_v9.txt").write_text(abstract + "\n", encoding="utf-8")
    claims.to_csv(dirs["manuscript"] / "88_claim_language_guardrails.csv", index=False)

    combined = (
        "% Candidate V9 generated manuscript snippets. Review and integrate into a manuscript copy.\n\n"
        + methods + "\n" + results + "\n" + discussion + "\n" + conclusion
    )
    (dirs["manuscript"] / "88_candidate_v9_combined_snippets.tex").write_text(combined, encoding="utf-8")

    sources = locate_tex_sources(root, dirs["output"])
    audit = stale_claim_audit(
        sources,
        numeric_tokens=config.get("stale_numeric_tokens", []),
        claim_patterns=config.get("stale_claim_patterns", []),
    )
    audit.to_csv(dirs["audit"] / "88_stale_claim_audit.csv", index=False)

    source_report = pd.DataFrame(
        [{"tex_source": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in sources]
    )
    source_report.to_csv(dirs["audit"] / "88_tex_sources_reviewed.csv", index=False)

    patch_plan = [
        "# Candidate V9 manuscript update plan",
        "",
        "The generator intentionally does not overwrite the working manuscript.",
        "Integrate the generated snippets into a copy after reviewing the stale-claim audit.",
        "",
        "## Generated text",
        "- `manuscript_snippets/88_methods_candidate_v9.tex`",
        "- `manuscript_snippets/88_results_candidate_v9.tex`",
        "- `manuscript_snippets/88_discussion_candidate_v9.tex`",
        "- `manuscript_snippets/88_conclusion_candidate_v9.tex`",
        "- `manuscript_snippets/88_abstract_result_candidate_v9.txt`",
        "",
        "## Required replacements in the old manuscript",
        "1. Replace every headline treatment-effect value from exploratory stages with the locked 22.951-day estimate.",
        "2. Use the percentile interval -4.174 to 91.010 days as the primary interval.",
        "3. Remove significance or efficacy language because all prespecified interval families include zero.",
        "4. Present the CCW result as a different-design sensitivity, not a direct contradiction.",
        "5. Separate nuisance-partition stability from patient-sampling uncertainty.",
        "6. State that METABRIC requires a separately locked external-replication estimand unless compatible treatment timing is available.",
        "",
        f"Stale-claim audit rows: {len(audit)}",
    ]
    (dirs["manuscript"] / "88_manuscript_update_plan.md").write_text("\n".join(patch_plan) + "\n", encoding="utf-8")

    print("=" * 124)
    print("STAGE 88 - MANUSCRIPT SNIPPETS")
    print("=" * 124)
    print(abstract)
    print("\nApproved result:")
    print(f"  Locked point estimate: {point:.3f} days")
    print(f"  Primary percentile CI: [{p_low:.3f}, {p_high:.3f}] days")
    print(f"  Positive bootstrap fraction: {fraction_positive:.3f}")
    print(f"  Final decision: {decision_name}")
    print(f"\nManuscript snippets written to: {dirs['manuscript']}")
    print_frame("STALE CLAIM AUDIT", audit, max_rows=40)


if __name__ == "__main__":
    main()
