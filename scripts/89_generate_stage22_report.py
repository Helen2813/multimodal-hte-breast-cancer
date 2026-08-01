from __future__ import annotations

from pathlib import Path

import pandas as pd

from _stage22_utils import (
    ensure_dirs,
    find_root,
    load_config,
    markdown_table,
    print_frame,
    verify_manifest,
)


def main() -> None:
    root = find_root(Path.cwd())
    config = load_config(root)
    dirs = ensure_dirs(root, config)

    manifest_after = verify_manifest(root)
    manifest_after.to_csv(dirs["audit"] / "89_locked_manifest_verification_after_assets.csv", index=False)
    hashes_pass = bool(len(manifest_after) > 0 and manifest_after["match"].fillna(False).all())

    files = []
    for path in sorted(dirs["output"].rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                }
            )
    inventory = pd.DataFrame(files)
    inventory.to_csv(dirs["audit"] / "89_stage22_output_inventory.csv", index=False)

    primary = pd.read_csv(dirs["tables"] / "86_table_primary_result.csv")
    intervals = pd.read_csv(dirs["tables"] / "86_table_interval_sensitivity.csv")
    convergence = pd.read_csv(dirs["tables"] / "86_table_bootstrap_convergence.csv")
    stale = pd.read_csv(dirs["audit"] / "88_stale_claim_audit.csv") if (dirs["audit"] / "88_stale_claim_audit.csv").exists() else pd.DataFrame()

    report = [
        "# Stage 22 publication-asset report",
        "",
        "## Status",
        "",
        "`PUBLICATION_ASSETS_GENERATED_FROM_LOCKED_CANDIDATE_V9`",
        "",
        f"- Locked manifest unchanged after asset generation: **{hashes_pass}**",
        f"- Generated files: **{len(inventory)}**",
        f"- Potential stale manuscript claims flagged: **{len(stale)}**",
        "",
        "## Locked primary result",
        "",
        markdown_table(primary),
        "## Interval sensitivity",
        "",
        markdown_table(intervals),
        "## Bootstrap convergence",
        "",
        markdown_table(convergence),
        "## Main generated figures",
        "",
        "- `figures/87_bootstrap_distribution.png` and `.svg`",
        "- `figures/87_bootstrap_ecdf.png` and `.svg`",
        "- `figures/87_bootstrap_prefix_convergence.png` and `.svg`",
        "- `figures/87_inner_partition_mcse.png` and `.svg` when the MCSE field is available",
        "- `figures/87_landmark_sensitivity_forest.png` and `.svg`",
        "",
        "## Manuscript text",
        "",
        "- Methods, Results, Discussion, Conclusion, and abstract-result snippets are under `manuscript_snippets/`.",
        "- The original manuscript was not overwritten.",
        "- Review `audit/88_stale_claim_audit.csv` before integrating the snippets.",
        "",
        "## Interpretation boundary",
        "",
        "The locked point estimate is positive, but the prespecified primary patient-bootstrap interval includes zero. The final manuscript must report statistical imprecision and must not convert the high fraction of positive bootstrap repetitions into a significance claim.",
        "",
        "## Next scientific step",
        "",
        "Complete manuscript integration and then run a METABRIC data-and-design audit. Do not copy the TCGA day-180 estimand unless METABRIC contains compatible treatment-initiation timing.",
    ]
    report_text = "\n".join(report).replace("\n\n\n", "\n\n") + "\n"
    (dirs["output"] / "89_stage22_publication_assets_report.md").write_text(report_text, encoding="utf-8")

    print("=" * 124)
    print("STAGE 89 - PUBLICATION ASSET REPORT")
    print("=" * 124)
    print(report_text)
    print_frame("OUTPUT INVENTORY", inventory, max_rows=100)

    if not hashes_pass:
        raise SystemExit("Locked Candidate V9 files changed during Stage 22. Stop and inspect the manifest audit.")

    print("PASS: Candidate V9 locked files are unchanged. Publication assets are ready for manuscript integration.")


if __name__ == "__main__":
    main()
