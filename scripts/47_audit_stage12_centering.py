#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from _stage13_utils import (
    bootstrap_stats_from_summary_or_checkpoint,
    ensure_output_dirs,
    find_point_rows,
    load_config,
    markdown_table,
    numeric,
    project_root,
    write_csv,
    write_text,
)


def audit_row(kind: str, point: float, stats, cfg: dict) -> dict:
    b = int(stats.successful_reps)
    sd = float(stats.sd)
    bias = float(stats.mean - point)
    mcse = sd / math.sqrt(b) if b > 0 and sd > 0 else float("nan")
    z = abs(bias) / mcse if np.isfinite(mcse) and mcse > 0 else float("nan")
    min_reps = int(cfg["minimum_reps_for_centering_gate"])
    z_threshold = float(cfg["centering_z_threshold"])

    if b < min_reps:
        status = "INSUFFICIENT_REPS_FOR_CENTERING_GATE"
    elif not np.isfinite(z):
        status = "CENTERING_UNASSESSABLE"
    elif z <= z_threshold:
        status = "BOOTSTRAP_CENTERING_ACCEPTABLE"
    else:
        status = "BOOTSTRAP_CENTERING_CONCERN"

    return {
        "analysis": kind,
        "point_estimate_days": point,
        "target_reps": stats.target_reps,
        "successful_reps": b,
        "bootstrap_mean_days": stats.mean,
        "bootstrap_median_days": stats.median,
        "bootstrap_sd_days": sd,
        "bootstrap_ci_low_days": stats.ci_low,
        "bootstrap_ci_high_days": stats.ci_high,
        "fraction_positive": stats.fraction_positive,
        "mean_minus_point_days": bias,
        "monte_carlo_se_of_mean_days": mcse,
        "centering_z": z,
        "minimum_reps_required": min_reps,
        "centering_status": status,
        "source": stats.source_path,
    }


def main() -> int:
    root = project_root()
    ensure_output_dirs(root)
    cfg = load_config(root)
    tables = root / "results" / "tables"
    landmark, ccw, _, _ = find_point_rows(root)

    lm_stats = bootstrap_stats_from_summary_or_checkpoint(root, "landmark")
    ccw_stats = bootstrap_stats_from_summary_or_checkpoint(root, "ccw")
    audit = pd.DataFrame(
        [
            audit_row("landmark", numeric(landmark.get("estimate_days")), lm_stats, cfg),
            audit_row("ccw", numeric(ccw.get("estimate_days")), ccw_stats, cfg),
        ]
    )
    write_csv(audit, tables / "47_bootstrap_centering_audit.csv")
    report = f"""# Stage 12 bootstrap-centering audit

{markdown_table(audit)}

The Monte Carlo centering statistic is

`abs(bootstrap mean - point estimate) / (bootstrap SD / sqrt(B))`.

It is used only as an implementation diagnostic. A small pilot cannot provide a publication-grade
confidence interval. At least {cfg["minimum_reps_for_centering_gate"]} successful repetitions are
required before Stage 13 permits a centering decision.
"""
    write_text(report, tables / "47_bootstrap_centering_audit.md")

    print("=" * 112)
    print("STAGE 47 — BOOTSTRAP CENTERING AUDIT")
    print("=" * 112)
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
