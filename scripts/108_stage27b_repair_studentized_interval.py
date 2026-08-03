from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def root() -> Path:
    return Path.cwd().resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def show(frame: pd.DataFrame) -> str:
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 280,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        return frame.to_string(index=False)


def main() -> int:
    project = root()
    config = load_json(
        project / "stage27b_interval_repair_config.json"
    )
    source = config["source"]
    output = config["output"]
    expected = config["expected"]

    print("=" * 120)
    print("STAGE 27B - INTERVAL SUMMARY AUDIT AND REPAIR")
    print("=" * 120)
    print("No model or bootstrap estimate is rerun.")

    lock = load_json(project / source["stage27_lock"])
    if lock["bootstrap_id"] != expected["bootstrap_id"]:
        raise RuntimeError("Unexpected Stage 27 bootstrap ID.")
    if lock["v10_protocol_id"] != expected["protocol_id"]:
        raise RuntimeError("Unexpected Candidate V10 protocol ID.")

    checkpoint = pd.read_csv(
        project / source["checkpoint"],
        low_memory=False,
    )
    checkpoint = checkpoint[
        checkpoint["success"].astype(str).str.lower().eq("true")
    ].copy()
    checkpoint["repetition"] = pd.to_numeric(
        checkpoint["repetition"],
        errors="raise",
    ).astype(int)
    checkpoint = (
        checkpoint.sort_values("repetition")
        .drop_duplicates("repetition", keep="last")
    )

    if len(checkpoint) != int(expected["repetitions"]):
        raise RuntimeError(
            f"Expected {expected['repetitions']} successful repetitions, "
            f"found {len(checkpoint)}."
        )

    theta_b = pd.to_numeric(
        checkpoint["estimate_days"],
        errors="raise",
    ).to_numpy(dtype=float)
    se_b = pd.to_numeric(
        checkpoint["diagnostic_if_se_days"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(theta_b).all():
        raise RuntimeError("Non-finite bootstrap estimates found.")
    if not np.isfinite(se_b).all() or not (se_b > 0).all():
        raise RuntimeError(
            "Every bootstrap diagnostic IF SE must be finite and positive."
        )

    point_row = pd.read_csv(
        project / source["stage26_point_estimate"],
        low_memory=False,
    ).iloc[0]
    theta_hat = float(
        point_row["aipw_ato_rmst_difference_days"]
    )
    se_hat = float(point_row["diagnostic_if_se_days"])

    tolerance = float(expected["tolerance_days"])
    if abs(theta_hat - float(expected["point_estimate_days"])) > tolerance:
        raise RuntimeError("Stage 26 point estimate mismatch.")
    if not np.isfinite(se_hat) or se_hat <= 0:
        raise RuntimeError("Invalid Stage 26 diagnostic IF SE.")

    q025 = float(np.quantile(theta_b, 0.025))
    q975 = float(np.quantile(theta_b, 0.975))
    bootstrap_sd = float(np.std(theta_b, ddof=1))

    t_values = (theta_b - theta_hat) / se_b
    tq025 = float(np.quantile(t_values, 0.025))
    tq975 = float(np.quantile(t_values, 0.975))

    intervals = pd.DataFrame([
        {
            "interval": "percentile",
            "ci_low_days": q025,
            "ci_high_days": q975,
            "primary": True,
            "status": "locked_primary",
        },
        {
            "interval": "basic",
            "ci_low_days": 2.0 * theta_hat - q975,
            "ci_high_days": 2.0 * theta_hat - q025,
            "primary": False,
            "status": "locked_sensitivity",
        },
        {
            "interval": "studentized",
            "ci_low_days": theta_hat - tq975 * se_hat,
            "ci_high_days": theta_hat - tq025 * se_hat,
            "primary": False,
            "status": "locked_sensitivity",
        },
        {
            "interval": "normal_descriptive",
            "ci_low_days": theta_hat - 1.96 * bootstrap_sd,
            "ci_high_days": theta_hat + 1.96 * bootstrap_sd,
            "primary": False,
            "status": "not_locked_descriptive_only",
        },
    ])

    old_intervals = pd.read_csv(
        project / source["stage27_interval_table"],
        low_memory=False,
    )
    old_percentile = old_intervals[
        old_intervals["interval"].astype(str).eq("percentile")
    ].iloc[0]

    percentile_ok = (
        abs(q025 - float(expected["percentile_ci_low_days"])) <= tolerance
        and abs(q975 - float(expected["percentile_ci_high_days"])) <= tolerance
        and abs(q025 - float(old_percentile["ci_low_days"])) <= tolerance
        and abs(q975 - float(old_percentile["ci_high_days"])) <= tolerance
    )
    if not percentile_ok:
        raise RuntimeError(
            "Primary percentile interval did not reproduce exactly."
        )

    out_table = project / output["corrected_interval_table"]
    out_table.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(
        out_table,
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "status": "STAGE27_INTERVAL_SUMMARY_REPAIRED",
        "protocol_id": expected["protocol_id"],
        "bootstrap_id": expected["bootstrap_id"],
        "bootstrap_repetitions": len(theta_b),
        "point_estimate_days": theta_hat,
        "stage26_diagnostic_if_se_days": se_hat,
        "primary_percentile_interval_reproduced": True,
        "studentized_t_quantiles": {
            "q025": tq025,
            "q975": tq975,
        },
        "bootstrap_se_distribution": {
            "minimum": float(np.min(se_b)),
            "median": float(np.median(se_b)),
            "maximum": float(np.max(se_b)),
        },
        "corrected_intervals": intervals.to_dict("records"),
        "identified_issue": (
            "Stage 27 locked a studentized sensitivity interval, "
            "but Stage 107 output a normal interval instead."
        ),
        "repair": (
            "The studentized interval was computed from the existing "
            "300 locked bootstrap estimates and replicate diagnostic IF SEs. "
            "No model or bootstrap estimate was rerun."
        ),
        "boundary": config["boundary"],
    }
    save_json(audit, project / output["audit"])

    print("Corrected interval table")
    print(show(intervals))
    print("\nAudit")
    print(json.dumps(audit, indent=2))
    print(
        "\nPASS: primary percentile interval reproduced and "
        "studentized sensitivity interval added."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
