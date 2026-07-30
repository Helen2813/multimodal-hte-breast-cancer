#!/usr/bin/env python3
"""Extend the existing Stage 12 checkpoints to a moderate centering pilot.

Stage 13 v2 fix:
The Stage 12 bootstrap scripts may leave a zero-byte or whitespace-only error CSV
after a successful run with no errors. Pandas raises EmptyDataError when that file
is read on resume. Before calling the original Stage 12 scripts, this module safely
backs up and removes only structurally empty error/failure CSV files associated
with Stages 42 and 43.

The Stage 12 bootstrap scripts and checkpoints are otherwise left unchanged.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from _stage13_utils import (
    bootstrap_stats_from_summary_or_checkpoint,
    discover_bootstrap_env_names,
    ensure_output_dirs,
    load_config,
    project_root,
    run_python_script,
    write_csv,
)


STAGE_BOOTSTRAP_SCRIPTS = {
    "landmark": "scripts/42_landmark_full_pipeline_bootstrap.py",
    "ccw": "scripts/43_ccw_sensitivity_bootstrap.py",
}


def _quoted_error_csv_names(script_path: Path) -> set[str]:
    """Extract quoted CSV file names containing error/fail tokens from a script."""
    if not script_path.exists():
        return set()
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    names = set()
    for match in re.finditer(r"""["']([^"']+\.csv)["']""", text, flags=re.IGNORECASE):
        name = Path(match.group(1)).name
        lower = name.lower()
        if any(token in lower for token in ("error", "errors", "fail", "failure")):
            names.add(name)
    return names


def _candidate_error_csvs(root: Path) -> list[Path]:
    """Find only Stage 42/43 error-like CSVs that can trigger EmptyDataError."""
    tables = root / "results" / "tables"
    candidates: set[Path] = set()

    for rel in STAGE_BOOTSTRAP_SCRIPTS.values():
        script_path = root / rel
        for name in _quoted_error_csv_names(script_path):
            candidates.add(tables / name)

    if tables.exists():
        for path in tables.glob("*.csv"):
            lower = path.name.lower()
            is_stage_bootstrap = lower.startswith("42") or lower.startswith("43")
            is_error_like = any(
                token in lower for token in ("error", "errors", "fail", "failure")
            )
            if is_stage_bootstrap and is_error_like:
                candidates.add(path)

    return sorted(candidates)


def _is_structurally_empty_csv(path: Path) -> bool:
    """Return True only when a CSV contains no parseable header or columns."""
    if not path.exists() or not path.is_file():
        return False

    # Zero bytes or whitespace-only content always causes EmptyDataError.
    raw = path.read_bytes()
    if len(raw) == 0 or raw.strip() == b"":
        return True

    try:
        pd.read_csv(path, nrows=1)
    except EmptyDataError:
        return True
    except Exception:
        # Do not delete malformed-but-nonempty evidence automatically.
        return False
    return False


def sanitize_empty_bootstrap_error_csvs(root: Path) -> pd.DataFrame:
    """Back up and remove only structurally empty Stage 42/43 error CSVs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "results" / "logs" / "stage13_empty_csv_backups" / stamp
    rows: list[dict[str, object]] = []

    for path in _candidate_error_csvs(root):
        exists = path.exists()
        structurally_empty = _is_structurally_empty_csv(path) if exists else False
        backup_path = ""

        if exists and structurally_empty:
            backup_dir.mkdir(parents=True, exist_ok=True)
            destination = backup_dir / path.name
            shutil.copy2(path, destination)
            path.unlink()
            backup_path = str(destination.relative_to(root))
            action = "BACKED_UP_AND_REMOVED"
        elif exists:
            action = "PRESERVED_NONEMPTY"
        else:
            action = "NOT_PRESENT"

        rows.append(
            {
                "path": str(path.relative_to(root)),
                "existed": exists,
                "structurally_empty": structurally_empty,
                "action": action,
                "backup_path": backup_path,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "path",
            "existed",
            "structurally_empty",
            "action",
            "backup_path",
        ],
    )


def _run_with_one_empty_csv_retry(
    root: Path,
    relative_script: str,
    env: dict[str, str],
) -> tuple[str, pd.DataFrame]:
    """Run a Stage 12 bootstrap script, retrying once after empty-CSV cleanup."""
    cleanup_before = sanitize_empty_bootstrap_error_csvs(root)
    try:
        run_python_script(root, relative_script, env)
        return "COMPLETED", cleanup_before
    except RuntimeError:
        cleanup_after_failure = sanitize_empty_bootstrap_error_csvs(root)
        removed_after_failure = (
            not cleanup_after_failure.empty
            and (cleanup_after_failure["action"] == "BACKED_UP_AND_REMOVED").any()
        )
        cleanup = pd.concat(
            [
                cleanup_before.assign(cleanup_phase="before_first_attempt"),
                cleanup_after_failure.assign(cleanup_phase="after_failed_attempt"),
            ],
            ignore_index=True,
        )
        if not removed_after_failure:
            raise
        run_python_script(root, relative_script, env)
        return "COMPLETED_AFTER_EMPTY_CSV_RETRY", cleanup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--skip-landmark", action="store_true")
    parser.add_argument("--skip-ccw", action="store_true")
    args = parser.parse_args()

    root = project_root()
    ensure_output_dirs(root)
    cfg = load_config(root)
    target = int(args.target or cfg["centering_pilot_target_reps"])
    tables = root / "results" / "tables"

    before_lm = bootstrap_stats_from_summary_or_checkpoint(root, "landmark")
    before_ccw = bootstrap_stats_from_summary_or_checkpoint(root, "ccw")
    env_names = discover_bootstrap_env_names(root)

    rows: list[dict[str, object]] = []
    cleanup_frames: list[pd.DataFrame] = []
    env = os.environ.copy()
    interface_ok = bool(env_names["landmark"] and env_names["ccw"])

    if interface_ok:
        env[str(env_names["landmark"])] = str(target)
        env[str(env_names["ccw"])] = str(target)

    if not interface_ok:
        rows.append(
            {
                "analysis": "both",
                "target_reps": target,
                "before_successful_reps": "",
                "after_successful_reps": "",
                "environment_variable": "",
                "status": "SAFE_INTERFACE_NOT_DISCOVERED_NO_SCRIPTS_RUN",
            }
        )
    else:
        if not args.skip_landmark and before_lm.successful_reps < target:
            status, cleanup = _run_with_one_empty_csv_retry(
                root,
                STAGE_BOOTSTRAP_SCRIPTS["landmark"],
                env,
            )
            if not cleanup.empty:
                cleanup_frames.append(cleanup.assign(analysis="landmark"))
        else:
            status = "ALREADY_COMPLETE_OR_SKIPPED"

        after_lm = bootstrap_stats_from_summary_or_checkpoint(root, "landmark")
        rows.append(
            {
                "analysis": "landmark",
                "target_reps": target,
                "before_successful_reps": before_lm.successful_reps,
                "after_successful_reps": after_lm.successful_reps,
                "environment_variable": env_names["landmark"],
                "status": status,
            }
        )

        if not args.skip_ccw and before_ccw.successful_reps < target:
            status, cleanup = _run_with_one_empty_csv_retry(
                root,
                STAGE_BOOTSTRAP_SCRIPTS["ccw"],
                env,
            )
            if not cleanup.empty:
                cleanup_frames.append(cleanup.assign(analysis="ccw"))
        else:
            status = "ALREADY_COMPLETE_OR_SKIPPED"

        after_ccw = bootstrap_stats_from_summary_or_checkpoint(root, "ccw")
        rows.append(
            {
                "analysis": "ccw",
                "target_reps": target,
                "before_successful_reps": before_ccw.successful_reps,
                "after_successful_reps": after_ccw.successful_reps,
                "environment_variable": env_names["ccw"],
                "status": status,
            }
        )

    result = pd.DataFrame(rows)
    write_csv(result, tables / "48_centering_pilot_extension.csv")

    cleanup_result = (
        pd.concat(cleanup_frames, ignore_index=True)
        if cleanup_frames
        else pd.DataFrame(
            columns=[
                "path",
                "existed",
                "structurally_empty",
                "action",
                "backup_path",
                "analysis",
            ]
        )
    )
    write_csv(cleanup_result, tables / "48_empty_error_csv_cleanup.csv")

    print("=" * 112)
    print("STAGE 48 v2 — EXTEND CHECKPOINTED CENTERING PILOT")
    print("=" * 112)
    print(f"Discovered environment variables: {env_names}")
    print("\nEmpty error-CSV cleanup")
    print(
        cleanup_result.to_string(index=False)
        if not cleanup_result.empty
        else "No candidate empty error CSV required cleanup."
    )
    print("\nPilot extension")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
