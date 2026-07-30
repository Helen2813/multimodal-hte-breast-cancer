#!/usr/bin/env python3
"""Re-run Stage 41 under a narrow Python trace and capture clone-level DataFrames.

Only frames originating from the existing Stage 41/Stage 12 utility files are inspected. The
script does not modify those source files. Candidate DataFrames are written under
data/derived/stage14_trace and ranked by columns related to clone strategy, event time, and weights.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _stage14_utils import (
    dataframe_candidate_score,
    ensure_dirs,
    load_config,
    project_root,
    sha256_file,
    write_csv,
)


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    cfg = load_config(root)
    trace_dir = root / "data/derived/stage14_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        str((root / "scripts/41_replicate_estimators.py").resolve()),
        str((root / "scripts/_stage12_utils.py").resolve()),
    }
    for optional in ("_stage11_utils.py", "_stage9_utils.py"):
        path = root / "scripts" / optional
        if path.exists():
            targets.add(str(path.resolve()))

    before = {}
    for rel in (
        "results/tables/41_landmark_replication_check.csv",
        "results/tables/41_ccw_point_estimate.csv",
    ):
        path = root / rel
        if path.exists():
            before[rel] = sha256_file(path)

    captured: dict[int, dict[str, Any]] = {}

    def inspect_value(value: Any, label: str, frame_file: str, function: str) -> None:
        if isinstance(value, pd.DataFrame):
            score = dataframe_candidate_score(value)
            if len(value) < int(cfg["candidate_capture"]["minimum_rows"]) or score <= 0:
                return
            key = id(value)
            candidate = {
                "score": score,
                "rows": len(value),
                "columns": len(value.columns),
                "column_names": "|".join(map(str, value.columns)),
                "label": label,
                "source_file": Path(frame_file).name,
                "function": function,
                "dataframe": value.copy(),
            }
            previous = captured.get(key)
            if previous is None or candidate["score"] > previous["score"]:
                captured[key] = candidate
        elif isinstance(value, dict):
            for key, item in list(value.items())[:30]:
                inspect_value(item, f"{label}[{key!r}]", frame_file, function)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value[:30]):
                inspect_value(item, f"{label}[{index}]", frame_file, function)

    def trace(frame, event, arg):
        filename = str(Path(frame.f_code.co_filename).resolve())
        if event == "return" and filename in targets:
            for name, value in list(frame.f_locals.items()):
                inspect_value(value, f"local:{name}", filename, frame.f_code.co_name)
            inspect_value(arg, "return_value", filename, frame.f_code.co_name)
        return trace

    original_argv = sys.argv[:]
    exit_code = 0
    try:
        sys.argv = [str(root / "scripts/41_replicate_estimators.py")]
        sys.settrace(trace)
        try:
            runpy.run_path(
                str(root / "scripts/41_replicate_estimators.py"),
                run_name="__main__",
            )
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    finally:
        sys.settrace(None)
        sys.argv = original_argv

    if exit_code != 0:
        raise RuntimeError(f"Stage 41 traced execution failed with exit code {exit_code}")

    ranked = sorted(captured.values(), key=lambda item: (item["score"], item["rows"]), reverse=True)
    limit = int(cfg["candidate_capture"]["maximum_candidates_saved"])
    manifest_rows = []
    for index, item in enumerate(ranked[:limit], start=1):
        path = trace_dir / f"53_candidate_{index:02d}.csv"
        item["dataframe"].to_csv(path, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "rank": index,
                "score": item["score"],
                "rows": item["rows"],
                "columns": item["columns"],
                "column_names": item["column_names"],
                "label": item["label"],
                "source_file": item["source_file"],
                "function": item["function"],
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
        )

    after_rows = []
    for rel, old_hash in before.items():
        path = root / rel
        after_rows.append(
            {
                "path": rel,
                "before_sha256": old_hash,
                "after_sha256": sha256_file(path) if path.exists() else "",
                "unchanged": path.exists() and old_hash == sha256_file(path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    hashes = pd.DataFrame(after_rows)
    write_csv(manifest, root / "results/tables/53_ccw_trace_candidate_manifest.csv")
    write_csv(hashes, root / "results/tables/53_stage41_output_hash_check.csv")

    print("=" * 116)
    print("STAGE 53 — CAPTURE CCW ANALYSIS STATE")
    print("=" * 116)
    if manifest.empty:
        print("No clone-level candidate DataFrame was captured.")
        print("Inspect 53_ccw_trace_candidate_manifest.csv before proceeding.")
        return 2
    print(manifest.drop(columns=["column_names"], errors="ignore").to_string(index=False))
    print("\nStage 41 output hash check")
    print(hashes.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
