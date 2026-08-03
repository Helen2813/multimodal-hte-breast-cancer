from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _metabric_m7_utils import fast_harrell_c_index, project_root


BOOTSTRAP_SEED = 81101


@dataclass
class Analysis:
    name: str
    frame: pd.DataFrame
    model_column: str
    clinical_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser.parse_args()


def common_repeat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"repeat", "sample_id", "time_months", "event"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"OOF file is missing columns: {missing}")

    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    repeats = sorted(frame["repeat"].astype(int).unique())
    id_sets = [
        set(frame.loc[frame["repeat"].astype(int) == repeat, "sample_id"])
        for repeat in repeats
    ]
    common = set.intersection(*id_sets)
    if not common:
        raise RuntimeError("No common patient population across repeats.")
    frame = frame.loc[frame["sample_id"].isin(common)].copy()

    counts = frame.groupby(["repeat", "sample_id"]).size()
    if int(counts.max()) != 1:
        raise RuntimeError("Expected one OOF prediction per patient per repeat.")
    per_repeat = frame.groupby("repeat")["sample_id"].nunique()
    if not bool((per_repeat == len(common)).all()):
        raise RuntimeError("Repeat populations are inconsistent.")
    return frame


def build_analyses(out: Path) -> List[Analysis]:
    full = pd.read_csv(
        out / "m52_rfs_oof_predictions_LOCAL_ONLY.csv",
        low_memory=False,
    )
    modality = pd.read_csv(
        out / "m53_rfs_oof_predictions_LOCAL_ONLY.csv",
        low_memory=False,
    )

    analyses = [
        Analysis(
            name="Multimodal",
            frame=common_repeat_frame(full),
            model_column="model_risk",
            clinical_column="clinical_risk",
        )
    ]
    for modality_name in ["RNA", "CNV", "Methylation", "Mutation"]:
        subset = modality.loc[
            modality["modality"].astype(str) == modality_name
        ].copy()
        analyses.append(
            Analysis(
                name=modality_name,
                frame=common_repeat_frame(subset),
                model_column="clinical_modality_risk",
                clinical_column="clinical_risk",
            )
        )
    return analyses


def point_estimates(analysis: Analysis) -> pd.DataFrame:
    rows = []
    for repeat, frame in analysis.frame.groupby("repeat", sort=True):
        time = frame["time_months"].to_numpy(float)
        event = frame["event"].to_numpy(int)
        clinical = fast_harrell_c_index(
            time,
            event,
            frame[analysis.clinical_column].to_numpy(float),
        )
        model = fast_harrell_c_index(
            time,
            event,
            frame[analysis.model_column].to_numpy(float),
        )
        rows.append({
            "analysis": analysis.name,
            "repeat": int(repeat),
            "n": int(len(frame)),
            "events": int(event.sum()),
            "clinical_c_index": float(clinical),
            "clinical_omics_c_index": float(model),
            "delta_c_index": float(model - clinical),
        })
    return pd.DataFrame(rows)


def lookup_by_repeat(
    analysis: Analysis,
) -> Tuple[List[int], np.ndarray, Dict[int, pd.DataFrame]]:
    repeats = sorted(analysis.frame["repeat"].astype(int).unique())
    ids = np.array(sorted(analysis.frame["sample_id"].astype(str).unique()))
    lookup = {}
    for repeat in repeats:
        lookup[repeat] = (
            analysis.frame.loc[
                analysis.frame["repeat"].astype(int) == repeat
            ]
            .set_index("sample_id")
            .loc[ids]
        )
    return repeats, ids, lookup


def bootstrap_analysis(
    analysis: Analysis,
    repetitions: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    seed: int,
) -> pd.DataFrame:
    repeats, ids, lookup = lookup_by_repeat(analysis)
    n = len(ids)

    if checkpoint_path.is_file():
        existing = pd.read_csv(checkpoint_path)
        rows = existing.to_dict("records")
        start = int(existing["bootstrap"].max()) + 1 if len(existing) else 0
        print(f"Resuming {analysis.name}: {start}/{repetitions}")
    else:
        rows = []
        start = 0

    for bootstrap_index in range(start, repetitions):
        rng = np.random.default_rng(seed + bootstrap_index)
        sampled_ids = ids[rng.integers(0, n, size=n)]

        repeat_values = []
        for repeat in repeats:
            frame = lookup[repeat].loc[sampled_ids]
            time = frame["time_months"].to_numpy(float)
            event = frame["event"].to_numpy(int)
            clinical = fast_harrell_c_index(
                time,
                event,
                frame[analysis.clinical_column].to_numpy(float),
            )
            model = fast_harrell_c_index(
                time,
                event,
                frame[analysis.model_column].to_numpy(float),
            )
            repeat_values.append((clinical, model, model - clinical))

        values = np.asarray(repeat_values, dtype=float)
        rows.append({
            "analysis": analysis.name,
            "bootstrap": bootstrap_index,
            "n": n,
            "events_in_resample": int(
                lookup[repeats[0]].loc[sampled_ids, "event"].sum()
            ),
            "repeats": len(repeats),
            "clinical_c_index": float(values[:, 0].mean()),
            "clinical_omics_c_index": float(values[:, 1].mean()),
            "delta_c_index": float(values[:, 2].mean()),
        })

        if (
            (bootstrap_index + 1) % checkpoint_every == 0
            or bootstrap_index + 1 == repetitions
        ):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(
                f"{analysis.name}: bootstrap "
                f"{bootstrap_index + 1}/{repetitions}"
            )
    return pd.DataFrame(rows)


def classify(low: float, high: float) -> str:
    if low > 0:
        return "positive incremental utility"
    if high < 0:
        return "negative incremental utility"
    return "no reliable incremental utility"


def summarize(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict:
    delta = bootstrap["delta_c_index"].to_numpy(float)
    low, high = np.quantile(delta, [0.025, 0.975])
    return {
        "analysis": str(point["analysis"].iloc[0]),
        "n": int(point["n"].iloc[0]),
        "events": int(point["events"].iloc[0]),
        "repeats": int(len(point)),
        "clinical_c_index": float(point["clinical_c_index"].mean()),
        "clinical_omics_c_index": float(
            point["clinical_omics_c_index"].mean()
        ),
        "delta_c_index": float(point["delta_c_index"].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_sd": float(delta.std(ddof=1)),
        "fraction_positive": float(np.mean(delta > 0)),
        "classification": classify(float(low), float(high)),
        "interval_interpretation": (
            "paired patient-bootstrap interval conditional on the "
            "fitted repeated RFS OOF models"
        ),
    }


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    data = summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    point = data["delta_c_index"].to_numpy(float)
    lower = point - data["ci_low"].to_numpy(float)
    upper = data["ci_high"].to_numpy(float) - point

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(
        point,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        linewidth=1.2,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, data["analysis"])
    ax.set_xlabel("Paired change in Harrell C-index (RFS)")
    ax.set_title("RFS endpoint sensitivity: incremental prognostic utility")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "figureS2_rfs_incremental_utility.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        fig_dir / "figureS2_rfs_incremental_utility.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = project_root()
    out = root / "results" / "tables" / "metabric_m11_rfs"
    fig_dir = root / "results" / "figures" / "metabric_m11_rfs"
    checkpoint_dir = out / "bootstrap_checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("METABRIC M11.54 - RFS CONDITIONAL PAIRED PATIENT BOOTSTRAP")
    print("=" * 124)
    print(f"Bootstrap repetitions: {args.bootstrap}")
    print("The primary overall-survival conclusions remain unchanged.")

    protocol_path = out / "m51_rfs_sensitivity_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("M11 RFS protocol lock is missing.")

    analyses = build_analyses(out)
    point_frames = []
    draw_frames = []
    summary_rows = []

    for index, analysis in enumerate(analyses):
        point = point_estimates(analysis)
        point_frames.append(point)
        checkpoint = (
            checkpoint_dir
            / f"{analysis.name.lower()}_rfs_bootstrap.csv"
        )
        draws = bootstrap_analysis(
            analysis=analysis,
            repetitions=args.bootstrap,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=checkpoint,
            seed=BOOTSTRAP_SEED + index * 100000,
        )
        draw_frames.append(draws)
        summary_rows.append(summarize(point, draws))

    points = pd.concat(point_frames, ignore_index=True)
    draws = pd.concat(draw_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    points.to_csv(out / "m54_rfs_repeat_level_metrics.csv", index=False)
    draws.to_csv(out / "m54_rfs_bootstrap_draws.csv", index=False)
    summary.to_csv(out / "m54_supplementary_table_s4.csv", index=False)
    plot_summary(summary, fig_dir)

    report = {
        "status": "METABRIC_M11_RFS_COMPLETE",
        "primary_os_claims_changed": False,
        "endpoint": "recurrence-free survival",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "analyses": summary.to_dict("records"),
        "outputs": [
            "m54_rfs_repeat_level_metrics.csv",
            "m54_rfs_bootstrap_draws.csv",
            "m54_supplementary_table_s4.csv",
            "figureS2_rfs_incremental_utility.png/pdf",
        ],
    }
    (out / "m54_rfs_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSUPPLEMENTARY TABLE S4")
    with pd.option_context(
        "display.max_columns", 20,
        "display.width", 180,
    ):
        print(summary.to_string(index=False))
    print("\nSTATUS: METABRIC_M11_RFS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
