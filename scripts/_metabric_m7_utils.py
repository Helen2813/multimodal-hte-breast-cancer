from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import roc_auc_score


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m7_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def out_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(
    path: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_csv(temp, rows, fieldnames)
    os.replace(temp, path)


def write_csv(
    path: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = list(fieldnames or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
        return
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def print_table(
    rows: Sequence[dict],
    columns: Sequence[str],
    max_rows: int | None = None,
) -> None:
    if not rows:
        print("<empty>")
        return
    visible = list(rows if max_rows is None else rows[:max_rows])
    widths = {column: len(column) for column in columns}
    rendered = []
    for row in visible:
        item = {column: str(row.get(column, "")) for column in columns}
        rendered.append(item)
        for column in columns:
            widths[column] = min(
                58,
                max(widths[column], len(item[column])),
            )
    print(
        "  ".join(
            column[:widths[column]].ljust(widths[column])
            for column in columns
        )
    )
    print(
        "  ".join("-" * widths[column] for column in columns)
    )
    for row in rendered:
        print(
            "  ".join(
                row[column][:widths[column]].ljust(widths[column])
                for column in columns
            )
        )
    if max_rows is not None and len(rows) > max_rows:
        print(
            f"... {len(rows) - max_rows} additional rows written to CSV"
        )


def normalize_id(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\.0$", "", text)
    return text


def normalize_tcga_id(value: object) -> str:
    text = normalize_id(value).replace("_", "-")
    match = re.search(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
        text,
    )
    return match.group(1) if match else text


def candidate_id_columns(
    frame: pd.DataFrame,
    maximum: int = 12,
) -> list[str]:
    candidates = []
    for position, column in enumerate(frame.columns):
        normalized = re.sub(
            r"[^A-Z0-9]+",
            "_",
            str(column).upper(),
        ).strip("_")
        score = 0.0
        if normalized in {
            "SAMPLE",
            "SAMPLE_ID",
            "PATIENT_ID",
            "CASE_ID",
            "SUBMITTER_ID",
            "UNNAMED_0",
        }:
            score += 120
        if any(
            token in normalized
            for token in (
                "SAMPLE",
                "PATIENT",
                "CASE",
                "SUBMITTER",
            )
        ):
            score += 50
        if normalized.startswith("UNNAMED"):
            score += 30
        sample = frame[column].dropna().astype(str).head(300)
        if len(sample):
            tcga_fraction = sample.str.contains(
                r"TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}",
                case=False,
                regex=True,
            ).mean()
            score += 150 * float(tcga_fraction)
            score += 25 * (
                sample.nunique() / max(1, len(sample))
            )
        score -= 0.001 * position
        candidates.append((score, column))
    candidates.sort(
        key=lambda item: (-item[0], str(item[1]))
    )
    return [column for _, column in candidates[:maximum]]


def resolve_tcga_id_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    minimum_overlap: int,
) -> tuple[str, str, list[dict]]:
    rows = []
    for left_column in candidate_id_columns(left):
        left_ids = {
            normalize_tcga_id(value)
            for value in left[left_column].dropna().astype(str)
        }
        for right_column in candidate_id_columns(right):
            right_ids = {
                normalize_tcga_id(value)
                for value in right[right_column].dropna().astype(str)
            }
            overlap = len(left_ids & right_ids)
            rows.append({
                "left_column": left_column,
                "right_column": right_column,
                "overlap": overlap,
                "left_unique": len(left_ids),
                "right_unique": len(right_ids),
                "left_coverage": (
                    overlap / len(left_ids) if left_ids else 0.0
                ),
                "right_coverage": (
                    overlap / len(right_ids) if right_ids else 0.0
                ),
            })
    rows.sort(
        key=lambda row: (
            -int(row["overlap"]),
            -float(row["left_coverage"]),
            -float(row["right_coverage"]),
            row["left_column"],
            row["right_column"],
        )
    )
    if not rows or int(rows[0]["overlap"]) < minimum_overlap:
        raise RuntimeError(
            f"Could not resolve TCGA identifiers. Best pair: "
            f"{rows[0] if rows else None}"
        )
    return (
        str(rows[0]["left_column"]),
        str(rows[0]["right_column"]),
        rows,
    )


def exact_tcga_stage_indicator(
    frame: pd.DataFrame,
    roman_stage: str,
) -> pd.Series:
    base = r"^CLIN_ajcc_pathologic_stage\.diagnoses_Stage "
    if roman_stage == "II":
        pattern = re.compile(
            base + r"II(?:A|B|C)?$",
            re.IGNORECASE,
        )
    elif roman_stage == "III":
        pattern = re.compile(
            base + r"III(?:A|B|C)?$",
            re.IGNORECASE,
        )
    elif roman_stage == "IV":
        pattern = re.compile(base + r"IV$", re.IGNORECASE)
    else:
        raise ValueError(roman_stage)
    columns = [
        column
        for column in frame.columns
        if pattern.match(str(column))
    ]
    if not columns:
        return pd.Series(0.0, index=frame.index)
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .astype(float)
    )


def tcga_node_positive_indicator(
    frame: pd.DataFrame,
) -> pd.Series:
    pattern = re.compile(
        r"^CLIN_ajcc_pathologic_n\.diagnoses_N(?:1|2|3)",
        re.IGNORECASE,
    )
    columns = [
        column
        for column in frame.columns
        if pattern.match(str(column))
    ]
    if not columns:
        return pd.Series(0.0, index=frame.index)
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .astype(float)
    )


def build_tcga_clinical(frame: pd.DataFrame) -> pd.DataFrame:
    age_column = next(
        (
            column
            for column in frame.columns
            if str(column).startswith("CLIN_age_at_index")
        ),
        None,
    )
    result = pd.DataFrame(index=frame.index)
    result["age"] = (
        pd.to_numeric(frame[age_column], errors="coerce")
        if age_column
        else np.nan
    )
    result["stage_II"] = exact_tcga_stage_indicator(
        frame,
        "II",
    )
    result["stage_III"] = exact_tcga_stage_indicator(
        frame,
        "III",
    )
    result["stage_IV"] = exact_tcga_stage_indicator(
        frame,
        "IV",
    )
    result["node_positive"] = tcga_node_positive_indicator(
        frame
    )
    return result


def build_metabric_clinical(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    stage = pd.to_numeric(frame["stage"], errors="coerce")
    nodes = pd.to_numeric(
        frame["positive_nodes"],
        errors="coerce",
    )
    result = pd.DataFrame(index=frame.index)
    result["age"] = pd.to_numeric(
        frame["age_at_diagnosis"],
        errors="coerce",
    )
    result["stage_II"] = (stage == 2).astype(float)
    result["stage_III"] = (stage == 3).astype(float)
    result["stage_IV"] = (stage == 4).astype(float)
    result["node_positive"] = (nodes > 0).astype(float)
    result.loc[nodes.isna(), "node_positive"] = np.nan
    return result


def median_impute_scale_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_numeric = train.apply(
        pd.to_numeric,
        errors="coerce",
    ).copy()
    test_numeric = test.apply(
        pd.to_numeric,
        errors="coerce",
    ).copy()
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_numeric = train_numeric.fillna(medians)
    test_numeric = test_numeric.fillna(medians)
    means = train_numeric.mean(axis=0)
    sds = (
        train_numeric.std(axis=0, ddof=0)
        .replace(0, 1.0)
        .fillna(1.0)
    )
    if scale:
        train_numeric = (train_numeric - means) / sds
        test_numeric = (test_numeric - means) / sds
    return train_numeric, test_numeric, {
        "medians": {
            str(key): float(value)
            for key, value in medians.items()
        },
        "means": {
            str(key): float(value)
            for key, value in means.items()
        },
        "sds": {
            str(key): float(value)
            for key, value in sds.items()
        },
    }


def rank_normalize_separately(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    n = len(frame)
    for column in frame.columns:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        median = values.median()
        values = values.fillna(
            median if np.isfinite(median) else 0.0
        )
        ranks = values.rank(method="average")
        probabilities = (ranks - 0.5) / n
        result[column] = stats.norm.ppf(
            np.clip(probabilities, 1e-6, 1 - 1e-6)
        )
    return result


def fast_harrell_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    valid = (
        np.isfinite(time)
        & np.isfinite(event)
        & np.isfinite(risk)
    )
    if valid.sum() < 3 or event[valid].sum() == 0:
        return float("nan")
    return float(
        concordance_index(
            time[valid],
            -risk[valid],
            event_observed=event[valid],
        )
    )


def binary_auc_at_horizon(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
) -> tuple[float, int]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    valid = cases | controls
    if (
        valid.sum() < 10
        or cases.sum() == 0
        or controls.sum() == 0
    ):
        return float("nan"), int(valid.sum())
    return (
        float(
            roc_auc_score(
                cases[valid].astype(int),
                risk[valid],
            )
        ),
        int(valid.sum()),
    )


def fit_external_cox(
    train_x: pd.DataFrame,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_x: pd.DataFrame,
    penalizer: float,
    prediction_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train = train_x.copy()
    train["__time"] = np.asarray(
        train_time,
        dtype=float,
    )
    train["__event"] = np.asarray(
        train_event,
        dtype=int,
    )
    model = CoxPHFitter(penalizer=float(penalizer))
    model.fit(
        train,
        duration_col="__time",
        event_col="__event",
        show_progress=False,
    )
    risk = (
        model.predict_partial_hazard(test_x)
        .to_numpy(dtype=float)
        .ravel()
    )
    survival = (
        model.predict_survival_function(
            test_x,
            times=prediction_times,
        )
        .T.to_numpy(dtype=float)
    )
    return risk, survival, {
        "coefficients": {
            str(key): float(value)
            for key, value in model.params_.to_dict().items()
        },
        "log_likelihood": float(model.log_likelihood_),
        "concordance_train": float(
            model.concordance_index_
        ),
        "baseline_survival_times": [
            float(value)
            for value in model.baseline_survival_.index
        ],
        "baseline_survival_values": [
            float(value)
            for value in model.baseline_survival_.iloc[:, 0]
        ],
    }


def infer_time_unit_and_convert_to_months(
    time: np.ndarray,
) -> tuple[np.ndarray, dict]:
    values = np.asarray(time, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise RuntimeError("Empty survival time vector.")
    median = float(np.median(finite))
    maximum = float(np.max(finite))
    if median > 300 or maximum > 1000:
        converted = values / 30.4375
        unit = "days_converted_to_months"
        divisor = 30.4375
    else:
        converted = values.copy()
        unit = "months"
        divisor = 1.0
    return converted, {
        "detected_unit": unit,
        "conversion_divisor": divisor,
        "original_median": median,
        "original_maximum": maximum,
        "converted_median_months": float(
            np.nanmedian(converted)
        ),
        "converted_maximum_months": float(
            np.nanmax(converted)
        ),
    }


def km_survival_at(
    time: np.ndarray,
    event: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    model = KaplanMeierFitter()
    model.fit(
        np.asarray(time, dtype=float),
        event_observed=np.asarray(event, dtype=int),
    )
    return model.survival_function_at_times(
        query_times
    ).to_numpy(dtype=float)


def censoring_survival_function(
    time: np.ndarray,
    event: np.ndarray,
):
    model = KaplanMeierFitter()
    model.fit(
        np.asarray(time, dtype=float),
        event_observed=1 - np.asarray(event, dtype=int),
    )
    timeline = model.survival_function_.index.to_numpy(
        dtype=float
    )
    values = model.survival_function_.iloc[:, 0].to_numpy(
        dtype=float
    )

    def evaluate(
        query: np.ndarray | float,
        left_limit: bool = False,
    ) -> np.ndarray:
        queries = np.atleast_1d(query).astype(float)
        side = "left" if left_limit else "right"
        indices = np.searchsorted(
            timeline,
            queries,
            side=side,
        ) - 1
        result = np.ones(len(queries), dtype=float)
        valid = indices >= 0
        result[valid] = values[indices[valid]]
        return np.clip(result, 1e-6, 1.0)

    return evaluate


def uno_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    tau: float,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    g = censoring_survival_function(time, event)
    concordant = 0.0
    denominator = 0.0
    for index in range(len(time)):
        if event[index] != 1 or time[index] > tau:
            continue
        comparable = time > time[index]
        if not np.any(comparable):
            continue
        weight = 1.0 / float(
            g(np.array([time[index]]), left_limit=True)[0]
        ) ** 2
        denominator += weight * float(np.sum(comparable))
        concordant += weight * (
            float(np.sum(risk[index] > risk[comparable]))
            + 0.5
            * float(
                np.sum(risk[index] == risk[comparable])
            )
        )
    return (
        concordant / denominator
        if denominator > 0
        else float("nan")
    )


def ipcw_dynamic_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    cases = np.where((event == 1) & (time <= horizon))[0]
    controls = np.where(time > horizon)[0]
    if len(cases) == 0 or len(controls) == 0:
        return float("nan")
    g = censoring_survival_function(time, event)
    case_weights = 1.0 / g(
        time[cases],
        left_limit=True,
    )
    control_weight = 1.0 / float(
        g(np.array([horizon]))[0]
    )
    numerator = 0.0
    denominator = 0.0
    control_risk = risk[controls]
    for case_index, weight in zip(cases, case_weights):
        numerator += weight * control_weight * (
            float(np.sum(risk[case_index] > control_risk))
            + 0.5
            * float(
                np.sum(risk[case_index] == control_risk)
            )
        )
        denominator += (
            weight * control_weight * len(controls)
        )
    return (
        numerator / denominator
        if denominator > 0
        else float("nan")
    )


def ipcw_brier_score(
    time: np.ndarray,
    event: np.ndarray,
    predicted_survival: np.ndarray,
    horizon: float,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    predicted_survival = np.asarray(
        predicted_survival,
        dtype=float,
    )
    g = censoring_survival_function(time, event)
    values = np.zeros(len(time), dtype=float)
    event_before = (time <= horizon) & (event == 1)
    survived = time > horizon
    if np.any(event_before):
        weights = 1.0 / g(
            time[event_before],
            left_limit=True,
        )
        values[event_before] = (
            weights
            * predicted_survival[event_before] ** 2
        )
    if np.any(survived):
        weight = 1.0 / float(
            g(np.array([horizon]))[0]
        )
        values[survived] = (
            weight
            * (1.0 - predicted_survival[survived]) ** 2
        )
    return float(np.mean(values))


def integrated_brier_score(
    time: np.ndarray,
    event: np.ndarray,
    predicted_survival: np.ndarray,
    grid: np.ndarray,
) -> float:
    scores = np.asarray(
        [
            ipcw_brier_score(
                time,
                event,
                predicted_survival[:, index],
                float(horizon),
            )
            for index, horizon in enumerate(grid)
        ],
        dtype=float,
    )
    return float(
        np.trapz(scores, grid)
        / (float(grid[-1]) - float(grid[0]))
    )


def external_calibration_slope(
    time: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
) -> float:
    frame = pd.DataFrame({
        "time": np.asarray(time, dtype=float),
        "event": np.asarray(event, dtype=int),
        "log_risk": np.asarray(log_risk, dtype=float),
    })
    try:
        model = CoxPHFitter(penalizer=0.0)
        model.fit(
            frame,
            duration_col="time",
            event_col="event",
            show_progress=False,
        )
        return float(model.params_["log_risk"])
    except Exception:
        return float("nan")


def standardize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    medians = np.nanmedian(x, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    x = x.copy()
    missing = np.where(~np.isfinite(x))
    x[missing] = medians[missing[1]]
    means = x.mean(axis=0)
    sds = x.std(axis=0)
    sds[sds < 1e-12] = 1.0
    return (x - means) / sds


def rank_gaussian_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    output = np.empty_like(x, dtype=float)
    n = len(x)
    for column in range(x.shape[1]):
        ranks = stats.rankdata(
            x[:, column],
            method="average",
        )
        probabilities = (ranks - 0.5) / n
        output[:, column] = stats.norm.ppf(
            np.clip(probabilities, 1e-6, 1 - 1e-6)
        )
    return output


def transform_for_ci(
    x: np.ndarray,
    y: np.ndarray,
    engine: str,
) -> tuple[np.ndarray, np.ndarray]:
    if engine == "rank_gaussian_partial":
        return (
            rank_gaussian_matrix(x),
            rank_gaussian_matrix(
                np.asarray(y).reshape(-1, 1)
            ).ravel(),
        )
    if engine == "spearman_partial":
        return (
            np.column_stack([
                stats.rankdata(
                    x[:, column],
                    method="average",
                )
                for column in range(x.shape[1])
            ]),
            stats.rankdata(y, method="average"),
        )
    if engine == "pearson_partial":
        return np.asarray(x, dtype=float), np.asarray(
            y,
            dtype=float,
        )
    raise ValueError(engine)


def residualize_vector(
    y: np.ndarray,
    z: np.ndarray | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if z is None or z.size == 0:
        return y - np.mean(y)
    design = np.column_stack(
        [np.ones(len(y)), np.asarray(z, dtype=float)]
    )
    coefficients, *_ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    return y - design @ coefficients


def residualize_matrix(
    x: np.ndarray,
    z: np.ndarray | None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if z is None or z.size == 0:
        return x - x.mean(axis=0, keepdims=True)
    design = np.column_stack(
        [np.ones(len(x)), np.asarray(z, dtype=float)]
    )
    coefficients, *_ = np.linalg.lstsq(
        design,
        x,
        rcond=None,
    )
    return x - design @ coefficients


def partial_pvalues(
    x: np.ndarray,
    y: np.ndarray,
    conditioning_indices: Sequence[int],
) -> np.ndarray:
    z = (
        x[:, list(conditioning_indices)]
        if conditioning_indices
        else None
    )
    residual_y = residualize_vector(y, z)
    residual_x = residualize_matrix(x, z)
    numerator = residual_x.T @ residual_y
    denominator = np.sqrt(
        np.sum(residual_x * residual_x, axis=0)
        * np.sum(residual_y * residual_y)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-15,
    )
    correlations = np.clip(
        correlations,
        -0.999999,
        0.999999,
    )
    degrees = max(
        1,
        len(y) - len(conditioning_indices) - 2,
    )
    statistic = np.abs(correlations) * np.sqrt(
        degrees
        / np.maximum(
            1e-12,
            1.0 - correlations * correlations,
        )
    )
    return 2.0 * stats.t.sf(statistic, degrees)


def iamb_select(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_selected: int,
) -> list[int]:
    selected: list[int] = []
    while len(selected) < min(
        max_selected,
        x.shape[1],
    ):
        pvalues = partial_pvalues(x, y, selected)
        if selected:
            pvalues[selected] = 1.0
        best = int(np.nanargmin(pvalues))
        if (
            not np.isfinite(pvalues[best])
            or pvalues[best] >= alpha
        ):
            break
        selected.append(best)

    changed = True
    while changed and selected:
        changed = False
        for feature in list(selected):
            conditioning = [
                index
                for index in selected
                if index != feature
            ]
            local_x = x[:, [feature] + conditioning]
            local_conditioning = list(
                range(1, 1 + len(conditioning))
            )
            pvalue = partial_pvalues(
                local_x,
                y,
                local_conditioning,
            )[0]
            if not np.isfinite(pvalue) or pvalue >= alpha:
                selected.remove(feature)
                changed = True
    return selected


def pad_selection(
    selected: Sequence[int],
    x: np.ndarray,
    y: np.ndarray,
    target: int,
) -> list[int]:
    selected = list(
        dict.fromkeys(int(index) for index in selected)
    )
    if len(selected) >= target:
        return selected
    scores = []
    for column in range(x.shape[1]):
        if column in selected:
            continue
        rho = stats.spearmanr(
            x[:, column],
            y,
            nan_policy="omit",
        ).statistic
        scores.append((
            abs(float(rho))
            if np.isfinite(rho)
            else 0.0,
            column,
        ))
    scores.sort(key=lambda item: (-item[0], item[1]))
    for _, column in scores:
        selected.append(column)
        if len(selected) >= min(target, x.shape[1]):
            break
    return selected


def top_signed_spearman_chunked(
    frame: pd.DataFrame,
    train_positions: Sequence[int],
    outcome: np.ndarray,
    positive: int,
    negative: int,
    chunk_size: int = 512,
) -> tuple[list[str], list[dict]]:
    y = np.asarray(outcome, dtype=float)
    y_rank = stats.rankdata(
        y,
        method="average",
    ).astype(np.float64)
    y_centered = y_rank - y_rank.mean()
    y_norm = np.sqrt(
        np.sum(y_centered * y_centered)
    )
    rows = []
    positions = np.asarray(
        train_positions,
        dtype=int,
    )
    columns = list(frame.columns)

    for start in range(0, len(columns), chunk_size):
        chunk_columns = columns[
            start:start + chunk_size
        ]
        chunk = (
            frame.iloc[positions][chunk_columns]
            .apply(pd.to_numeric, errors="coerce")
        )
        medians = chunk.median(axis=0).fillna(0.0)
        ranks = (
            chunk.fillna(medians)
            .rank(axis=0, method="average")
            .to_numpy(dtype=np.float64)
        )
        ranks -= ranks.mean(axis=0, keepdims=True)
        denominator = (
            np.sqrt(np.sum(ranks * ranks, axis=0))
            * y_norm
        )
        correlations = np.divide(
            ranks.T @ y_centered,
            denominator,
            out=np.zeros(len(chunk_columns)),
            where=denominator > 1e-15,
        )
        for column, correlation in zip(
            chunk_columns,
            correlations,
        ):
            rows.append({
                "feature": column,
                "spearman_rho": float(correlation),
            })

    positive_rows = [
        row
        for row in sorted(
            rows,
            key=lambda row: (
                -row["spearman_rho"],
                row["feature"],
            ),
        )
        if row["spearman_rho"] > 0
    ][:positive]
    negative_rows = [
        row
        for row in sorted(
            rows,
            key=lambda row: (
                row["spearman_rho"],
                row["feature"],
            ),
        )
        if row["spearman_rho"] < 0
    ][:negative]
    selected = []
    for row in positive_rows + negative_rows:
        if row["feature"] not in selected:
            selected.append(row["feature"])
    return selected, rows


def jaccard(
    first: Iterable[str],
    second: Iterable[str],
) -> float:
    first_set = set(first)
    second_set = set(second)
    denominator = len(first_set | second_set)
    return (
        len(first_set & second_set) / denominator
        if denominator
        else 1.0
    )


def overlap_coefficient(
    first: Iterable[str],
    second: Iterable[str],
) -> float:
    first_set = set(first)
    second_set = set(second)
    denominator = min(
        len(first_set),
        len(second_set),
    )
    return (
        len(first_set & second_set) / denominator
        if denominator
        else 1.0
    )
