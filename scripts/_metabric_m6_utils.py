from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    path = root / "metabric_m6_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def out_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def print_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> None:
    if not rows:
        print("<empty>")
        return
    show = list(rows if max_rows is None else rows[:max_rows])
    widths = {column: len(column) for column in columns}
    rendered = []
    for row in show:
        item = {column: str(row.get(column, "")) for column in columns}
        rendered.append(item)
        for column in columns:
            widths[column] = min(58, max(widths[column], len(item[column])))
    print("  ".join(column[:widths[column]].ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rendered:
        print("  ".join(row[column][:widths[column]].ljust(widths[column]) for column in columns))
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} additional rows written to CSV")


def normalize_id(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\.0$", "", text)
    return text


def normalize_tcga_id(value: object) -> str:
    text = normalize_id(value).replace("_", "-")
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", text)
    return match.group(1) if match else text


def guess_id_column(frame: pd.DataFrame) -> str:
    candidates = []
    for column in frame.columns:
        n = norm(column)
        score = 0
        if n in {"SAMPLE", "SAMPLE_ID", "PATIENT_ID", "CASE_ID", "UNNAMED_0"}:
            score += 100
        if "SAMPLE" in n or "PATIENT" in n or "CASE" in n:
            score += 40
        unique_fraction = frame[column].astype(str).nunique() / max(1, len(frame))
        score += int(20 * unique_fraction)
        candidates.append((score, column))
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    return candidates[0][1]


def read_feature_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    values = []
    for token in re.split(r"[\s,;]+", text):
        token = token.strip().strip('"').strip("'")
        if token and token.lower() not in {"gene", "genes", "feature", "features"}:
            if token not in values:
                values.append(token)
    return values


def rank_gaussian_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    n, p = x.shape
    for j in range(p):
        column = x[:, j]
        ranks = stats.rankdata(column, method="average")
        probabilities = (ranks - 0.5) / n
        out[:, j] = stats.norm.ppf(np.clip(probabilities, 1e-6, 1 - 1e-6))
    return out


def transform_for_ci(x: np.ndarray, y: np.ndarray, engine: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if engine == "rank_gaussian_partial":
        return rank_gaussian_matrix(x), rank_gaussian_matrix(y.reshape(-1, 1)).ravel()
    if engine == "spearman_partial":
        xr = np.column_stack([stats.rankdata(x[:, j], method="average") for j in range(x.shape[1])])
        yr = stats.rankdata(y, method="average")
        return xr, yr
    if engine == "pearson_partial":
        return x, y
    raise ValueError(f"Unknown CI engine: {engine}")


def standardize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    medians = np.nanmedian(x, axis=0)
    bad = ~np.isfinite(medians)
    medians[bad] = 0.0
    inds = np.where(~np.isfinite(x))
    x = x.copy()
    x[inds] = medians[inds[1]]
    means = x.mean(axis=0)
    sds = x.std(axis=0)
    sds[sds < 1e-12] = 1.0
    return (x - means) / sds


def residualize_vector(y: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if z is None or z.size == 0:
        return y - np.mean(y)
    z = np.asarray(z, dtype=float)
    design = np.column_stack([np.ones(len(y)), z])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def residualize_matrix(x: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if z is None or z.size == 0:
        return x - x.mean(axis=0, keepdims=True)
    z = np.asarray(z, dtype=float)
    design = np.column_stack([np.ones(len(x)), z])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def partial_pvalues(x: np.ndarray, y: np.ndarray, conditioning_indices: Sequence[int]) -> np.ndarray:
    z = x[:, list(conditioning_indices)] if conditioning_indices else None
    ry = residualize_vector(y, z)
    rx = residualize_matrix(x, z)
    numerator = rx.T @ ry
    denom = np.sqrt(np.sum(rx * rx, axis=0) * np.sum(ry * ry))
    corr = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 1e-15)
    corr = np.clip(corr, -0.999999, 0.999999)
    dof = max(1, len(y) - len(conditioning_indices) - 2)
    t_stat = np.abs(corr) * np.sqrt(dof / np.maximum(1e-12, 1.0 - corr * corr))
    return 2.0 * stats.t.sf(t_stat, dof)


def iamb_select(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_selected: int = 120,
) -> list[int]:
    selected: list[int] = []
    available = set(range(x.shape[1]))
    while available and len(selected) < max_selected:
        pvalues = partial_pvalues(x, y, selected)
        for index in selected:
            pvalues[index] = 1.0
        best = int(np.nanargmin(pvalues))
        if not np.isfinite(pvalues[best]) or pvalues[best] >= alpha:
            break
        selected.append(best)
        available.discard(best)

    changed = True
    while changed and selected:
        changed = False
        for index in list(selected):
            conditioning = [j for j in selected if j != index]
            pvalue = partial_pvalues(x[:, [index] + conditioning], y, list(range(1, 1 + len(conditioning))))[0]
            if not np.isfinite(pvalue) or pvalue >= alpha:
                selected.remove(index)
                changed = True
    return selected


def pad_selection(
    selected: Sequence[int],
    x: np.ndarray,
    y: np.ndarray,
    target: int,
) -> list[int]:
    selected = list(dict.fromkeys(int(i) for i in selected))
    if len(selected) >= target:
        return selected
    correlations = []
    for j in range(x.shape[1]):
        if j in selected:
            continue
        try:
            rho = stats.spearmanr(x[:, j], y, nan_policy="omit").statistic
        except Exception:
            rho = 0.0
        correlations.append((abs(float(rho)) if np.isfinite(rho) else 0.0, j))
    correlations.sort(key=lambda item: (-item[0], item[1]))
    for _, index in correlations:
        selected.append(index)
        if len(selected) >= min(target, x.shape[1]):
            break
    return selected


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 1.0


def overlap_coefficient(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    denominator = min(len(sa), len(sb))
    return len(sa & sb) / denominator if denominator else 1.0


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        mask = time > time[i]
        if not np.any(mask):
            continue
        comparable += float(np.sum(mask))
        concordant += float(np.sum(risk[i] > risk[mask]))
        concordant += 0.5 * float(np.sum(risk[i] == risk[mask]))
    return concordant / comparable if comparable > 0 else np.nan


def binary_auc_at_horizon(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: float) -> tuple[float, int]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    valid = cases | controls
    if valid.sum() < 10 or cases.sum() == 0 or controls.sum() == 0:
        return np.nan, int(valid.sum())
    y = cases[valid].astype(int)
    return float(roc_auc_score(y, risk[valid])), int(valid.sum())


def require_lifelines():
    try:
        from lifelines import CoxPHFitter  # noqa: F401
        import lifelines
        return lifelines.__version__
    except Exception as exc:
        raise RuntimeError(
            "lifelines is required for M6. Install it in the active venv with: "
            "python -m pip install lifelines"
        ) from exc


def fit_cox_risk(
    train_x: pd.DataFrame,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_x: pd.DataFrame,
    penalizer: float,
) -> tuple[np.ndarray, dict]:
    from lifelines import CoxPHFitter

    train = train_x.copy()
    train["__time"] = np.asarray(train_time, dtype=float)
    train["__event"] = np.asarray(train_event, dtype=int)
    model = CoxPHFitter(penalizer=float(penalizer))
    model.fit(train, duration_col="__time", event_col="__event", show_progress=False)
    risk = model.predict_partial_hazard(test_x).to_numpy(dtype=float).ravel()
    return risk, {
        "coefficients": {str(k): float(v) for k, v in model.params_.to_dict().items()},
        "log_likelihood": float(model.log_likelihood_),
        "concordance_train": float(model.concordance_index_),
    }


def median_impute_scale_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_num = train.apply(pd.to_numeric, errors="coerce").copy()
    test_num = test.apply(pd.to_numeric, errors="coerce").copy()
    medians = train_num.median(axis=0).fillna(0.0)
    train_num = train_num.fillna(medians)
    test_num = test_num.fillna(medians)
    means = train_num.mean(axis=0)
    sds = train_num.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    if scale:
        train_num = (train_num - means) / sds
        test_num = (test_num - means) / sds
    return train_num, test_num, {
        "medians": {str(k): float(v) for k, v in medians.items()},
        "means": {str(k): float(v) for k, v in means.items()},
        "sds": {str(k): float(v) for k, v in sds.items()},
    }


def rank_normalize_separately(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    n = len(frame)
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = values.median()
        values = values.fillna(median if np.isfinite(median) else 0.0)
        ranks = values.rank(method="average")
        probabilities = (ranks - 0.5) / n
        result[column] = stats.norm.ppf(np.clip(probabilities, 1e-6, 1 - 1e-6))
    return result
