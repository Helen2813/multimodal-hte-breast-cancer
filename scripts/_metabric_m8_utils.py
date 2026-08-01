from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import roc_auc_score

NONSYNONYMOUS_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "Splice_Region", "In_Frame_Del",
    "In_Frame_Ins", "Nonstop_Mutation", "Translation_Start_Site",
}


def project_root() -> Path:
    return Path.cwd().resolve()


def load_config(root: Path) -> dict:
    return json.loads((root / "metabric_m8_config.json").read_text(encoding="utf-8"))


def out_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["output_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(root: Path, cfg: dict) -> Path:
    path = (root / cfg["figure_dir"]).resolve()
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


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = list(fieldnames or ["empty"])
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
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


def atomic_write_csv(path: Path, rows: Sequence[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, rows)
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def print_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> None:
    if not rows:
        print("<empty>")
        return
    shown = list(rows if max_rows is None else rows[:max_rows])
    widths = {column: len(column) for column in columns}
    rendered = []
    for row in shown:
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


def fast_harrell_c_index(time_values, event_values, risk_values) -> float:
    time_values = np.asarray(time_values, dtype=float)
    event_values = np.asarray(event_values, dtype=int)
    risk_values = np.asarray(risk_values, dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(event_values) & np.isfinite(risk_values)
    if valid.sum() < 3 or event_values[valid].sum() == 0:
        return float("nan")
    return float(concordance_index(time_values[valid], -risk_values[valid], event_observed=event_values[valid]))


def binary_auc_at_horizon(time_values, event_values, risk_values, horizon: float) -> tuple[float, int]:
    time_values = np.asarray(time_values, dtype=float)
    event_values = np.asarray(event_values, dtype=int)
    risk_values = np.asarray(risk_values, dtype=float)
    cases = (event_values == 1) & (time_values <= horizon)
    controls = time_values > horizon
    valid = cases | controls
    if valid.sum() < 10 or cases.sum() == 0 or controls.sum() == 0:
        return float("nan"), int(valid.sum())
    return float(roc_auc_score(cases[valid].astype(int), risk_values[valid])), int(valid.sum())


def median_impute_scale_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_numeric = train.apply(pd.to_numeric, errors="coerce").copy()
    test_numeric = test.apply(pd.to_numeric, errors="coerce").copy()
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_numeric = train_numeric.fillna(medians)
    test_numeric = test_numeric.fillna(medians)
    means = train_numeric.mean(axis=0)
    sds = train_numeric.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return (train_numeric - means) / sds, (test_numeric - means) / sds


def fit_penalized_cox(train_features, train_time, train_event, test_features, penalizer_sequence):
    last_error = ""
    for penalizer in penalizer_sequence:
        try:
            frame = train_features.copy()
            frame["__time"] = np.asarray(train_time, dtype=float)
            frame["__event"] = np.asarray(train_event, dtype=int)
            model = CoxPHFitter(penalizer=float(penalizer))
            model.fit(frame, duration_col="__time", event_col="__event", show_progress=False)
            risk = model.predict_partial_hazard(test_features).to_numpy(dtype=float).ravel()
            return risk, float(model.concordance_index_), float(penalizer)
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
    raise RuntimeError(f"All locked Cox penalizers failed: {last_error}")


def rank_gaussian_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    output = np.empty_like(matrix)
    n = len(matrix)
    for column in range(matrix.shape[1]):
        ranks = stats.rankdata(matrix[:, column], method="average")
        output[:, column] = stats.norm.ppf(np.clip((ranks - 0.5) / n, 1e-6, 1 - 1e-6))
    return output


def residualize_vector(y, z):
    y = np.asarray(y, dtype=float)
    if z is None or z.size == 0:
        return y - y.mean()
    design = np.column_stack([np.ones(len(y)), z])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def residualize_matrix(x, z):
    x = np.asarray(x, dtype=float)
    if z is None or z.size == 0:
        return x - x.mean(axis=0, keepdims=True)
    design = np.column_stack([np.ones(len(x)), z])
    coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coefficients


def partial_pvalues(x, y, conditioning_indices):
    z = x[:, list(conditioning_indices)] if conditioning_indices else None
    ry = residualize_vector(y, z)
    rx = residualize_matrix(x, z)
    numerator = rx.T @ ry
    denominator = np.sqrt(np.sum(rx * rx, axis=0) * np.sum(ry * ry))
    correlations = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15)
    correlations = np.clip(correlations, -0.999999, 0.999999)
    degrees = max(1, len(y) - len(conditioning_indices) - 2)
    statistic = np.abs(correlations) * np.sqrt(degrees / np.maximum(1e-12, 1 - correlations * correlations))
    return 2.0 * stats.t.sf(statistic, degrees)


def iamb_select(x, y, alpha: float, maximum_selected: int) -> list[int]:
    x = rank_gaussian_matrix(np.asarray(x, dtype=float))
    y = rank_gaussian_matrix(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()
    selected: list[int] = []
    while len(selected) < min(maximum_selected, x.shape[1]):
        pvalues = partial_pvalues(x, y, selected)
        if selected:
            pvalues[selected] = 1.0
        best = int(np.nanargmin(pvalues))
        if not np.isfinite(pvalues[best]) or pvalues[best] >= alpha:
            break
        selected.append(best)
    changed = True
    while changed and selected:
        changed = False
        for feature in list(selected):
            conditioning = [index for index in selected if index != feature]
            local = x[:, [feature] + conditioning]
            pvalue = partial_pvalues(local, y, list(range(1, 1 + len(conditioning))))[0]
            if not np.isfinite(pvalue) or pvalue >= alpha:
                selected.remove(feature)
                changed = True
    return selected


def pad_selection(selected, matrix, outcome, target: int) -> list[int]:
    selected = list(dict.fromkeys(int(index) for index in selected))
    scores = []
    for column in range(matrix.shape[1]):
        if column in selected:
            continue
        rho = stats.spearmanr(matrix[:, column], outcome, nan_policy="omit").statistic
        scores.append((abs(float(rho)) if np.isfinite(rho) else 0.0, column))
    scores.sort(key=lambda item: (-item[0], item[1]))
    for _, column in scores:
        selected.append(column)
        if len(selected) >= min(target, matrix.shape[1]):
            break
    return selected


def top_signed_spearman_chunked(frame, training_positions, outcome, positive, negative, chunk_size=512):
    outcome = np.asarray(outcome, dtype=float)
    outcome_rank = stats.rankdata(outcome, method="average").astype(np.float64)
    outcome_centered = outcome_rank - outcome_rank.mean()
    outcome_norm = np.sqrt(np.sum(outcome_centered * outcome_centered))
    positions = np.asarray(training_positions, dtype=int)
    rows = []
    columns = list(frame.columns)
    for start in range(0, len(columns), chunk_size):
        chunk_columns = columns[start:start + chunk_size]
        chunk = frame.iloc[positions][chunk_columns].apply(pd.to_numeric, errors="coerce")
        medians = chunk.median(axis=0).fillna(0.0)
        ranks = chunk.fillna(medians).rank(axis=0, method="average").to_numpy(dtype=np.float64)
        ranks -= ranks.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(ranks * ranks, axis=0)) * outcome_norm
        correlations = np.divide(ranks.T @ outcome_centered, denominator, out=np.zeros(len(chunk_columns)), where=denominator > 1e-15)
        rows.extend({"feature": column, "spearman_rho": float(correlation)} for column, correlation in zip(chunk_columns, correlations))
    positive_rows = [row for row in sorted(rows, key=lambda item: (-item["spearman_rho"], item["feature"])) if row["spearman_rho"] > 0][:positive]
    negative_rows = [row for row in sorted(rows, key=lambda item: (item["spearman_rho"], item["feature"])) if row["spearman_rho"] < 0][:negative]
    selected = []
    for row in positive_rows + negative_rows:
        if row["feature"] not in selected:
            selected.append(row["feature"])
    return selected, {row["feature"]: row["spearman_rho"] for row in rows}


def read_continuous_sample_matrix(path: Path, prefix: str) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    dtype = {column: "float32" for column in header[1:]}
    frame = pd.read_csv(path, dtype=dtype, low_memory=False)
    identifier = frame.columns[0]
    frame = frame.rename(columns={identifier: "sample_id"})
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.set_index("sample_id")
    frame.columns = [f"{prefix}__{column}" for column in frame.columns]
    return frame


def read_gene_row_matrix(path: Path, prefix: str) -> pd.DataFrame:
    header = list(pd.read_csv(path, sep="\t", comment="#", nrows=0).columns)
    identifier = next(column for column in header if str(column).upper() in {"HUGO_SYMBOL", "HUGO_SYMBOLS"})
    sample_columns = [column for column in header if column not in {identifier, "Entrez_Gene_Id", "ENTREZ_GENE_ID"}]
    dtype = {column: "float32" for column in sample_columns}
    dtype[identifier] = "string"
    frame = pd.read_csv(path, sep="\t", comment="#", dtype=dtype, low_memory=False)
    values = frame[sample_columns].astype("float32")
    values.insert(0, "gene", frame[identifier].astype(str).str.upper())
    matrix = values.groupby("gene", sort=True).median(numeric_only=True).T
    matrix.index = matrix.index.astype(str)
    matrix.index.name = "sample_id"
    matrix.columns = [f"{prefix}__{column}" for column in matrix.columns]
    return matrix


def build_mutation_matrix(root: Path, cfg: dict) -> pd.DataFrame:
    genes = sorted(set(pd.read_csv(root / cfg["files"]["metabric_173_genes"], dtype=str)["gene"].dropna().astype(str).str.upper()))
    assignments = pd.read_csv(root / cfg["files"]["gene_panel_matrix"], sep="\t", comment="#", dtype=str, low_memory=False)
    assigned = sorted(set(assignments.loc[assignments["mutations"].astype(str) == "METABRIC_173", "SAMPLE_ID"].dropna().astype(str)))
    calls = pd.read_csv(root / cfg["files"]["mutations"], sep="\t", comment="#", dtype=str, low_memory=False)
    calls = calls[calls["Variant_Classification"].isin(NONSYNONYMOUS_CLASSES)].copy()
    calls["Hugo_Symbol"] = calls["Hugo_Symbol"].astype(str).str.upper()
    calls = calls[calls["Hugo_Symbol"].isin(genes) & calls["Tumor_Sample_Barcode"].astype(str).isin(assigned)]
    matrix = pd.DataFrame(0.0, index=pd.Index(assigned, name="sample_id"), columns=[f"MUT__{gene}" for gene in genes], dtype=np.float32)
    for sample, group in calls.groupby("Tumor_Sample_Barcode"):
        columns = [f"MUT__{gene}" for gene in sorted(set(group["Hugo_Symbol"])) if gene in genes]
        if columns:
            matrix.loc[str(sample), columns] = 1.0
    return matrix


def strip_feature_prefix(feature: str) -> str:
    text = str(feature)
    for prefix in ("RNA__", "CNA__", "METH__", "MUT__", "CLIN__"):
        if text.startswith(prefix):
            return text[len(prefix):].upper()
    return text.upper()


def read_feature_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    output = []
    for token in re.split(r"[\s,;]+", text):
        token = token.strip().strip('"').strip("'")
        if token and token.lower() not in {"gene", "genes", "feature", "features"} and token not in output:
            output.append(token)
    return output


def canonical_ensembl(value: str) -> str:
    match = re.search(r"ENSG\d+", str(value).upper())
    return match.group(0) if match else ""


def gene_like_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    output = []
    for token in re.split(r"[;,|/\s]+", str(value).upper()):
        token = token.strip()
        if 2 <= len(token) <= 30 and re.match(r"^[A-Z0-9][A-Z0-9._-]*$", token) and not token.startswith("CG") and token not in {"NA", "NAN", "NONE", "UNKNOWN", "GENE"} and any(character.isalpha() for character in token):
            output.append(token)
    return sorted(set(output))


def jaccard(first: Iterable[str], second: Iterable[str]) -> float:
    first_set, second_set = set(first), set(second)
    denominator = len(first_set | second_set)
    return len(first_set & second_set) / denominator if denominator else 1.0


def overlap_coefficient(first: Iterable[str], second: Iterable[str]) -> float:
    first_set, second_set = set(first), set(second)
    denominator = min(len(first_set), len(second_set))
    return len(first_set & second_set) / denominator if denominator else 1.0


def hypergeometric_overlap(universe_size, first_size, second_size, overlap_size) -> float:
    if min(universe_size, first_size, second_size) <= 0:
        return float("nan")
    return float(stats.hypergeom.sf(overlap_size - 1, universe_size, first_size, second_size))


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    output = np.empty_like(corrected)
    output[order] = corrected
    return output


def download_with_retries(url: str, path: Path, timeout_seconds: int, retries: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "METABRIC-M8/1.0", "Accept": "*/*"})
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(min(8.0, 0.75 * (2 ** attempt)))
    raise RuntimeError(f"Download failed: {last_error}")


def load_reactome_gmt(zip_path: Path, extraction_directory: Path) -> dict[str, set[str]]:
    extraction_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".gmt")]
        if not members:
            raise RuntimeError("Reactome archive contains no GMT file")
        archive.extract(members[0], extraction_directory)
    pathways: dict[str, set[str]] = {}
    for line in (extraction_directory / members[0]).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        genes = {gene.strip().upper() for gene in fields[2:] if gene.strip() and gene.strip().upper() != "REACTOME PATHWAY"}
        if genes:
            pathways.setdefault(fields[0].strip(), set()).update(genes)
    return pathways
