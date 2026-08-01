from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from _metabric_m5_utils import (
    exact_column, load_config, out_dir, print_table, project_root, raw_dir,
    read_cbio, rel, sha256, write_csv
)


def request_json(url: str, timeout: int, retries: int) -> tuple[object | None, str]:
    last_error = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "METABRIC-M5-panel-recovery/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), ""
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(5.0, 0.75 * (2 ** attempt)))
    return None, last_error


def extract_genes(payload: object) -> tuple[list[str], str]:
    if not isinstance(payload, dict):
        return [], ""
    description = str(payload.get("description", ""))
    candidates = payload.get("genes") or payload.get("genePanelGenes") or []
    genes = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict):
                symbol = (
                    item.get("hugoGeneSymbol")
                    or item.get("hugoSymbol")
                    or item.get("geneSymbol")
                    or item.get("symbol")
                )
            else:
                symbol = item
            if symbol:
                symbol = str(symbol).strip().upper()
                if symbol and symbol not in genes:
                    genes.append(symbol)
    return sorted(genes), description


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    raw = raw_dir(root, cfg)
    out = out_dir(root, cfg)
    settings = cfg["cbioportal"]

    print("=" * 124)
    print("METABRIC M5.27 - OFFICIAL METABRIC_173 PANEL RECOVERY")
    print("=" * 124)

    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "cbioportal_METABRIC_173_DETAILED.json"

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        source = "cache"
        error = ""
    else:
        payload, error = request_json(
            settings["panel_url"],
            int(settings["timeout_seconds"]),
            int(settings["max_retries"]),
        )
        if payload is not None:
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            source = "official_cbioportal_api"
        else:
            source = "unavailable"

    genes, description = extract_genes(payload)
    if not genes:
        raise RuntimeError(
            f"Could not recover {settings['panel_id']} gene list from official cBioPortal API. Error: {error}"
        )

    mutation = read_cbio(raw / cfg["metabric_files"]["mutations"])
    mutation_gene_col = exact_column(mutation.columns, ["Hugo_Symbol", "HUGO_SYMBOL"])
    mutation_sample_col = exact_column(
        mutation.columns,
        ["Tumor_Sample_Barcode", "TUMOR_SAMPLE_BARCODE", "SAMPLE_ID"],
    )
    variant_class_col = exact_column(mutation.columns, ["Variant_Classification"])
    if mutation_gene_col is None or mutation_sample_col is None:
        raise RuntimeError("Mutation file lacks gene or sample column.")

    observed_genes = sorted(set(
        mutation[mutation_gene_col].dropna().astype(str).str.upper()
    ))
    observed_samples = sorted(set(
        mutation[mutation_sample_col].dropna().astype(str)
    ))

    panel_matrix = read_cbio(raw / cfg["metabric_files"]["gene_panel_matrix"])
    panel_sample_col = exact_column(panel_matrix.columns, ["SAMPLE_ID"])
    mutation_profile_col = exact_column(panel_matrix.columns, ["mutations"])
    if panel_sample_col is None or mutation_profile_col is None:
        raise RuntimeError("Gene panel matrix lacks SAMPLE_ID or mutations column.")

    assigned = panel_matrix[
        panel_matrix[mutation_profile_col].astype(str) == settings["panel_id"]
    ][panel_sample_col].dropna().astype(str)
    assigned_samples = sorted(set(assigned))

    selected_mutation_rows = pd.read_csv(
        root / cfg["metabric_m4_dir"] / "m20_selected_tcga_feature_identifiers.csv",
        dtype=str,
    )
    selected_genes = sorted(set(
        selected_mutation_rows.loc[
            selected_mutation_rows["modality"] == "mutations",
            "canonical_identifier",
        ].dropna().astype(str).str.upper()
    ))
    selected_genes = [gene for gene in selected_genes if not gene.endswith("_MISSING")]

    panel_set = set(genes)
    selected_covered = sorted(set(selected_genes) & panel_set)
    selected_not_covered = sorted(set(selected_genes) - panel_set)
    observed_selected = sorted(set(selected_genes) & set(observed_genes))

    sample_to_positive = {
        sample: set(group[mutation_gene_col].dropna().astype(str).str.upper())
        for sample, group in mutation.groupby(mutation_sample_col)
    }
    matrix_rows = []
    for sample in assigned_samples:
        positives = sample_to_positive.get(sample, set())
        row = {"sample_id": sample}
        for gene in selected_covered:
            row[f"MUT_{gene}"] = 1 if gene in positives else 0
        matrix_rows.append(row)
    mutation_matrix = pd.DataFrame(matrix_rows)
    mutation_matrix.to_csv(
        out / "m27_metabric_selected_mutation_panel_LOCAL_ONLY.csv",
        index=False,
    )

    class_rows = []
    if variant_class_col:
        counts = mutation[variant_class_col].fillna("[MISSING]").astype(str).value_counts()
        class_rows = [
            {"variant_classification": label, "count": int(count)}
            for label, count in counts.items()
        ]

    panel_rows = [{"gene": gene} for gene in genes]
    write_csv(out / "m27_metabric_173_gene_list.csv", panel_rows)
    write_csv(out / "m27_variant_classification_counts.csv", class_rows)

    summary = {
        "panel_id": settings["panel_id"],
        "description": description,
        "source": source,
        "source_url": settings["panel_url"],
        "cache_path": rel(root, cache_path),
        "cache_sha256": sha256(cache_path),
        "panel_gene_count": len(genes),
        "assigned_samples": len(assigned_samples),
        "mutation_file_samples_with_positive_calls": len(observed_samples),
        "unique_observed_mutated_genes": len(observed_genes),
        "observed_mutated_genes_inside_panel": len(set(observed_genes) & panel_set),
        "observed_mutated_genes_outside_panel": len(set(observed_genes) - panel_set),
        "selected_tcga_mutation_genes": len(selected_genes),
        "selected_genes_covered_by_panel": len(selected_covered),
        "selected_genes_not_covered_by_panel": len(selected_not_covered),
        "selected_covered_examples": selected_covered,
        "selected_not_covered_examples": selected_not_covered,
        "selected_genes_observed_positive": len(observed_selected),
        "selected_observed_positive_examples": observed_selected,
        "negative_coding_allowed": len(genes) == 173 and len(assigned_samples) > 0,
        "negative_coding_rule": (
            "Zero is assigned only for samples explicitly assigned to METABRIC_173 and only for genes in the recovered panel."
        ),
    }
    (out / "m27_metabric_173_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Panel recovery summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nVariant classification counts")
    print_table(class_rows, ["variant_classification", "count"])

    if len(genes) != 173:
        raise RuntimeError(f"Recovered panel contains {len(genes)} genes, expected 173.")

    print("\nPatient/sample identifiers are stored only in LOCAL_ONLY matrix and are not printed.")
    print("PASS: official METABRIC_173 definition recovered; panel-aware mutation coding is now possible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
