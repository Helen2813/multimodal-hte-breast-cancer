from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from _metabric_m4_utils import (
    load_config, out_dir, print_table, project_root, sha256, write_csv
)


def fetch_json(url: str, timeout: int, retries: int) -> tuple[object | None, str]:
    last_error = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "METABRIC-M4-reproducible-mapping/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload), ""
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(4.0, 0.5 * (2 ** attempt)))
    return None, last_error


def hgnc_symbols(payload: object | None) -> list[str]:
    if not isinstance(payload, list):
        return []
    symbols = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("dbname", "")).upper() == "HGNC":
            display = str(item.get("display_id", "")).strip().upper()
            if display and display not in symbols:
                symbols.append(display)
    return sorted(symbols)


def lookup_display_name(payload: object | None) -> str:
    if isinstance(payload, dict):
        value = str(payload.get("display_name", "")).strip().upper()
        return value
    return ""


def main() -> int:
    root = project_root()
    cfg = load_config(root)
    out = out_dir(root, cfg)
    settings = cfg["ensembl_mapping"]

    selected = pd.read_csv(
        out / "m20_selected_tcga_feature_identifiers.csv",
        dtype=str,
        low_memory=False,
    )
    ids = sorted({
        value
        for value in selected.loc[
            selected["identifier_type"] == "ensembl",
            "canonical_identifier",
        ].dropna().astype(str)
    })

    cache_dir = out / "cache" / "ensembl"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("METABRIC M4.21 - REPRODUCIBLE ENSEMBL-TO-HGNC MAPPING")
    print("=" * 124)
    print(f"Unique selected Ensembl IDs: {len(ids)}")
    print("Servers: current Ensembl plus GRCh37 cross-check")
    print("Responses are cached and hashed.")

    rows = []
    for index, ensembl_id in enumerate(ids, 1):
        cache_path = cache_dir / f"{ensembl_id}.json"
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            source = "cache"
        else:
            record = {
                "ensembl_id": ensembl_id,
                "requests": {},
            }
            for label, server in (
                ("current", settings["current_server"]),
                ("grch37", settings["grch37_server"]),
            ):
                xref_url = (
                    f"{server}/xrefs/id/{urllib.parse.quote(ensembl_id)}"
                    "?external_db=HGNC;object_type=gene;content-type=application/json"
                )
                lookup_url = (
                    f"{server}/lookup/id/{urllib.parse.quote(ensembl_id)}"
                    "?content-type=application/json"
                )
                xref_payload, xref_error = fetch_json(
                    xref_url,
                    int(settings["timeout_seconds"]),
                    int(settings["max_retries"]),
                )
                lookup_payload, lookup_error = fetch_json(
                    lookup_url,
                    int(settings["timeout_seconds"]),
                    int(settings["max_retries"]),
                )
                record["requests"][label] = {
                    "server": server,
                    "xref_url": xref_url,
                    "xref_payload": xref_payload,
                    "xref_error": xref_error,
                    "lookup_url": lookup_url,
                    "lookup_payload": lookup_payload,
                    "lookup_error": lookup_error,
                }
                time.sleep(float(settings["request_pause_seconds"]))
            cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            source = "network"

        current = record["requests"].get("current", {})
        grch37 = record["requests"].get("grch37", {})
        current_symbols = hgnc_symbols(current.get("xref_payload"))
        grch37_symbols = hgnc_symbols(grch37.get("xref_payload"))
        current_display = lookup_display_name(current.get("lookup_payload"))
        grch37_display = lookup_display_name(grch37.get("lookup_payload"))

        union = sorted(set(current_symbols) | set(grch37_symbols))
        if len(union) == 1:
            selected_symbol = union[0]
            if current_symbols and grch37_symbols and current_symbols == grch37_symbols:
                status = "MAPPED_CURRENT_AND_GRCH37_AGREE"
            elif current_symbols:
                status = "MAPPED_CURRENT_ONLY"
            else:
                status = "MAPPED_GRCH37_ONLY"
        elif len(union) > 1:
            selected_symbol = ""
            status = "AMBIGUOUS_HGNC_SYMBOLS"
        else:
            display_union = sorted({
                x for x in (current_display, grch37_display)
                if x and not x.startswith("ENSG")
            })
            if len(display_union) == 1:
                selected_symbol = display_union[0]
                status = "MAPPED_BY_ENSEMBL_DISPLAY_NAME_FALLBACK"
            elif len(display_union) > 1:
                selected_symbol = ""
                status = "AMBIGUOUS_DISPLAY_NAMES"
            else:
                selected_symbol = ""
                status = "UNMAPPED"

        rows.append({
            "ensembl_id": ensembl_id,
            "selected_hgnc_symbol": selected_symbol,
            "mapping_status": status,
            "current_hgnc_symbols": " | ".join(current_symbols),
            "grch37_hgnc_symbols": " | ".join(grch37_symbols),
            "current_display_name": current_display,
            "grch37_display_name": grch37_display,
            "cache_source": source,
            "cache_path": cache_path.relative_to(root).as_posix(),
            "cache_sha256": sha256(cache_path),
            "current_error": current.get("xref_error", ""),
            "grch37_error": grch37.get("xref_error", ""),
        })
        print(
            f"{index:03d}/{len(ids):03d} {ensembl_id} -> "
            f"{selected_symbol or '[UNRESOLVED]'} ({status}; {source})"
        )

    write_csv(out / "m21_ensembl_to_hgnc_mapping.csv", rows)

    status_rows = []
    for status in sorted({row["mapping_status"] for row in rows}):
        status_rows.append({
            "mapping_status": status,
            "count": sum(row["mapping_status"] == status for row in rows),
        })
    write_csv(out / "m21_mapping_status_summary.csv", status_rows)

    mapped = sum(bool(row["selected_hgnc_symbol"]) for row in rows)
    summary = {
        "selected_ensembl_ids": len(rows),
        "mapped_ids": mapped,
        "mapped_fraction": mapped / len(rows) if rows else 0.0,
        "ambiguous_or_unmapped": len(rows) - mapped,
        "current_server": settings["current_server"],
        "grch37_server": settings["grch37_server"],
    }
    (out / "m21_mapping_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nMapping status summary")
    print_table(status_rows, ["mapping_status", "count"])
    print("\nMapping summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if mapped == 0:
        raise RuntimeError(
            "No selected Ensembl ID was mapped. Check internet access and cached errors."
        )

    print("\nPASS: Ensembl mapping completed and cached. No outcome was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
