from __future__ import annotations

from pathlib import Path
import pandas as pd

from _common import (
    MODALITY_FOLDERS, PROCESSED_DIR, DERIVED_DIR, ensure_dirs,
    find_ite_file, find_merge_file, save_json, write_markdown,
)


def main() -> int:
    ensure_dirs()
    out_dir = DERIVED_DIR / "audits"
    checks: list[dict[str, object]] = []

    def add(item: str, path: Path | None, required: bool, note: str) -> None:
        checks.append({
            "item": item,
            "required": required,
            "found": bool(path and path.exists()),
            "path": str(path) if path else "",
            "note": note,
        })

    add("processed directory", PROCESSED_DIR, True, "Copied legacy processed data")
    add("ITE treatment/outcome table", find_ite_file(), True, "Prefer v2")
    add("MERGE 08 composite", find_merge_file("MERGE"), True, "Complete-case frozen matrix")
    add("MERGE_continuous_outer 08 composite", find_merge_file("MERGE_continuous_outer"), True, "Outer frozen matrix")

    for modality, folder_name in MODALITY_FOLDERS.items():
        folder = PROCESSED_DIR / folder_name
        add(f"{modality} folder", folder, True, "Legacy modality folder")
        if modality != "clinical":
            summaries = list(folder.rglob("summary_all_results.csv")) if folder.exists() else []
            add(f"{modality} summary", summaries[0] if summaries else None, True, "MB summary")
            add(f"{modality} statistical_filtered", folder / "statistical_filtered", True, "Candidate panels")
            add(f"{modality} mb_results", folder / "mb_results", True, "Selected feature lists")

    df = pd.DataFrame(checks)
    df.to_csv(out_dir / "00_input_validation.csv", index=False)
    save_json(checks, out_dir / "00_input_validation.json")

    missing = df[(df["required"]) & (~df["found"])]
    lines = [
        "# Input validation", "",
        f"- Processed root: `{PROCESSED_DIR}`",
        f"- Required missing: **{len(missing)}**", "", "## Results", "",
    ]
    for row in checks:
        status = "FOUND" if row["found"] else "MISSING"
        lines.append(f"- **{status}** — {row['item']}" + (f": `{row['path']}`" if row['path'] else ""))
    write_markdown(lines, out_dir / "00_input_validation.md")

    print(df[["item", "required", "found", "path"]].to_string(index=False))
    print(f"\nSaved to: {out_dir}")
    if not missing.empty:
        print("\nSTOP: required inputs are missing.")
        return 1
    print("\nPASS: all required inputs were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
