from __future__ import annotations

import ast

import pandas as pd

from _common import PROJECT_ROOT, RESULTS_DIR, read_table


REQUIRED_INPUTS = {
    "candidate": (
        RESULTS_DIR / "tables" / "34_paperA_candidate_summary.csv",
        {"primary_effect_days", "if_ci_low", "if_ci_high"},
    ),
    "composition": (
        RESULTS_DIR / "tables" / "37_control_strategy_composition.csv",
        {"control_component", "n", "later_prediction_oof_auc"},
    ),
    "later_balance": (
        RESULTS_DIR / "tables" / "37_later_vs_never_balance.csv",
        {"feature", "abs_smd"},
    ),
    "era": (
        RESULTS_DIR / "tables" / "37_era_interaction_feasibility.csv",
        {"formal_interaction_feasible"},
    ),
    "ccw": (
        RESULTS_DIR / "tables" / "39_ccw_feasibility_decision.csv",
        {"feasibility_status"},
    ),
}


def inspect_optional_dependencies(script_path):
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr == "to_markdown":
                violations.append(
                    f"real to_markdown call at line {node.lineno}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == "tabulate":
                    violations.append(
                        f"tabulate import at line {node.lineno}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] == "tabulate":
                violations.append(
                    f"tabulate import at line {node.lineno}"
                )
    return violations


def main() -> int:
    rows = []
    for name, (path, required_columns) in REQUIRED_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 40 input: {path}")
        frame = read_table(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )
        rows.append(
            {
                "input": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "status": "OK",
                "path": str(path),
            }
        )

    script_path = PROJECT_ROOT / "scripts" / "40_update_paperA_candidate.py"
    violations = inspect_optional_dependencies(script_path)
    if violations:
        raise RuntimeError(
            "Optional Markdown dependencies remain in Stage 40: "
            f"{violations}"
        )

    print("=" * 115)
    print("STAGE 40 PREFLIGHT PASSED")
    print("=" * 115)
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nAST inspection found no real DataFrame.to_markdown call "
        "and no tabulate import. All Stage 37–39 inputs are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
