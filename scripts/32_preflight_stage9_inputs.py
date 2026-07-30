from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from _common import PROJECT_ROOT


def load_stage32_module():
    path = PROJECT_ROOT / "scripts" / "32_paperB_landmark_power.py"
    spec = importlib.util.spec_from_file_location("stage32_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_stage32_module()
    cohort, landmark, horizon = module.choose_design()
    result = module.validate_stage32_inputs(
        cohort, landmark, horizon
    )
    print("=" * 115)
    print("STAGE 32 INPUT PREFLIGHT PASSED")
    print("=" * 115)
    print(pd.DataFrame([result]).to_string(index=False))
    print(
        "\nThe real pseudo-outcome filename was discovered from disk; "
        "no hard-coded G-min suffix is used."
    )
    print(
        "The exact modeling table was assembled successfully, including "
        "clinical and RNA feature selection. No _x/_y merge suffixes remain."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
