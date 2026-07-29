from __future__ import annotations

import argparse

from modality_hte.pipelines.validate_inputs import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate clinical and modality input tables.")
    parser.add_argument("--config", required=True, help="Path to YAML analysis configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
