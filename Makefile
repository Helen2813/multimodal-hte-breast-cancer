.PHONY: install test lint validate

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

validate:
	python scripts/validate_inputs.py --config configs/tcga_brca.example.yaml
