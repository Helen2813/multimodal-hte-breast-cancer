# Multimodal HTE Utility in Breast Cancer

Reproducible codebase for comparing the **prognostic** and **prescriptive** value of clinical and multi-omics modalities for heterogeneous treatment-effect (HTE) estimation in breast cancer.

## Why this is a separate repository

The feature panels derived in Paper 1 are treated here as **frozen, versioned inputs**. Paper 1 remains the source of the selection methodology; this repository contains the new treatment-effect, modality-utility, simulation, and external-replication analyses.

This separation prevents accidental modification of the published pipeline, gives Paper 2 an independent environment and commit history, and makes the analysis easier to archive with a DOI.

## Initial analysis contract

The first milestone does not fit causal models. It verifies that:

1. clinical, treatment, survival, and modality tables use a consistent patient identifier;
2. every table contains one row per patient;
3. treatment and event indicators are binary;
4. survival time is non-negative;
5. frozen Paper 1 panels are copied with SHA-256 checksums;
6. patient overlap and missingness are reported before any complete-case restriction.

## Repository layout

```text
configs/                       Analysis configuration files
data/
  raw/                         Never committed
  processed/paper1_panels/     Frozen exported candidate panels
  manifests/                   Provenance and SHA-256 checksums
results/reports/               Validation and cohort reports
scripts/                       Command-line entry points
src/modality_hte/              Reusable Python package
tests/                         Unit tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

Copy the frozen Paper 1 panel files into this repository while recording provenance:

```bash
python scripts/import_frozen_panels.py \
  --source-dir /path/to/Thesis_v3/exported_panels \
  --destination data/processed/paper1_panels \
  --source-repository Helen2813/Thesis_v3 \
  --source-commit <PAPER1_COMMIT_SHA>
```

Edit `configs/tcga_brca.example.yaml`, then run:

```bash
python scripts/validate_inputs.py --config configs/tcga_brca.example.yaml
```

Outputs:

- `results/reports/input_validation.json`
- `results/reports/patient_overlap.csv`
- `results/reports/modality_missingness.csv`

## Planned milestones

1. Input validation and cohort audit.
2. Frozen survival-informed panels from Paper 1.
3. Common confounding adjustment with varying effect-modifier spaces.
4. Prognostic versus prescriptive utility benchmark.
5. Grouped modality ablation/Shapley utility with uncertainty.
6. Semi-synthetic simulations.
7. METABRIC harmonized replication.

## Data policy

Do not commit controlled, patient-level, or large raw molecular data. Commit only schemas, feature names, checksums, synthetic test fixtures, and scripts that reconstruct the analysis from authorized local files.
