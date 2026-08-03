# Breast Cancer Treatment-Sequencing and Cross-Cohort Multi-Omics Evaluation

This repository contains reproducible research code for two companion breast-cancer studies:

1. **Treatment-sequencing audit and landmark analysis**
2. **Cross-cohort evaluation of multimodal prognostic features**

The project began as a follow-up to a published dependency-aware feature-selection framework and now includes cohort reconstruction, treatment-timing audits, survival estimation, simulation studies, cross-platform harmonization, repeated nested validation, bootstrap inference, and publication-asset generation.

> **Research-use only.** This repository is not clinical software and must not be used to make treatment decisions.

---

## Studies supported by this repository

### Study A: treatment sequencing in landmark analyses

**Working title**

> *Auditing Treatment Sequencing in Landmark Analyses of Treatment Initiation: An Empirically Anchored Simulation and Breast Cancer Application*

This study evaluates whether an upstream therapy can remain strongly imbalanced after treatment assignment and follow-up have been aligned at a fixed landmark.

The breast-cancer application examines hormone-therapy initiation by day 180 among patients with HR-positive, HER2-negative TCGA-BRCA disease. The workflow includes:

- reconstruction of hormone-therapy and chemotherapy timing;
- a day-180 landmark design;
- an outcome-blind upstream-treatment sequencing audit;
- overlap weighting;
- a 730-day post-landmark restricted mean survival time contrast;
- cross-fitted censoring and outcome nuisance models;
- a fully refitted patient bootstrap;
- censoring-floor and patient-influence sensitivity analyses;
- an empirically anchored confirmatory simulation.

The historical full-cohort analysis and the sequencing-aware amended analysis are preserved as different target populations. Their numerical difference is not interpreted as a direct estimate of bias removed.

### Study B: cross-cohort multi-omics prognostic evaluation

**Working title**

> *Cross-Cohort Evaluation of Multimodal Prognostic Features in Breast Cancer: Assayability, Stability, and Incremental Utility*

This study separates five properties that are often conflated in cross-cohort multi-omics research:

- cross-platform assayability;
- exact fixed-panel transport;
- feature-selection stability;
- biological recurrence;
- incremental prognostic utility beyond clinical information.

The design contains two complementary tracks:

- **Track A:** outcome-blind transport of fixed TCGA feature panels to METABRIC without METABRIC refitting;
- **Track B:** leakage-controlled reconstruction of the dependency-aware selection framework within METABRIC using repeated nested cross-validation.

Overall survival is the primary endpoint. Molecular utility is assessed through paired comparison with a matched clinical-only model. Recurrence-free survival is evaluated separately as a prespecified sensitivity endpoint.

### Relationship to the published feature-selection study

The upstream feature-selection methodology is described in:

> Krikun E, Alkhateeb A. A Markov Blanket-based framework for dependency-aware feature selection in multimodal breast cancer data. *Network Modeling Analysis in Health Informatics and Bioinformatics*. 2026;15:158.  
> DOI: https://doi.org/10.1007/s13721-026-00832-1

Feature panels derived from that work are treated here as frozen, versioned inputs. This repository does not modify the published selection pipeline.

---

## Repository organization

The exact contents may vary by release, but the project follows this structure:

```text
configs/
    Analysis configuration files and locked protocol settings.

data/
    raw/
        Source cohort files stored locally. Never committed.
    processed/
        Harmonized local analysis tables and frozen panel inputs.
    manifests/
        Provenance records, file inventories, and SHA-256 checksums.

scripts/
    Command-line analysis stages, protocol locks, bootstrap procedures,
    simulations, diagnostics, and publication-asset generators.

src/modality_hte/
    Reusable Python package code.

tests/
    Unit and validation tests.

results/
    reports/
        Input validation, cohort audits, and schema checks.
    tables/
        Aggregate analysis tables and locked numerical summaries.
    figures/
        Publication and supplementary figures.
    logs/
        Execution transcripts and stage logs.
```

Some large or patient-level outputs are intentionally marked `LOCAL_ONLY` and must not be committed.

---

## Data sources

The analyses use de-identified research data from:

- **TCGA-BRCA**
- **METABRIC**

Raw patient-level, clinical, treatment, and molecular files are not distributed in this repository. Users must obtain the source data through the original authorized repositories and comply with the applicable access terms.

The repository may contain:

- schemas;
- source identifiers;
- feature names;
- data dictionaries;
- deterministic preprocessing code;
- checksums;
- synthetic fixtures;
- aggregate, non-identifying outputs.

The repository must not contain:

- controlled-access source data;
- raw patient-level tables;
- patient-level out-of-fold predictions;
- large molecular matrices;
- identifiable or potentially re-identifiable information.

---

## Environment setup

Python 3.10 or newer is required.

### Clone the repository

```bash
git clone https://github.com/Helen2813/multimodal-hte-breast-cancer.git
cd multimodal-hte-breast-cancer
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install the package

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,causal,survival]"
```

The current `pyproject.toml` defines the core package and optional development, causal-inference, and survival-analysis dependencies. Publication-stage releases should also preserve a frozen environment file or package lock containing the exact versions used for the manuscript.

---

## Initial input validation

An example TCGA-BRCA configuration is provided at:

```text
configs/tcga_brca.example.yaml
```

After adapting the local paths, run:

```bash
python scripts/validate_inputs.py --config configs/tcga_brca.example.yaml
```

Typical validation outputs include:

```text
results/reports/input_validation.json
results/reports/patient_overlap.csv
results/reports/modality_missingness.csv
```

Validation checks include:

- consistent patient identifiers;
- one row per patient where required;
- binary treatment and event indicators;
- non-negative survival times;
- required clinical variables;
- modality availability;
- patient overlap and missingness before complete-case restriction.

---

## Importing frozen feature panels

Frozen panels from the published feature-selection repository should be imported with provenance and checksums rather than copied informally.

Example:

```bash
python scripts/import_frozen_panels.py \
  --source-dir /path/to/exported_panels \
  --destination data/processed/paper1_panels \
  --source-repository Helen2813/Thesis_v3 \
  --source-commit <PAPER1_COMMIT_SHA>
```

The import step should record:

- source repository;
- source commit;
- original file names;
- destination paths;
- SHA-256 hashes.

---

## Reproducing the analyses

The project uses staged, non-interactive pipelines. Each major stage generally performs one of the following roles:

1. data and schema audit;
2. harmonization or cohort construction;
3. protocol lock;
4. point estimation;
5. repeated validation or bootstrap inference;
6. simulation;
7. sensitivity analysis;
8. publication-table or figure generation.

Do not run all scripts blindly. Use the runner, stage README, and configuration files associated with the manuscript-specific release.

### Study A stage groups

The treatment-sequencing analysis includes stages for:

- chemotherapy-sequencing audit;
- Candidate V10 protocol and cohort lock;
- propensity-balance verification;
- primary point estimation;
- fully refitted patient bootstrap;
- interval sensitivity;
- censoring-floor sensitivity;
- event and non-event patient-influence analysis;
- simulation calibration;
- independent confirmatory simulation;
- compact publication summaries.

### Study B stage groups

The cross-cohort analysis includes stages for:

- TCGA and METABRIC data-design audit;
- feature and platform harmonization;
- outcome-blind fixed-panel transport;
- Track B protocol lock;
- nested-analysis pilot;
- repeated multimodal analysis;
- modality-specific repeated analyses;
- paired patient-bootstrap inference;
- feature and pathway concordance;
- secondary Nottingham Prognostic Index comparison;
- recurrence-free-survival sensitivity analysis.

Each stage should write to a new output namespace and preserve prior locked results.

---

## Reproducibility safeguards

The analysis code follows these principles:

- **Outcome-blind transport:** target-cohort outcomes are not used to modify fixed Track A feature panels.
- **Leakage control:** outcome-informed preprocessing, screening, selection, and fitting are confined to outer-training folds in Track B.
- **Matched comparison:** clinical-only and clinical-plus-omics models are evaluated on the same patients and resampling structure.
- **Protocol locking:** populations, estimands, model settings, seeds, and decision rules are saved before final results are calculated.
- **Immutable historical analyses:** earlier completed analyses are preserved rather than overwritten.
- **Checkpointing:** long repeated analyses and bootstrap jobs save resumable intermediate outputs.
- **Hashing and manifests:** scripts, configurations, inputs, and key outputs are tracked using checksums.
- **Conditional-bootstrap labeling:** bootstrap intervals based on locked out-of-fold predictions are not described as full-pipeline bootstrap intervals.
- **No outcome-driven claim switching:** sensitivity endpoints and secondary comparators do not replace the prespecified primary analysis.

---

## Results and output policy

Safe-to-share outputs generally include:

- aggregate cohort counts;
- aggregate performance estimates;
- confidence intervals;
- stability summaries;
- simulation summaries;
- non-identifying figures;
- code and configuration hashes.

Do not commit files containing one row per patient unless they are synthetic. Files containing names such as the following should remain local:

```text
*_LOCAL_ONLY.csv
*patient_level*
*oof_predictions*
*bootstrap_patient*
```

Before publishing a release, inspect the Git history as well as the current working tree. Deleting a sensitive file in a later commit does not remove it from earlier history.

---

## Tests and code quality

Run the test suite with:

```bash
pytest
```

Run static checks with:

```bash
ruff check .
```

A manuscript release should be created only after:

- required tests pass;
- the main stage logs show successful completion;
- manuscript values are linked to locked output files;
- patient-level files have been excluded;
- the release commit has been recorded.

---

## Manuscript-specific releases

Use separate immutable tags for manuscript submissions, for example:

```text
paper-2a-v1.0
paper-2b-v1.0
```

Each release should include:

- the exact code used for the submitted manuscript;
- locked configurations;
- environment information;
- aggregate tables and figures;
- a manuscript-to-output inventory;
- a checksum manifest;
- a short reproduction guide.

For long-term archival, connect the GitHub release to Zenodo or another repository that issues a permanent DOI.

---

## Citation

When using this repository, cite:

1. the relevant companion manuscript or archived release; and
2. the published Markov Blanket feature-selection article when frozen panels or the dependency-aware framework are used.

A `CITATION.cff` file should be updated for each manuscript-specific release.

---

## License

A software license has not yet been finalized for this repository. Add a `LICENSE` file before the archival public release and update this section accordingly. Until a license is added, public visibility should not be interpreted as permission to reuse, modify, or redistribute the code.

---

## Authors

- **Elena Krikun** — Lakehead University  
- **Abedalrhman Alkhateeb** — Lakehead University

Correspondence: `ekrikun@lakeheadu.ca`

---

## Disclaimer

The analyses are retrospective research studies based on observational data. They do not establish treatment recommendations, validate a clinical decision-support tool, or replace prospective clinical evidence. Any causal interpretation depends on the stated identification assumptions, data quality, and measurement of relevant treatment and clinical history.
