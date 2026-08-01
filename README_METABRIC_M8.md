# METABRIC Stage M8

Run from the project root:

```powershell
.\run_metabric_stage_m8_modality_concordance.ps1
```

M8 follows the completed M7 core analysis.

## M8.40

Locks the modality-specific and biological-concordance protocol before any new
modality result is inspected.

## M8.41

Runs four separate leakage-controlled nested analyses:

- RNA
- CNV
- promoter/gene-level methylation
- panel-aware nonsynonymous mutations

Each modality uses 10 repeats of five-fold outer validation. Within every
training fold, the code performs candidate screening, reconstructed IAMB
selection, imputation/scaling, and Cox fitting. It compares clinical-only,
selected modality-only, and clinical plus selected modality models.

Repeated-split distributions describe algorithmic variability, not sampling
confidence intervals. The script checkpoints after every fold and resumes
automatically.

## M8.42

Maps the historical TCGA-selected panels to gene symbols, restricts comparisons
to features that METABRIC could assay, and reports exact overlap, Jaccard,
overlap coefficient, assayability, and a hypergeometric overlap test.

Mutation comparison is explicitly restricted to METABRIC_173. Methylation uses
the existing annotated TCGA probe table and reports mapping coverage.

## M8.43

Downloads the current Reactome pathway GMT from the official Reactome endpoint,
caches and hashes it, and performs background-aware over-representation
analysis. Concordance is summarized using pathway-score rank correlation and
top-20 pathway Jaccard. Pathways never modify the prediction models.

## M8.44

Creates manuscript-ready summary tables, figures, hashes, and guarded wording
that separates biological recurrence from incremental clinical utility.

Single log:

```text
results/logs/metabric_stage_m8_modality_concordance_YYYYMMDD_HHMMSS.log
```
