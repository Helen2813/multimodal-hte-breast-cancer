# METABRIC Stage M9

Run from the project root:

```powershell
.\run_metabric_stage_m9_final_inference.ps1
```

M9 does not refit the M7/M8 models. It:

1. Locks a final-inference protocol.
2. Runs a 2,000-repetition paired patient bootstrap of the locked repeated
   out-of-fold predictions for the combined reconstructed analysis and all
   four modality-specific analyses.
3. Calibrates raw feature-selection stability against the random overlap
   expected from each pair of folds' candidate-set union.
4. Audits the historical TCGA CpG-to-gene mapping and whether mapped genes
   were assayable in METABRIC promoter-level methylation.
5. Creates a final claim table, figures, LaTeX table, Results/Discussion draft,
   and a separate manuscript scaffold.

The OOF bootstrap is conditional on the fitted repeated models. It is not a
full-pipeline model-refitting bootstrap.

The new manuscript is written to:

```text
paper_cross_cohort_transport
```

The original ITE manuscript assigned to Wael is not modified.

Single log:

```text
results/logs/metabric_stage_m9_final_inference_YYYYMMDD_HHMMSS.log
```
