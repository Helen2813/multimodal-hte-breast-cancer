# METABRIC Stage M7: full core analysis

Run from the project root:

```powershell
.\run_metabric_stage_m7_full_core_analysis.ps1
```

M7 scales only the analyses that passed the M6 computational pilot.

## Track A

- trains the four locked Cox models in TCGA;
- evaluates them without refitting in METABRIC;
- verifies that the first 200 paired bootstrap draws reproduce M6;
- extends the paired patient bootstrap to 1,000 repetitions;
- reports Harrell C, 5- and 10-year binary AUC, Uno C at 10 years,
  IPCW dynamic AUC, IPCW Brier scores, integrated Brier score,
  external calibration slope, and observed-minus-predicted survival.

## Track B

- remains explicitly labelled as a reconstructed dependency-aware replication;
- runs 20 deterministic repeats of five-fold nested validation;
- repeats RNA/CNA screening, mutation-frequency filtering, imputation,
  scaling, reconstructed IAMB selection, and Cox fitting inside every
  training fold;
- compares every fold with a matched clinical-only model;
- checkpoints after every repeat/fold;
- reports repeat-level OOF performance, algorithmic variability,
  feature frequencies, modality composition, within-repeat stability,
  and stability of repeat-level consensus sets.

The repeated-split quantiles are algorithmic variability summaries, not sampling
confidence intervals.

Methylation-specific and pathway-level replication are deliberately left for M8.

Single log:

```text
results/logs/metabric_stage_m7_full_core_analysis_YYYYMMDD_HHMMSS.log
```
