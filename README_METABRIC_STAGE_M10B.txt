METABRIC STAGE M10B - NPI BENCHMARK AND PUBLICATION FIGURES
================================================================

Purpose
-------
Use the locked Stage M7/M8 repeated out-of-fold risk scores to compare:
- NPI-only;
- the locked clinical-only model;
- the locked clinical+omics model;
- the locked modality-only model where available.

All comparisons use the same NPI-observed patient subset within each analysis.

Scientific boundaries
---------------------
- M1-M9 are not rerun.
- Feature selection is not repeated.
- No prognostic model is refitted.
- No NPI+omics model is introduced.
- Track A is not rerun because its patient-level predictions were not saved.
- The NPI bootstrap is conditional on the locked fitted repeated OOF models.
- A 5-year calibration curve is not reconstructed from relative risk scores
  without saved survival probabilities or baseline hazards.
- Primary M7-M9 conclusions remain unchanged.

Install
-------
Extract this ZIP into:

C:\Users\olegk\Desktop\multimodal-hte-breast-cancer

It adds:
- scripts\m51_metabric_m10b_npi_benchmark.py
- run_metabric_stage_m10b_npi_benchmark.ps1
- README_METABRIC_STAGE_M10B.txt

Run
---
From the active project PowerShell:

.\run_metabric_stage_m10b_npi_benchmark.ps1

Default:
- 2000 paired patient bootstrap repetitions;
- deterministic seeds;
- checkpoint every 100 repetitions;
- automatic resume from partial checkpoints.

The run may take time because Harrell C-index is recalculated across all
repeated OOF predictions in every patient bootstrap sample.

Outputs
-------
results\tables\metabric_m10b\
  m51_m10b_protocol.json
  m51_npi_repeat_metrics.csv
  m51_npi_bootstrap_draws.csv
  m51_npi_bootstrap_summary.csv
  m51_calibration_and_extension_status.csv
  m51_incremental_utility_figure_data.csv
  m51_stability_vs_utility_figure_data.csv
  m51_m10b_report.json
  checkpoints\...

results\figures\metabric_m10b\
  figure2_incremental_utility.png
  figure2_incremental_utility.pdf
  figure4_stability_vs_incremental_utility.png
  figure4_stability_vs_incremental_utility.pdf
  figureS_npi_benchmark.png
  figureS_npi_benchmark.pdf

Return
------
Upload only:

results\logs\metabric_stage_m10b_npi_benchmark_YYYYMMDD_HHMMSS.log

Interrupted run
---------------
Run the same PowerShell command again. Completed bootstrap checkpoints are
resumed automatically and are deterministic.
