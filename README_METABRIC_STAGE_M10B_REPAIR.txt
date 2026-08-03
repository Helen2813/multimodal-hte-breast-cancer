METABRIC M10B REPAIR 1
========================

Reason
------
The 2000-patient bootstraps completed for every Track B analysis. The run then
failed only while discovering the already-existing M7 Track A paired-delta
summary. M7 uses the column name `model_set`; the original M10B discovery code
incorrectly required `model`.

This repair
-----------
- changes only Track A summary-file discovery;
- accepts and normalizes `model_set`;
- does not alter any bootstrap calculation;
- resumes from the completed checkpoint files;
- does not rerun M1-M9;
- does not repeat feature selection or fit a model.

Install
-------
Extract this ZIP into the project root and allow it to replace:

scripts\m51_metabric_m10b_npi_benchmark.py

Run
---
.\run_metabric_stage_m10b_npi_benchmark.ps1

The script will report that each completed bootstrap is being resumed from
2000/2000 and will proceed directly to summaries and figures.

Return
------
Upload the new single transcript log.
