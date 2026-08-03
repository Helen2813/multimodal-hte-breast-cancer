METABRIC STAGE M10A - READ-ONLY SCHEMA PREFLIGHT
=================================================

Purpose
-------
This preflight inspects the exact locked M2/M7/M8/M9 output schemas needed
for the final NPI and calibration extension.

It does NOT:
- rerun M1-M9;
- fit a model;
- repeat feature selection;
- modify locked outputs;
- generate manuscript prose;
- print patient identifier values.

Install
-------
Extract this ZIP into the root of:

C:\Users\olegk\Desktop\multimodal-hte-breast-cancer

Run
---
From the active project PowerShell:

.\run_metabric_stage_m10a_schema_preflight.ps1

Outputs
-------
results\tables\metabric_m10a\
  m50_file_inventory.csv
  m50_column_inventory.csv
  m50_detected_roles.csv
  m50_m10a_readiness.json

One transcript:
results\logs\metabric_stage_m10a_schema_preflight_YYYYMMDD_HHMMSS.log

Return
------
Upload only the single transcript log. The exact M10 NPI/calibration code will
then be written from the real column schemas, without guessing or refitting the
locked M7-M9 analyses.

Repeat protection
-----------------
The runner refuses to overwrite existing M10A outputs. An intentional repeat
of this read-only preflight can be run with:

.\run_metabric_stage_m10a_schema_preflight.ps1 -Force
