METABRIC STAGE M11 RFS - REPAIR 1
=================================

Reason
------
The protocol-lock script loaded only metabric_m7_config.json. The multimodal
settings live there, but the modality-specific settings live in
metabric_m8_config.json. This caused:

KeyError: 'modality_analysis'

Repair
------
This patch changes only the protocol-lock script. It now:
- loads multimodal Track B settings from metabric_m7_config.json;
- loads modality-specific settings from metabric_m8_config.json;
- verifies the locked 20x5 and 10x5 designs before any model is fitted;
- hashes both configuration files in the protocol manifest.

No RFS model or fold was run before the failure, so there is nothing to delete
or recompute.

Install
-------
Extract this ZIP into the project root and allow replacement of:

scripts\m51_lock_rfs_sensitivity_protocol.py

Run
---
From the same PowerShell session:

.\run_metabric_stage_m11_rfs.ps1

If a new PowerShell window is used, first run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Return
------
Upload the new M11 transcript log after the run completes or if a new error
appears.
