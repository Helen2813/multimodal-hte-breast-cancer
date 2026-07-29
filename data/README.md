# Data directory

Raw TCGA, METABRIC, or controlled-access data must not be committed.

Expected table format:

- one row per patient;
- first/shared key column: `patient_id` (configurable);
- clinical table includes survival time, event indicator, treatment indicators, and baseline confounders;
- every modality table contains the patient identifier plus feature columns only.

The Paper 1 candidate panels should be exported from the exact published-code commit and imported using `scripts/import_frozen_panels.py`. The generated manifest records source repository, source commit, file size, and SHA-256 digest.
