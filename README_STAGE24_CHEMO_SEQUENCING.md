# Paper A Stage 24 — chemotherapy sequencing audit

Run from the project root:

```powershell
.\run_stage24_chemo_sequencing_audit.ps1
```

This stage is diagnostic only. It does not modify Candidate V9, rerun the
publication bootstrap, add an ever-chemotherapy flag to the adjustment set, fit a
new treatment-effect model, or generate manuscript text.

It reconstructs chemotherapy and hormone-treatment timing from
`data/processed/01_Clinical/clinical.tsv`, compares sequencing patterns between the
day-180 treatment strategies, and reports the sizes/events of possible later
sensitivity populations.

Patient identifiers appear only in:

```text
results/tables/stage24_chemo_sequencing/s24_patient_sequence_registry_LOCAL_ONLY.csv
```

Single log:

```text
results/logs/stage24_chemo_sequencing_audit_YYYYMMDD_HHMMSS.log
```
