# METABRIC Stage M5

Run from the project root:

```powershell
.\run_metabric_stage_m5_protocol_lock.ps1
```

M5 performs no model fitting. It:

1. Corrects the Paper-1 recipe audit by classifying `02_CNV` explicitly as CNV.
2. Recovers the official `METABRIC_173` gene panel from the cBioPortal API,
   caches the response, and creates panel-aware mutation coding.
3. Separates high-confidence primary RNA/CNA mappings from display-name-only
   sensitivities and unavailable features.
4. Audits Paper-1/TCGA and METABRIC endpoint semantics and cohort sizes.
5. Locks the dual-track protocol before any METABRIC feature-selection or
   prognostic modeling.

The two tracks remain separate:

- Track A: fixed TCGA-selected panel external transport.
- Track B: independent nested Paper-1 feature-selection replication in METABRIC.

Single log:

```text
results/logs/metabric_stage_m5_protocol_lock_YYYYMMDD_HHMMSS.log
```
