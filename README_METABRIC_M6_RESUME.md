# METABRIC M6 repair: resume from M33

The original M6 run completed M31 and M32, then stopped because `master_outer.csv`
already contained an `OS.time`-named field and the outcome merge created suffixed
columns. The script subsequently requested a column that no longer existed.

This patch:

- resumes from M33 without rerunning M31 or M32;
- resolves the TCGA patient/sample ID pair by maximum verified overlap;
- renames outcome fields before merging, preventing `OS.time_x`/`OS.time_y`;
- fixes the stage-II/III prefix collision in clinical harmonization;
- uses fast concordance computation and paired external bootstrap samples;
- builds Track B with all 173 panel-aware mutation genes rather than only GPS2;
- uses nonsynonymous mutation calls for the primary pilot;
- keeps the 20k RNA and 22k CNA matrices separate and screens them in chunks;
- adds a fold-matched clinical-only comparator;
- keeps Track B labelled as reconstructed because M32 did not reproduce the
  historical TCGA feature sets at the locked threshold.

Extract into the project root and run:

```powershell
.\run_metabric_stage_m6_resume_from_m33.ps1
```

Single log:

```text
results/logs/metabric_stage_m6_resume_from_m33_YYYYMMDD_HHMMSS.log
```
