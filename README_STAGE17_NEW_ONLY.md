# Stage 17 — new analyses only

This package does not rerun Stages 15 or 16 and skips the Stage 66 reconstruction script.
It performs only the new analyses:

- Stage 67: patient influence and original fold-2 forensics;
- Stage 68: prespecified nuisance-partition stability experiment;
- Stage 69: repeated-score aggregation;
- Stage 70: decision gate.

The repeated partitions in Stage 68 are the scientific stability experiment required by the
Stage 16 fold-instability finding; they are not a rerun of Stage 16.

Copy the package contents into the project root and run:

```powershell
.\run_stage17_new_analysis_once.ps1
```

The runner is one-time only. It stops if Stage 17 checkpoint or final-output files already exist,
so results from separate runs cannot be mixed accidentally.

All summaries are printed to the terminal and captured in one transcript:

```text
results\logs\stage17_new_analysis_once_YYYYMMDD_HHMMSS.log
```

No publication bootstrap is started.
