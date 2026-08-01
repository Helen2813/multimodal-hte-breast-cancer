# METABRIC Stage M6

Run from the project root:

```powershell
.\run_metabric_stage_m6_nested_pilot.ps1
```

M6 has five parts:

1. Verify the M5 protocol lock and search for historical METABRIC artifacts.
2. Validate a reconstructed Python IAMB implementation against the historical TCGA
   feature lists. Engine choice uses TCGA history only, never METABRIC outcomes.
3. Run Track A: train fixed-panel Cox models on TCGA and evaluate unchanged models
   in METABRIC, with a 200-repetition external bootstrap pilot.
4. Run Track B: one five-fold nested METABRIC pilot matching the historical
   `min200` logic:
   - 200 RNA,
   - 200 CNA,
   - panel-aware mutation candidates,
   - nine clinical variables,
   - final IAMB at alpha 0.20,
   - penalized Cox at 0.05.
5. Report selection stability, external performance, and the decision about scaling.

Important differences from the historical thesis analysis:

- all supervised filtering is inside the outer training fold;
- the test fold is never used for feature screening or MB discovery;
- the pilot does not claim that Markov Blanket features are experimentally proven causes;
- the historical benchmark (C-index 0.7197, 5-year AUC 0.7639, 87 features)
  is used only as a reference, not as a target to optimize against.

Dependency:

```powershell
python -m pip install lifelines
```

The runner will stop before modeling if `lifelines` is unavailable.

Single log:

```text
results/logs/metabric_stage_m6_nested_pilot_YYYYMMDD_HHMMSS.log
```
