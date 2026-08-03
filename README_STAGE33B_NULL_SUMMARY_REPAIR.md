# Paper A Stage 33B v2 — null-scenario summary repair

Extract into the project root and allow these Stage 33B files to replace the
previous failed Stage 33B package. Then run:

```powershell
.\run_stage33b_null_summary_repair.ps1
```

## Why the first Stage 33B audit failed

During Stage 33, the checkpoint append function repeatedly re-read the CSV with
the default pandas NA parser. The valid string `null` was converted to `NaN`,
then written back as an empty CSV field. Therefore `keep_default_na=False`
correctly preserved the empty cells, but could not recover the lost literal.

The locked `scenario_id` still contains the complete regime:

- `_NULL`
- `_EMPIRICALLY_CALIBRATED_BENEFIT`

Stage 33B v2 deterministically reconstructs `effect_regime` from that suffix.

## What this package does

1. Reruns no simulation repetition.
2. Verifies 3,600 unique checkpoint method-runs.
3. Reconstructs exactly 1,800 null and 1,800 benefit rows from `scenario_id`.
4. Rejects any non-empty effect label that contradicts its scenario ID.
5. Rebuilds all 36 scenario-method summaries.
6. Reapplies the original gates unchanged.
7. Reports true-null coverage and positive CI-exclusion rates.
8. Leaves all original Stage 33 outputs untouched.

The detailed row-level reconstruction audit is LOCAL_ONLY.
