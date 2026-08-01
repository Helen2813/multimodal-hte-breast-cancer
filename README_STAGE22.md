# Stage 22: Candidate V9 publication assets

This stage reads the completed and cryptographically locked Candidate V9 analysis. It does not rerun any estimator or bootstrap and does not overwrite the working manuscript.

## Run

From the project root with `.venv` active:

```powershell
.\run_stage22_publication_assets.ps1
```

In a new Windows PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
.\run_stage22_publication_assets.ps1
```

## New analyses

None. Stage 22 is reporting-only.

## Outputs

All outputs are written under:

```text
paper_A_treatment_effects/publication_assets_candidate_v9/
```

The package generates:

- primary-result, interval-sensitivity, convergence, computational, and design-sensitivity tables in CSV, Markdown, and LaTeX;
- publication figures in PNG and SVG;
- Methods, Results, Discussion, Conclusion, and abstract-result snippets;
- an audit of stale exploratory numbers and potential overclaims in local `.tex` files;
- a final report and output inventory;
- before/after verification that the locked Candidate V9 files remain unchanged.

## Interpretation enforced by the generator

- Locked point estimate: 22.951 days.
- Primary 95% percentile patient-bootstrap interval: -4.174 to 91.010 days.
- The interval includes zero.
- The result is directionally positive but statistically imprecise.
- The diagnosis-time CCW analysis is a different-design sensitivity, not a direct numerical replication.
- The fraction of positive bootstrap repetitions is descriptive and is not converted into a p-value or significance claim.
