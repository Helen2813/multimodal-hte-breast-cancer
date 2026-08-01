# Stage 25 Candidate V10 review-bundle builder

Copy both files to the project root and run:

```powershell
.\make_stage25_v10_review_bundle.ps1
```

The script does not fit models and does not modify Candidate V9.

It creates:

```text
results\review_bundles\stage25_v10_review_bundle_YYYYMMDD_HHMMSS.zip
```

Upload that ZIP. It contains the exact V9 estimator/configuration, Stage 24
diagnostics, and small schema samples needed to write Candidate V10 without
guessing function names or estimator interfaces.
