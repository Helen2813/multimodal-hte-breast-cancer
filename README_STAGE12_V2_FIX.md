# Stage 12 v2 — exact Stage 30 propensity compatibility

The candidate estimate used OOF propensity scores saved by Stage 30. Stage 12 v1 fitted a new propensity model during replication.

- Replication loads saved Stage 30 scores.
- Bootstrap still refits propensity.
- Bootstrap uses the exact Stage 30 C grid, L2/LBFGS, and up to four inner folds.
- The weight-file hash is included in checkpoint compatibility.

Run `./run_stage12_v2_pilot.ps1`.
