# Repository Structure

This page explains the main repository areas and what should be committed to each one.

## Tracked Source and Documentation

| Path | Purpose |
| --- | --- |
| `src/TBMD/` | Python package source code. |
| `src/TBMD/core/` | Decomposition, reconstruction, forecasting, sensor placement, geometry, metrics, and data utilities. |
| `src/TBMD/config/` | Dataclass configuration objects and backward-compatible constants. |
| `src/TBMD/digital_twin/` | Digital twin orchestration layer. |
| `src/TBMD/modules/` | Compatibility wrappers for legacy import paths. |
| `src/TBMD/experiments/` | Experiment-specific model registries, forecasting helpers, and runners. |
| `src/TBMD/visualization/` | Plotting and visualization helpers. |
| `examples/` | Runnable examples grouped by topic. |
| `scripts/` | Evaluation, tuning, diagnostics, and artifact-generation scripts. |
| `tests/` | Unit and audit tests. |
| `docs/` | User, developer, API, and experiment documentation. |

## Local-Only Artifacts

The following paths are ignored by git and should remain local unless maintainers explicitly decide otherwise:

| Path or pattern | Reason |
| --- | --- |
| `data/` | Local datasets can be large, private, or externally licensed. |
| `results/` | Generated experiment outputs. |
| `scripts/plots/` | Generated plots, metrics, summaries, checkpoints, and visualizations. |
| `.env` and `.env.*` | Local environment settings and potential secrets. |
| `.venv/`, `venv/` | Local Python environments. |
| `*.npy`, `*.npz`, `*.h5`, `*.pth`, `*.joblib` | Dataset, checkpoint, and intermediate artifact formats. |

Commit generated artifacts only when they are intentionally curated documentation assets, small enough for normal review, and free of local paths or sensitive data.

## Legacy and Compatibility Areas

The repository keeps compatibility modules and examples for historical import paths. Prefer the `TBMD.core` and `TBMD.config` APIs for new code, but do not remove compatibility files without a migration note and tests.

## Adding New Files

When adding a new script or example, document:

- required inputs;
- expected command;
- output directory;
- whether it needs local data;
- whether it is suitable for CI.

Keep algorithmic changes separate from documentation-only cleanup when practical.
