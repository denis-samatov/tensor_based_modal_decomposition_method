# Experiment Runbook

## Purpose
Provides a standard procedure for running and logging numerical experiments reproducibly.

## Audience
Researchers running end-to-end benchmarks.

## Summary
Experiments should be run via standardized scripts, logging configuration metadata alongside the results, rather than relying on interactive Jupyter notebooks.

## Details
### 1. Preparation
- Ensure local datasets are placed in `data/` and not tracked by git.
- Verify that `TBMD.config` parameters are correctly set for the experiment.

### 2. Execution
Run the experiment script:
```bash
python examples/applications/brugge_field/run_brugge_enhanced.py
```

### 3. Artifact Logging
- Ensure the script outputs the trained decomposition basis and forecast results to a timestamped folder within `results/`.
- Save the exact configuration parameters (JSON or YAML) alongside the numerical outputs to guarantee reproducibility.

## Related docs
- [Digital Twin Analysis](../product/brugge-digital-twin-analysis.md)
