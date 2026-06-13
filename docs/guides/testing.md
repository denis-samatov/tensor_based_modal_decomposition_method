# Testing Guide

## Default Test Suite

```bash
pytest
```

## Targeted Tests

```bash
pytest tests/unit -q
pytest tests/audit -q
pytest tests/unit/test_navier_stokes_pipeline.py -q
```

The audit tests check repository hygiene, documentation entry points, tracked generated artifacts, and compatibility imports.

## Syntax and Import Smoke Check

```bash
python -m compileall src tests examples scripts
```

## Dataset-Dependent Checks

Some scripts under `scripts/` and `examples/applications/` need local datasets. These are not expected to pass in a clean clone unless the required data is available under `data/`.

When adding a dataset-dependent check, document:

- required input paths;
- expected command;
- expected outputs;
- whether the command is suitable for CI.

Generated outputs from these checks should be written to ignored paths such as `results/` or `scripts/plots/`.
