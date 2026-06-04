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
