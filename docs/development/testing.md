# Testing

## Purpose
To explain how to validate changes made to the codebase.

## Audience
Developers and Maintainers.

## Summary
The project uses `pytest` for unit testing and repository audit checks. A syntax smoke test is also available using `compileall`.

## Details

### Default Test Suite
Run all tests:
```bash
pytest
```

### Targeted Tests
Run only unit tests or audit tests for faster feedback:
```bash
pytest tests/unit -q
pytest tests/audit -q
pytest tests/unit/test_navier_stokes_pipeline.py -q
```
*Note: Audit tests check repository hygiene, documentation entry points, tracked generated artifacts, and compatibility imports.*

### Syntax and Import Smoke Check
Catch syntax errors without running logic:
```bash
python -m compileall src tests examples scripts
```

### Dataset-Dependent Checks
Some scripts under `scripts/` and `examples/applications/` need local datasets. These will not pass in a clean clone unless the required data is available under `data/`. When adding such checks, always document the required input paths and ensure generated outputs are written to ignored paths like `results/`.

## Examples
N/A

## Validation
To verify testing is set up correctly:
```bash
pytest tests/unit -q
```
Expected result: The unit tests pass seamlessly.

## Related docs
- [Contribution Guide](contribution-guide.md)
