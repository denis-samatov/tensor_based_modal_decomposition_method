# Contributing

## Purpose
To outline how developers can safely and effectively contribute code, documentation, and tests to the Tensor-Based Modal Decomposition Method (TBMD) repository.

## Audience
External contributors and internal developers.

## Summary
Contributors should use a local virtual environment with an editable installation, avoid committing generated artifacts or local datasets, and run validation checks before opening a pull request.

## Details

### Development Setup
To set up your local environment for active development:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Contribution Guidelines
- **Scope**: Keep algorithmic changes separate from documentation or cleanup changes.
- **Hygiene**: Do not commit local datasets, virtual environments, caches, `.env` files, generated plots, generated metrics, or model artifacts.
- **Documentation**: Document new scripts with required inputs, outputs, and whether they need local data.
- **Testing**: Add tests for public API behavior, shape contracts, and bug fixes.
- **Claims**: Avoid unsupported performance or accuracy claims unless the repository includes a reproducible command and dataset description.
- **Links**: Keep documentation links case-correct so they work on GitHub and Linux file systems.

### Code Style
The project does not currently define a strict formatter or linter configuration. Follow the surrounding style, keep public docstrings clear, and avoid broad refactors unless they are needed for the change.

## Examples
*(See Validation commands below for typical workflows).*

## Validation
Before opening a Pull Request, run the relevant checks:
```bash
# Run the full test suite
pytest

# Ensure syntax is correct
python -m compileall src tests examples scripts
```

For documentation or repository-structure changes, also run:
```bash
pytest tests/audit -q
```
Expected result: All tests pass without warnings.

## Related docs
- [Testing Guide](docs/development/testing.md)
- [Code Style Guide](docs/development/code-style.md)
