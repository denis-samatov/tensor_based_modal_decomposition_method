# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before Opening a Pull Request

Run the relevant checks:

```bash
pytest
python -m compileall src tests examples scripts
```

For focused changes, run the closest unit test file in addition to the full suite when practical.

For documentation or repository-structure changes, also run:

```bash
pytest tests/audit -q
```

## Contribution Guidelines

- Keep algorithmic changes separate from documentation or cleanup changes.
- Do not commit local datasets, virtual environments, caches, `.env` files, generated plots, generated metrics, or model artifacts.
- Document new scripts with required inputs, outputs, and whether they need local data.
- Add tests for public API behavior, shape contracts, and bug fixes.
- Avoid unsupported performance or accuracy claims unless the repository includes a reproducible command and dataset description.
- Keep documentation links case-correct so they work on GitHub and Linux file systems.

## Code Style

The project does not currently define a formatter or linter configuration. Follow the surrounding style, keep public docstrings clear, and avoid broad refactors unless they are needed for the change.
