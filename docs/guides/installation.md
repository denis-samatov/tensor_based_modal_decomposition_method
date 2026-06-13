# Installation Guide

This guide describes a reproducible local development installation for TBMD.

## Requirements

- Python 3.10 or newer. The repository currently records Python 3.12.7 in `.python-version`.
- A local virtual environment.
- Enough disk space for optional datasets and generated experiment outputs.

The package does not require external service credentials for the core examples and tests.

## Editable Development Install

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the package from `src/` and the development dependencies declared in `pyproject.toml`.

## Requirements File Install

Use `requirements.txt` when you need the pinned dependency set used by this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file is intended for local development and experiments. It is not a lockfile with hashes.

## Verify Installation

```bash
python -c "import TBMD; print(TBMD.__version__)"
pytest tests/audit -q
```

For a broader check, run:

```bash
pytest
python -m compileall src tests examples scripts
```

## Optional Local Data

Dataset-dependent scripts expect files under `data/` by convention. The `data/` directory is ignored by git and is not required for the core package tests.

Do not commit local datasets, generated results, model checkpoints, or `.env` files.
