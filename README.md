# Tensor-Based Modal Decomposition Method

## Overview

Tensor-Based Modal Decomposition Method (TBMD) is a Python research library for reduced-order modeling of spatiotemporal tensor data. The repository includes implementations for Tucker/HOSVD decomposition, modal tensor processing, sensor placement, field reconstruction from sparse measurements, geometry-aware variants, and experimental digital twin workflows.

The code is aimed at scientific computing and reservoir-modeling experiments. It should be treated as a research and engineering codebase rather than a validated production simulator.

## Features

- Tucker/HOSVD decomposition for tensor data.
- Modal tensor processing utilities for building reduced bases.
- Tensor QR-based sensor placement.
- Compressive sensing reconstruction with ADMM-based solvers.
- Geometry-aware decomposition, reconstruction, and sensor placement helpers.
- Digital twin orchestration that combines decomposition, sensor placement, reconstruction, and forecasting components.
- Navier-Stokes forecasting experiments and model comparison scripts.
- Pytest-based unit and audit tests.

## Project Structure

```text
tensor-based-modal-decomposition-method/
├── src/TBMD/              # Python package source code
│   ├── config/            # Dataclass configuration objects
│   ├── core/              # Decomposition, forecasting, reconstruction, geometry, and data utilities
│   ├── digital_twin/      # Digital twin orchestration layer
│   ├── experiments/       # Experiment-specific model registries and runners
│   ├── modules/           # Compatibility layer for legacy module paths
│   └── visualization/     # Plotting helpers
├── examples/              # Runnable examples grouped by topic
├── scripts/               # Evaluation, tuning, and diagnostic scripts
├── tests/                 # Unit and audit tests
├── docs/                  # User and developer documentation
├── data/                  # Local datasets; ignored by git
└── results/               # Generated outputs; ignored by git
```

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/denis-samatov/tensor_based_modal_decomposition_method.git
cd tensor_based_modal_decomposition_method

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If editable installation is not needed, the pinned dependency file can be used instead:

```bash
python -m pip install -r requirements.txt
```

## Configuration

The core package does not require secrets or external service credentials. Runtime options are primarily passed through dataclass configuration objects in `TBMD.config`.

An optional `.env.example` file is provided to document local paths and logging preferences. Keep local `.env` files out of git.

## Usage

Minimal Tucker decomposition example:

```python
import torch

from TBMD.config import DecompositionConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposer

data = torch.randn(64, 64, 20)
config = DecompositionConfig(ranks=[16, 16, 8], verbose=True)

decomposer = TuckerDecomposer(tensors=data, config=config)
decomposer.decompose()
decomposer.reconstruct()

core = decomposer.cores
factors = decomposer.factors
reconstructed = decomposer.reconstructed_tensors
```

Digital twin example:

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=15,
    forecaster_type="linear",
)

twin = DigitalTwin(config)
summary = twin.train(historical_data, normalize=False)
forecast = twin.predict(current_state, n_steps=5)
```

More examples are available under `examples/` and `docs/guides/quick_start.md`.

## Testing

Run the default test suite:

```bash
pytest
```

Useful targeted checks:

```bash
pytest tests/unit -q
pytest tests/audit -q
python -m compileall src tests examples scripts
```

Some experiment scripts require local datasets under `data/` and are not suitable as lightweight CI checks.

## Documentation

- [Documentation index](docs/README.md)
- [Quick start guide](docs/guides/quick_start.md)
- [API reference](docs/api/api_reference.md)
- [Configuration guide](docs/guides/configuration.md)
- [Testing guide](docs/guides/testing.md)
- [Model and data guide](docs/guides/data_and_models.md)

## Security and Privacy

Do not commit secrets, credentials, private datasets, generated model artifacts, or local environment files. The repository ignores `.env`, virtual environments, caches, local datasets, and generated results by default.

Report security concerns using the process in `SECURITY.md`.

## Roadmap / TODO

- Add continuous integration for tests and packaging.
- Add a lightweight documentation link checker.
- Define a reproducible benchmark protocol for any public accuracy or performance claims.

## License

MIT License. See `LICENSE`.
