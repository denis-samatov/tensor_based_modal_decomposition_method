# Tensor-Based Modal Decomposition Method

A Python research library for reduced-order modeling of spatiotemporal tensor data. 

## What this project does
Tensor-Based Modal Decomposition Method (TBMD) compresses high-dimensional spatiotemporal data (such as computational fluid dynamics or reservoir-modeling datasets) into a compact modal representation. It uses these representations to select optimal sensor placements, reconstruct full fields from sparse measurements, and build digital twin pipelines for forecasting future states.

## Who this is for
- **ML/AI Engineers & Data Scientists**: For building and orchestrating digital twin forecasting pipelines.
- **Scientific Computing Researchers**: For experimenting with tensor decompositions (Tucker/HOSVD) and geometry-aware representations.
- **Developers**: For extending and integrating the core mathematical components into larger simulation workflows.

## Core capabilities
- **Tucker/HOSVD decomposition** for spatiotemporal tensor data.
- **Modal tensor processing** utilities for building reduced bases.
- **Tensor QR-based sensor placement** to find the most informative measurement locations.
- **Compressive sensing reconstruction** with ADMM-based solvers.
- **Geometry-aware variants** for decomposition, reconstruction, and sensor placement on irregular grids.
- **Digital twin orchestration** that unites decomposition, sensor placement, reconstruction, and forecasting.

## Architecture at a glance
The library is composed of modular components built primarily on PyTorch. 
Data flows from `(x, y, time)` tensors through a `Decomposer` to extract modal bases, which are then passed to a `Sensor Placer` to find optimal measurement locations, and optionally into a `Digital Twin` orchestrator which trains a forecaster (e.g., Linear, MLP, LSTM) to predict future states. 

For more details, see the [Architecture Overview](docs/architecture/overview.md).

## Quick start
1. Clone the repository:
```bash
git clone https://github.com/denis-samatov/tensor_based_modal_decomposition_method.git
cd tensor_based_modal_decomposition_method
```
2. Install as an editable package with development dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```
3. Run a basic decomposition script:
```bash
python examples/basic/01_tucker_decomposition.py
```

## Configuration
Configuration is managed strictly through Python dataclasses located in `TBMD.config`, rather than environment variables or external files. See the [Configuration Guide](docs/setup/configuration.md) for details.

## Testing
To verify the installation and run unit tests:
```bash
pytest
```
To run targeted repository hygiene and architecture checks:
```bash
pytest tests/audit -q
```
For more information, see the [Testing Guide](docs/development/testing.md).

### Map of documentation

- **Product & Concepts**: [`docs/product/overview.md`](docs/product/overview.md)
- **Architecture**: [`docs/architecture/overview.md`](docs/architecture/overview.md)
- **Mathematical & Research Pipeline**: [`docs/research-system/reconstruction-pipeline.md`](docs/research-system/reconstruction-pipeline.md)
- **Interfaces & Python Usage**: [`docs/interfaces/python-api.md`](docs/interfaces/python-api.md)
- **Installation & Setup**: [`docs/setup/local-development.md`](docs/setup/local-development.md)
- **Running Experiments**: [`docs/operations/runbook.md`](docs/operations/runbook.md)
- **Contributing & Code Style**: [`docs/development/contribution-guide.md`](docs/development/contribution-guide.md)
- **Operations & Runbooks**: [`docs/operations/runbook.md`](docs/operations/runbook.md)


## Known limitations
This project is an experimental research codebase. Claims regarding accuracy, performance, or "production-readiness" require explicit verification. Local datasets and generated artifacts must not be tracked in version control. See [Limitations](docs/product/limitations.md).

## Contributing
We welcome improvements! Please review the [Contribution Guidelines](CONTRIBUTING.md) before opening a Pull Request.

## License / ownership
MIT License. See `LICENSE`.
