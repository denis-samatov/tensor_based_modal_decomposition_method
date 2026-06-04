# Configuration Guide

TBMD configuration is primarily defined through dataclasses in `TBMD.config`.

## Common Pattern

```python
from TBMD.config import DecompositionConfig

config = DecompositionConfig(
    ranks=[20, 20, 10],
    device="cpu",
    dtype="float32",
    verbose=True,
)
```

## Environment Variables

The core package does not require environment variables. `.env.example` documents optional local conventions such as data and output directories.

Do not commit `.env` files.

## Main Configuration Classes

| Class | Purpose |
| --- | --- |
| `BaseConfig` | Shared backend, dtype, device, seed, and logging options. |
| `DecompositionConfig` | Tucker/HOSVD decomposition settings. |
| `SensorPlacementConfig` | Tensor QR sensor placement settings. |
| `CompressiveSensingConfig` | ADMM-based reconstruction settings. |
| `DigitalTwinConfig` | Digital twin orchestration settings. |
| `ForecasterConfig` and subclasses | Forecasting model settings. |

## Reproducibility

Most configs inherit `seed` and `deterministic` from `BaseConfig`. Deterministic behavior can still depend on PyTorch backend support and installed package versions.

## Local Data Paths

Datasets are expected under `data/` by convention, but scripts may define their own paths. Keep dataset paths configurable when adding new scripts.
