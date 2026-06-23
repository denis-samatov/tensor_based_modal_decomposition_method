# Configuration

## Purpose
To detail how TBMD is configured at runtime.

## Audience
Developers and users initializing TBMD models or running experiments.

## Summary
TBMD avoids hidden state and environment variables for core logic. Instead, configuration is explicitly defined via Python dataclasses located in `TBMD.config`.

## Details

### Main Configuration Classes
| Class | Purpose |
| --- | --- |
| `BaseConfig` | Shared backend, dtype, device, seed, and logging options. |
| `DecompositionConfig` | Tucker/HOSVD decomposition settings (e.g., target ranks). |
| `SensorPlacementConfig` | Tensor QR sensor placement settings (e.g., number of sensors). |
| `CompressiveSensingConfig` | ADMM-based reconstruction settings (e.g., iterations, tolerance). |
| `DigitalTwinConfig` | Orchestrator settings uniting spatial/temporal mode counts and forecaster type. |
| `ForecasterConfig` | Forecasting model parameters. |

### Reproducibility
Most configuration classes inherit `seed` and `deterministic` flags from `BaseConfig`. Setting these ensures deterministic operations across runs, though behavior may still depend slightly on the specific PyTorch backend and hardware used.

## Examples
**Instantiating a configuration object:**
```python
from TBMD.config import DecompositionConfig

config = DecompositionConfig(
    ranks=[20, 20, 10],
    device="cpu",
    dtype="float32",
    verbose=True,
)
```

## Validation
Ensure that configurations are type-correct by running the unit tests:
```bash
pytest tests/unit -q
```
Expected result: Tests pass, confirming that the default initialization of configuration objects is valid and properly processed by the core algorithms.

## Related docs
- [Environment Variables](environment-variables.md)
- [Architecture Components](../architecture/components.md)
