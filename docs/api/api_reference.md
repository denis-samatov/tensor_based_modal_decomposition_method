# API Reference

This reference covers the primary public classes used by examples and tests. It is not generated automatically, so check source docstrings for implementation-level details.

## Configuration

### `BaseConfig`

Base configuration shared by most TBMD components.

```python
from TBMD.config import BaseConfig

config = BaseConfig(
    backend="pytorch",
    dtype="float32",
    device=None,
    seed=0,
    deterministic=True,
    verbose=True,
)
```

### `DecompositionConfig`

Configuration for Tucker/HOSVD decomposition.

```python
from TBMD.config import DecompositionConfig

config = DecompositionConfig(
    ranks=[20, 20, 10],
    method="hosvd",
    energy_threshold=0.99,
    normalize=False,
)
```

### `SensorPlacementConfig`

Configuration for tensor QR sensor placement.

```python
from TBMD.config import SensorPlacementConfig

config = SensorPlacementConfig(
    n_sensors=30,
    uniform_distribution=False,
    check_orthogonality=False,
)
```

### `CompressiveSensingConfig`

Core ADMM configuration for TBMD compressive sensing reconstruction.

```python
from TBMD.config import CompressiveSensingConfig

config = CompressiveSensingConfig(
    max_iter=1000,
    tol=1e-4,
    epsilon_l1=1e-2,
    delta_init=1.0,
    delta_max=1.0,
    relax_lambda=0.95,
    device="cpu",
    dtype="float32",
)
```

### `DigitalTwinConfig`

Configuration for the digital twin orchestration class.

```python
from TBMD.config import DigitalTwinConfig

config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_temporal_modes=20,
    n_sensors=30,
    forecaster_type="linear",
)
```

## Decomposition

### `TuckerDecomposer`

Performs Tucker/HOSVD decomposition and reconstruction.

```python
from TBMD.core.decomposition.hosvd import TuckerDecomposer

decomposer = TuckerDecomposer(tensors=data, config=config)
decomposer.decompose()
decomposer.reconstruct()

core = decomposer.cores
factors = decomposer.factors
reconstructed = decomposer.reconstructed_tensors
```

## Sensor Placement

### `TensorTubeQRDecomposition`

Selects sensor locations using a tensor tube QR factorization workflow.

```python
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition

placer = TensorTubeQRDecomposition(tensor=A_tensor, config=sensor_config)
P, Q, R = placer.factorize()
```

`P` is the sensor mask used by reconstruction code.

## Reconstruction

### `TensorCompressiveSensing`

Solves for modal coefficients from sparse measurements.

```python
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

reconstructor = TensorCompressiveSensing(
    A=A_tensor,
    P=P,
    Y=Y,
    core_cfg=cs_config,
)

x_hat, metrics = reconstructor.solve()
```

## Digital Twin

### `DigitalTwin`

Coordinates decomposition, modal processing, sensor placement, reconstruction, and forecasting.

```python
from TBMD.digital_twin.digital_twin import DigitalTwin

twin = DigitalTwin(config)
summary = twin.train(historical_data, normalize=False)
forecast = twin.predict(current_state, n_steps=10)
```

Common methods:

- `train(historical_data, normalize=False, ranks=None)`: fit decomposition, sensor placement, and forecaster components.
- `predict(current_state, n_steps=1, return_full_field=True)`: forecast future states.
- `update_from_sensors(sensor_readings, timestamp=None)`: reconstruct state from sensor readings.
- `get_sensor_locations()`: return selected sensor indices.
- `get_statistics()`: return current model and monitoring state.

## Forecasting

The package includes linear, MLP, LSTM, latent modal, and multi-resolution forecasters under `TBMD.core.forecasting`. The Navier-Stokes experiment layer also provides registry helpers in `TBMD.experiments`.

## Utilities

```python
from TBMD.core.utils.misc import get_torch_device, reconstruct_tensor, to_torch_tensor
```

These helpers centralize tensor conversion, device selection, seeding, and modal reconstruction utilities.
