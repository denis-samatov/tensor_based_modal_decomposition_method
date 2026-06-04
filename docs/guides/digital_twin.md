# Digital Twin Guide

## Overview

The TBMD digital twin layer coordinates several repository components:

- Tucker/HOSVD decomposition.
- Modal tensor processing.
- Tensor QR sensor placement.
- Compressive sensing reconstruction.
- Modal-space forecasting.
- Optional proxy-model hooks for scenario analysis.

The implementation is an experimental orchestration layer. Validate the workflow, data assumptions, and forecast quality for each dataset before using results in a technical report.

## Architecture

```text
historical data
    -> decomposition
    -> modal tensor processing
    -> sensor placement
    -> forecaster training
    -> prediction and sensor updates
```

Primary class:

```python
from TBMD.digital_twin.digital_twin import DigitalTwin
```

Primary configuration:

```python
from TBMD.config import DigitalTwinConfig
```

## Minimal Example

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=15,
    forecaster_type="linear",
    device="cpu",
)

twin = DigitalTwin(config)
summary = twin.train(historical_data, normalize=False)
forecast = twin.predict(current_state, n_steps=5)
```

## Training Inputs

`DigitalTwin.train()` accepts a tensor or a dictionary of tensors. The expected layout is spatial dimensions followed by time:

```text
(x, y, time)
(x, y, z, time)
(active_cells, time)
```

If preprocessing or normalization is required, prefer doing it explicitly before calling `train()` and pass `normalize=False` unless the code path is being tested.

## Forecasting

`predict()` projects the current state into modal space, advances modal coefficients with the configured forecaster, and reconstructs the predicted full field when `return_full_field=True`.

```python
forecast = twin.predict(
    current_state=current_state,
    n_steps=10,
    return_full_field=True,
)
```

## Sensor Updates

After training, `update_from_sensors()` can reconstruct a field from sparse sensor readings using the selected sensor mask.

```python
result = twin.update_from_sensors(
    sensor_readings=sensor_readings,
    timestamp=90.0,
)
```

Check the returned metrics and `alert_status` before consuming the reconstructed field.

## Forecaster Options

The configuration currently supports:

- `linear`
- `mlp`
- `lstm`
- `persistence`

Use simple forecasters for smoke tests and small synthetic examples. More complex forecasters require dataset-specific validation.

## Practical Guidance

- Start with synthetic or reduced data and run the unit tests before using local datasets.
- Keep train/test splits explicit and avoid mixing trajectories into one artificial time axis.
- Store generated artifacts under ignored output directories such as `results/` or `scripts/plots/`.
- Do not commit local datasets, private simulator outputs, or trained model artifacts.

## Related Files

- `examples/digital_twin/01_digital_twin_basic.py`
- `examples/digital_twin/02_digital_twin_advanced.py`
- `examples/digital_twin/04_digital_twin_type_demo.py`
- `docs/tutorials/digital_twin_tutorial.md`
