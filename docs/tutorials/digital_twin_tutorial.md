# Tutorial: Build a Digital Twin From Synthetic Data

This tutorial trains the TBMD digital twin workflow on a synthetic pressure-like field and runs a short forecast.

## Prerequisites

- Python 3.10 or newer.
- Dependencies installed with `python -m pip install -e ".[dev]"`.
- Basic familiarity with PyTorch tensors.

## 1. Generate Synthetic Data

```python
import numpy as np
import torch


def generate_synthetic_field(nx=50, ny=50, nt=100):
    """Generate a simple damped wave field."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 10, nt)

    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    radius = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    field = np.exp(-0.1 * T) * np.sin(10 * radius - T)

    return torch.from_numpy(field).float()


data = generate_synthetic_field()
print(data.shape)
```

## 2. Configure the Digital Twin

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=15,
    forecaster_type="linear",
    device="cpu",
    verbose=True,
)

twin = DigitalTwin(config)
```

## 3. Train

```python
train_data = data[..., :80]
test_data = data[..., 80:]

summary = twin.train(
    historical_data=train_data,
    normalize=False,
)

print(summary["n_sensors"])
```

## 4. Forecast

```python
current_state = train_data[..., -1]
forecast = twin.predict(current_state=current_state, n_steps=20)

print(forecast.shape)
```

## 5. Inspect Forecast Error

```python
import matplotlib.pyplot as plt

predicted_field = forecast[..., -1]
true_field = test_data[..., -1]
absolute_error = torch.abs(predicted_field - true_field)

plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.title("Forecast")
plt.imshow(predicted_field)
plt.colorbar()

plt.subplot(132)
plt.title("Reference")
plt.imshow(true_field)
plt.colorbar()

plt.subplot(133)
plt.title("Absolute Error")
plt.imshow(absolute_error)
plt.colorbar()

plt.tight_layout()
plt.show()
```

## 6. Update From Sensor Measurements

```python
sensor_mask = twin.sensor_mask
real_field = test_data[..., 10]
sensor_readings = real_field[sensor_mask]

update_result = twin.update_from_sensors(
    sensor_readings=sensor_readings,
    timestamp=90.0,
)

print(update_result["alert_status"])
```

## Next Steps

- Change `n_spatial_modes` and `n_sensors` and compare reconstruction metrics.
- Try `forecaster_type="lstm"` on a small dataset before scaling up.
- Read [Geometry-Aware TBMD](../guides/geometry_aware_tbmd.md) if the spatial domain has irregular connectivity.
