# Use Cases and Quick Start

## Purpose
To outline the primary use cases for the TBMD library and provide a quick starting point to run minimal examples.

## Audience
Product readers evaluating the library's capabilities and developers looking to quickly test out core functionality.

## Summary
TBMD supports several use cases: Tucker Decomposition, Sensor Placement, Compressive Sensing Reconstruction, and Digital Twin Forecasting. This document provides minimal working examples for each.

## Details

### 1. Tucker Decomposition
Reduces high-dimensional tensor data into a smaller core tensor and factor matrices.

### 2. Sensor Placement
Finds the most informative locations to place sensors using tensor QR decomposition on the modal basis.

### 3. Reconstruction From Measurements
Reconstructs the full tensor field from sparse sensor measurements.

### 4. Digital Twin Workflow
Combines decomposition, modal processing, sensor placement, and forecasting into a unified pipeline to predict future states.

## Examples

### Digital Twin Example
```python
import torch
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

# Setup synthetic data
data = torch.randn(64, 64, 20)

config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=15,
    forecaster_type="linear",
    verbose=True,
)

twin = DigitalTwin(config)
twin.train(data, normalize=False)

# Predict future states
forecast = twin.predict(data[..., -1], n_steps=5)
print(forecast.shape)
```

## Validation
To run the included example scripts and verify they execute successfully, run:
```bash
python examples/basic/01_tucker_decomposition.py
python examples/basic/02_sensor_placement.py
python examples/basic/03_field_reconstruction.py
python examples/basic/04_complete_pipeline.py
python examples/digital_twin/01_digital_twin_basic.py
```
Expected result: The scripts will execute and print tensor shapes or metrics to the console without errors.

## Related docs
- [Product Overview](overview.md)
- [Local Development Setup](../setup/local-development.md)
