# Quick Start

This guide shows the shortest path from a clean checkout to a working TBMD example.

## Requirements

- Python 3.10 or newer.
- A local virtual environment is recommended.
- PyTorch-compatible CPU installation is enough for the examples below.

## Installation

```bash
git clone https://github.com/denis-samatov/tensor_based_modal_decomposition_method.git
cd tensor_based_modal_decomposition_method

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternative dependency installation:

```bash
python -m pip install -r requirements.txt
```

## 1. Tucker Decomposition

```python
import torch

from TBMD.config import DecompositionConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposer

data = torch.randn(64, 64, 20)
config = DecompositionConfig(ranks=[16, 16, 8], verbose=True)

decomposer = TuckerDecomposer(tensors=data, config=config)
decomposer.decompose()

print(decomposer.cores.shape)
print([factor.shape for factor in decomposer.factors])
```

## 2. Sensor Placement

```python
from TBMD.config import ModalProcessorConfig, ProcessingStrategy, SensorPlacementConfig
from TBMD.core.modal_processor.modes import BatchModalProcessor, ModalTensorStacker
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition

modal_config = ModalProcessorConfig(
    device="cpu",
    processing_strategy=ProcessingStrategy.BATCH,
    return_numpy=False,
)

processor = BatchModalProcessor(modal_config)
stacker = ModalTensorStacker(modal_config)
modal_tensors = processor.process_multiple_subjects(decomposer.cores, decomposer.factors)
A_tensor = stacker.stack_modal_tensors(modal_tensors)

placement_config = SensorPlacementConfig(n_sensors=30, verbose=True)
placer = TensorTubeQRDecomposition(tensor=A_tensor, config=placement_config)
P, Q, R = placer.factorize()

print(P.shape)
```

## 3. Reconstruction From Measurements

```python
from TBMD.config import CompressiveSensingConfig
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

true_field = data[..., -1]
Y = torch.zeros_like(true_field)
Y[P.bool()] = true_field[P.bool()]

reconstructor = TensorCompressiveSensing(
    A=A_tensor,
    P=P,
    Y=Y,
    core_cfg=CompressiveSensingConfig(max_iter=100, tol=1e-4),
)

x_hat, metrics = reconstructor.solve()
print(metrics)
```

## 4. Digital Twin Workflow

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=15,
    forecaster_type="linear",
    verbose=True,
)

twin = DigitalTwin(config)
twin.train(data, normalize=False)
forecast = twin.predict(data[..., -1], n_steps=5)

print(forecast.shape)
```

## Run Included Examples

Run commands from the repository root:

```bash
python examples/basic/01_tucker_decomposition.py
python examples/basic/02_sensor_placement.py
python examples/basic/03_field_reconstruction.py
python examples/basic/04_complete_pipeline.py
python examples/digital_twin/01_digital_twin_basic.py
```

Some examples require local datasets under `data/`. Those datasets are intentionally ignored by git.

## Run Tests

```bash
pytest
```

For faster checks during development:

```bash
pytest tests/unit -q
python -m compileall src tests examples scripts
```
