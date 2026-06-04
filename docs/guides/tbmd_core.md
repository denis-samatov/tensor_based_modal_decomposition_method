# TBMD Core Guide

## Purpose

Tensor-Based Modal Decomposition (TBMD) reduces high-dimensional spatiotemporal data into a compact modal representation. The repository focuses on tensor decomposition, sensor placement, and reconstruction workflows that can be combined with forecasting models.

Typical input data is a tensor such as:

```text
(x, y, time)
(x, y, z, time)
(active_cells, time)
```

## Core Workflow

1. Prepare tensor data.
2. Decompose the tensor with Tucker/HOSVD.
3. Build modal tensors and a modal basis.
4. Select sensor locations using tensor QR.
5. Reconstruct full fields from sparse measurements.
6. Optionally train a forecaster in the modal space.

## Tucker / HOSVD

The Tucker approximation represents a tensor as a core tensor multiplied by factor matrices:

```text
X ~= G x_1 U_1 x_2 U_2 ... x_n U_n
```

In this repository, `TuckerDecomposer` stores:

- `cores`: the reduced core tensor or tensors.
- `factors`: factor matrices for each tensor mode.
- `reconstructed_tensors`: reconstruction after `reconstruct()` is called.

Example:

```python
from TBMD.config import DecompositionConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposer

config = DecompositionConfig(ranks=[20, 20, 10])
decomposer = TuckerDecomposer(tensors=tensor_data, config=config)
decomposer.decompose()
decomposer.reconstruct()
```

## Modal Tensor Processing

Modal tensor processing converts decomposition outputs into a modal basis used by sensor placement and reconstruction code.

```python
from TBMD.config import ModalProcessorConfig, ProcessingStrategy
from TBMD.core.modal_processor.modes import BatchModalProcessor, ModalTensorStacker

modal_config = ModalProcessorConfig(
    processing_strategy=ProcessingStrategy.BATCH,
    return_numpy=False,
)

processor = BatchModalProcessor(modal_config)
stacker = ModalTensorStacker(modal_config)
modal_tensors = processor.process_multiple_subjects(decomposer.cores, decomposer.factors)
A_tensor = stacker.stack_modal_tensors(modal_tensors)
```

## Sensor Placement

`TensorTubeQRDecomposition` selects informative sensor locations from the modal basis.

```python
from TBMD.config import SensorPlacementConfig
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition

placer = TensorTubeQRDecomposition(
    tensor=A_tensor,
    config=SensorPlacementConfig(n_sensors=30),
)
P, Q, R = placer.factorize()
```

## Reconstruction

`TensorCompressiveSensing` reconstructs modal coefficients from sparse measurements and returns solver metrics.

```python
from TBMD.config import CompressiveSensingConfig
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

reconstructor = TensorCompressiveSensing(
    A=A_tensor,
    P=P,
    Y=measurements,
    core_cfg=CompressiveSensingConfig(max_iter=100),
)
x_hat, metrics = reconstructor.solve()
```

## Notes for Maintainers

- Keep decomposition, placement, and reconstruction contracts stable because examples and tests compose them directly.
- Treat benchmark or accuracy claims as experiment outputs that must be tied to reproducible scripts and datasets.
- Prefer adding small tests around shape contracts before changing public configuration defaults.
