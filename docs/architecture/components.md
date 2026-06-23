# System Components

## Purpose
To detail the specific modules and classes that implement the core functionality of TBMD.

## Audience
Developers modifying or extending the core mathematical operations.

## Summary
The system comprises standard TBMD components (Tucker decomposition, ADMM reconstruction), geometry-aware extensions, and the Digital Twin orchestrator.

## Details

### Standard Core Components
1. **TuckerDecomposer**: Uses HOSVD to approximate tensors. Driven by `DecompositionConfig` which specifies target ranks. Stores `cores` and `factors`.
2. **Modal Processing**: `BatchModalProcessor` and `ModalTensorStacker` prepare the modal basis required by placement and reconstruction tools.
3. **TensorTubeQRDecomposition**: Implements tensor QR factorization to identify the most mathematically informative spatial indices for sensor placement.
4. **TensorCompressiveSensing**: Reconstructs modal coefficients from sparse spatial measurements using Alternating Direction Method of Multipliers (ADMM).

### Geometry-Aware Extensions
Geometry-aware components extend standard TBMD for irregular spatial connectivity, active-cell reservoir grids, or masked domains.
- **GeometryAwareTuckerDecomposer**: Applies graph Laplacian regularization during decomposition. Controlled by `alpha` (regularization strength) and `spatial_modes`.
- **GeometryAwareTensorCS**: Reconstruction with graph-based penalties.
- **MeshGraphBuilder & MeshGeometry**: Utilities for converting grid shapes or explicit adjacency matrices into graph Laplacian matrices for the decomposer.

### Digital Twin Orchestrator
The `DigitalTwin` class acts as a facade over the core components.
It orchestrates the flow from `train()` (decomposition + forecaster fitting) to `predict()` and `update_from_sensors()`. Supported forecasters include linear, mlp, lstm, and persistence.

## Examples
**Instantiating a Geometry-Aware Decomposer:**
```python
from TBMD.core.decomposition.geometry_aware import GeometryAwareConfig, GeometryAwareTuckerDecomposer
from TBMD.core.geometry import MeshGraphBuilder

builder = MeshGraphBuilder(connectivity_type="grid")
mesh = builder.build_from_shape(spatial_shape=(100, 100))

geo_config = GeometryAwareConfig(alpha=0.1, spatial_modes=[0], laplacian_type="normalized")

decomposer = GeometryAwareTuckerDecomposer(
    tensor=data_tensor,
    mesh=mesh,
    geo_config=geo_config,
    ranks=[20, 10],
)
```

## Validation
To verify component integrity individually:
```bash
pytest tests/unit/ -q
```
Expected result: Unit tests testing shape contracts, exceptions, and isolated logic for each component pass successfully.

## Related docs
- [Data Flow](data-flow.md)
