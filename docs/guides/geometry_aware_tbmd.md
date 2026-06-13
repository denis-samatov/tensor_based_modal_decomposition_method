# Geometry-Aware TBMD

## Overview

Geometry-aware TBMD extends the standard tensor workflow with graph or mesh information. It is intended for data where Euclidean tensor-grid assumptions are too restrictive, such as active-cell reservoir grids, masked domains, or irregular spatial connectivity.

The implementation provides geometry-aware decomposition, reconstruction, and sensor placement utilities. These components are experimental and should be validated on the target dataset before being used for conclusions.

## Motivation

Standard TBMD treats tensor axes as regular dimensions. Real spatial domains may include:

- inactive cells;
- non-rectangular boundaries;
- irregular connectivity;
- anisotropic or graph-defined neighborhoods.

Geometry-aware variants encode spatial relationships through graph and mesh structures, including graph Laplacian regularization.

## Main Components

- `GeometryAwareTuckerDecomposer`: decomposition with geometry-aware regularization.
- `GeometryAwareTensorCS`: reconstruction with geometry-aware penalties.
- `GeometryAwareTensorQR`: sensor placement with geometry-aware scoring.
- `MeshGeometry` and `MeshGraphBuilder`: helpers for constructing spatial graph representations.

## Basic Example

```python
from TBMD.core.decomposition.geometry_aware import (
    GeometryAwareConfig,
    GeometryAwareTuckerDecomposer,
)
from TBMD.core.geometry import MeshGraphBuilder

builder = MeshGraphBuilder(connectivity_type="grid")
mesh = builder.build_from_shape(spatial_shape=(100, 100))

geo_config = GeometryAwareConfig(
    alpha=0.1,
    spatial_modes=[0],
    laplacian_type="normalized",
)

decomposer = GeometryAwareTuckerDecomposer(
    tensor=data_tensor,
    mesh=mesh,
    geo_config=geo_config,
    ranks=[20, 10],
)
decomposer.decompose()
reconstructed = decomposer.reconstruct()
```

## Configuration Notes

- `alpha` controls the strength of Laplacian regularization.
- `spatial_modes` selects which tensor modes receive geometry-aware regularization.
- `laplacian_type` selects the Laplacian normalization strategy supported by the implementation.
- Graph construction parameters should be chosen to match the physical interpretation of adjacency in the dataset.

## Working With Masks

For masked domains, verify the current graph builder API before assuming direct mask support. If a direct mask API is not available, construct the adjacency matrix explicitly and pass it through the mesh representation.

```python
from TBMD.core.geometry import MeshGeometry

mesh = MeshGeometry(adjacency_matrix=adjacency_matrix)
```

## Validation

Before using geometry-aware outputs in a report:

1. Compare reconstruction error against standard TBMD on the same split.
2. Inspect selected sensor locations for domain validity.
3. Check whether graph construction matches the intended physical neighborhood.
4. Record all graph and regularization parameters with the experiment output.

## Examples

Runnable examples are under `examples/geometry_aware/`.
