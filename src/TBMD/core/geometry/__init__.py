"""
Geometry-aware utilities for TBMD on unstructured meshes.

This package provides:
1. Cell adjacency graph construction for unstructured meshes
2. Laplacian matrix computation (graph Laplacian for spatial smoothness)
3. Geometric distance and gradient estimation
4. Utilities for geometry-aware sensor placement
"""

from .graph import MeshGraphBuilder
from .mesh import MeshGeometry, TorchMeshGeometry
from .metrics import GeometricWeightComputer, estimate_characteristic_length

__all__ = [
    "MeshGeometry",
    "TorchMeshGeometry",
    "MeshGraphBuilder",
    "GeometricWeightComputer",
    "estimate_characteristic_length",
]
