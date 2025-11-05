"""
TBMD Utilities Module

Includes standard utilities and geometry-aware extensions.
"""

from .utils import (
    to_torch_tensor,
    get_torch_device,
    extract_step_number,
    auto_select_mode,
    reconstruct_tensor,
    build_Y_matrices,
    build_wells_matrix
)

from .geometry import (
    MeshGeometry,
    TorchMeshGeometry,
    MeshGraphBuilder,
    GeometricWeightComputer,
    estimate_characteristic_length
)

__all__ = [
    # Standard utils
    'to_torch_tensor',
    'get_torch_device',
    'extract_step_number',
    'auto_select_mode',
    'reconstruct_tensor',
    'build_Y_matrices',
    'build_wells_matrix',
    # Geometry utilities
    'MeshGeometry',
    'TorchMeshGeometry',
    'MeshGraphBuilder',
    'GeometricWeightComputer',
    'estimate_characteristic_length'
]

