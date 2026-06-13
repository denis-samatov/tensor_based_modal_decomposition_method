"""TBMD core algorithms with a modular v2.0 structure."""

# Imports from submodules
from .decomposition import BatchModalProcessor, GeometryAwareTuckerDecomposer, TuckerDecomposer
from .reconstruction import GeometryAwareTensorCS, TensorCompressiveSensing
from .sensor_placement import GeometryAwareTensorQR, TensorTubeQRDecomposition

__all__ = [
    # Decomposition
    "TuckerDecomposer",
    "GeometryAwareTuckerDecomposer",
    "BatchModalProcessor",
    # Sensor Placement
    "TensorTubeQRDecomposition",
    "GeometryAwareTensorQR",
    # Reconstruction
    "TensorCompressiveSensing",
    "GeometryAwareTensorCS",
]

__version__ = "2.0.0"
