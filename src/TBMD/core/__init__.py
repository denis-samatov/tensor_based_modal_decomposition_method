"""TBMD core algorithms with a modular v2.0 structure."""

# Imports from submodules
from .decomposition import (
    TuckerDecomposer,
    GeometryAwareTuckerDecomposer,
    BatchModalProcessor
)

from .sensor_placement import (
    TensorTubeQRDecomposition,
    GeometryAwareTensorQR
)

from .reconstruction import (
    TensorCompressiveSensing,
    GeometryAwareTensorCS
)



__all__ = [
    # Decomposition
    'TuckerDecomposer',
    'GeometryAwareTuckerDecomposer',
    'BatchModalProcessor',
    
    # Sensor Placement
    'TensorTubeQRDecomposition',
    'GeometryAwareTensorQR',
    
    # Reconstruction
    'TensorCompressiveSensing',
    'GeometryAwareTensorCS',
    

]

__version__ = '2.0.0'
