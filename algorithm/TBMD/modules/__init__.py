"""
TBMD Modules

Core tensor decomposition and sensor placement algorithms,
including standard and geometry-aware variants.
"""

# Standard modules
from .TensorHOSVD import (
    TuckerDecomposer,
    TuckerDecomposerInterface,
    DecomposerState
)

from .TensorBasedTubeFiberPivotQRFactorization import (
    TensorTubeQRDecomposition,
    TensorQRConfig
)

from .TensorBasedCompressiveSensing import (
    TensorCompressiveSensing,
    CompressiveSensingConfig,
    CompressiveSensingMetrics
)

# Geometry-aware modules
from .GeometryAwareTensorHOSVD import (
    GeometryAwareTuckerDecomposer,
    GeometryAwareConfig
)

from .GeometryAwareTensorQR import (
    GeometryAwareTensorQR,
    GeometricQRConfig
)

__all__ = [
    # Standard HOSVD
    'TuckerDecomposer',
    'TuckerDecomposerInterface',
    'DecomposerState',
    # Standard QR
    'TensorTubeQRDecomposition',
    'TensorQRConfig',
    # Standard CS
    'TensorCompressiveSensing',
    'CompressiveSensingConfig',
    'CompressiveSensingMetrics',
    # Geometry-aware HOSVD
    'GeometryAwareTuckerDecomposer',
    'GeometryAwareConfig',
    # Geometry-aware QR
    'GeometryAwareTensorQR',
    'GeometricQRConfig'
]

