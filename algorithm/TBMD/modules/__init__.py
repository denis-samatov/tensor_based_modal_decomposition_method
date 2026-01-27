"""
TBMD Modules

Core tensor decomposition and sensor placement algorithms,
including standard and geometry-aware variants.

.. deprecated:: 2.0.0
   Use 'TBMD.core' and 'TBMD.config' instead.
"""
import warnings
warnings.warn(
    "The 'TBMD.modules' package is deprecated. Use 'TBMD.core' instead.",
    DeprecationWarning,
    stacklevel=2
)

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

from .GeometryAwareTensorCS import (
    GeometryAwareTensorCS,
    GeometryAwareCSConfig
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
    'GeometricQRConfig',
    # Geometry-aware CS
    'GeometryAwareTensorCS',
    'GeometryAwareCSConfig'
]

