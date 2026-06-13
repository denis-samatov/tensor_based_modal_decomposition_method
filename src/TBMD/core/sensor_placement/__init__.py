"""Sensor placement modules."""

from TBMD.config import (
    GeometricSensorConfig as GeometricQRConfig,  # Alias for backward compatibility
)
from TBMD.config import SensorPlacementConfig as TensorQRConfig  # Alias for backward compatibility

from .geometry_aware import GeometryAwareTensorQR
from .tensor_qr_factorization import (
    TensorBasedTubeFiberPivotQRFactorization,
    TensorTubeQRDecomposition,
)

__all__ = [
    "TensorTubeQRDecomposition",
    "TensorBasedTubeFiberPivotQRFactorization",
    "GeometryAwareTensorQR",
    "TensorQRConfig",
    "GeometricQRConfig",
]
