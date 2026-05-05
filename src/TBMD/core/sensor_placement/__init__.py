"""
Sensor Placement Module

Модули для оптимального размещения сенсоров
"""



from .tensor_qr_factorization import TensorTubeQRDecomposition, TensorBasedTubeFiberPivotQRFactorization
from .geometry_aware import GeometryAwareTensorQR
from TBMD.config import (
    SensorPlacementConfig as TensorQRConfig, # Alias for backward compatibility
    GeometricSensorConfig as GeometricQRConfig # Alias for backward compatibility
)

__all__ = [
    'TensorTubeQRDecomposition',
    'TensorBasedTubeFiberPivotQRFactorization',
    'GeometryAwareTensorQR',
    'TensorQRConfig',
    'GeometricQRConfig'
]


