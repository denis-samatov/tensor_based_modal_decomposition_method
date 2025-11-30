"""
Sensor Placement Module

Модули для оптимального размещения сенсоров
"""


from .tensor_qr_factorization import TensorTubeQRDecomposition, TensorBasedTubeFiberPivotQRFactorization
from .geometry_aware import GeometryAwareTensorQR

__all__ = [

    'TensorTubeQRDecomposition',
    'TensorBasedTubeFiberPivotQRFactorization',
    'GeometryAwareTensorQR'
]


