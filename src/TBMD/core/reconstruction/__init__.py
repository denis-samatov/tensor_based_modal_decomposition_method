"""
Reconstruction Module

Модули для реконструкции полных полей по измерениям сенсоров
"""



from .tensor_compressive_sensing import (
    TensorCompressiveSensing, 
    TensorBasedCompressiveSensing,
    TensorCSReconstructor
)
from .geometry_aware import GeometryAwareTensorCS
from TBMD.config import (
    CompressiveSensingConfig,
    GeometryAwareReconstructionConfig as GeometryAwareCSConfig # Alias for backward compatibility
)

__all__ = [
    'TensorCompressiveSensing',
    'TensorBasedCompressiveSensing',
    'TensorCSReconstructor',
    'GeometryAwareTensorCS',
    'CompressiveSensingConfig',
    'GeometryAwareCSConfig'
]


