"""
TBMD Core Package

Ядро алгоритмов TBMD с модульной структурой v2.0
"""

# Импорты из подмодулей
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

from .digital_twin import (
    DigitalTwin,
    DigitalTwinState,
    # DigitalTwinTBMD  # Alias для обратной совместимости
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
    
    # Digital Twin
    'DigitalTwin',
    'DigitalTwinState',
    # 'DigitalTwinTBMD',
]

__version__ = '2.0.0'
