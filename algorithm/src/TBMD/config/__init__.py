"""
TBMD Configuration Module

Модульные конфигурации для всех компонентов TBMD v2.0
"""

from .base import BaseConfig
from .decomposition import DecompositionConfig, GeometryAwareDecompositionConfig
from .modal_processor import ModalProcessorConfig, ProcessingStrategy
from .sensor_placement import SensorPlacementConfig, GeometricSensorConfig
from .reconstruction import (
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig,
    ReconstructionConfig,
    GeometryAwareReconstructionConfig,
    TensorCSConfig,
    CSConfig
)
from .digital_twin import DigitalTwinConfig
from .forecaster import (
    ForecasterConfig,
    LinearForecasterConfig,
    MLPForecasterConfig,
    LSTMForecasterConfig,
    create_forecaster_config_from_dict
)

__all__ = [
    'BaseConfig',
    'DecompositionConfig',
    'GeometryAwareDecompositionConfig',
    'ModalProcessorConfig',
    'ProcessingStrategy',
    'SensorPlacementConfig',
    'GeometricSensorConfig',
    # New CS configs (primary)
    'CompressiveSensingConfig',
    'ExtensionCompressiveSensingConfig',
    'TensorCSConfig',
    'CSConfig',
    # Legacy configs
    'ReconstructionConfig',
    'GeometryAwareReconstructionConfig',
    'DigitalTwinConfig',
    'ForecasterConfig',
    'LinearForecasterConfig',
    'MLPForecasterConfig',
    'LSTMForecasterConfig',
    'create_forecaster_config_from_dict'
]


##########################################################################
# Deprecated Constants - Для обратной совместимости со старым API
##########################################################################

# Создать дефолтные экземпляры для константот
_base_config = BaseConfig()
_decomposition_config = DecompositionConfig()
_sensor_config = SensorPlacementConfig()
_reconstruction_config = ReconstructionConfig()
_digital_twin_config = DigitalTwinConfig()

# Базовые параметры
SEED = _base_config.seed
SET_BACKEND = _base_config.backend
DTYPE = _base_config.dtype

# Sensor configuration
NUMBER_SENSORS = _sensor_config.n_sensors

# Reconstruction parameters
MAX_ITERATIONS = _reconstruction_config.max_iterations
CONVERGENCE_EPS = _reconstruction_config.convergence_eps
DAMPING_FACTOR = _reconstruction_config.damping_factor
INITIAL_STEP_SIZE = _reconstruction_config.initial_step_size
MAX_STEP_SIZE = _reconstruction_config.max_step_size

# Добавить в экспорт
__all__.extend([
    'SEED',
    'SET_BACKEND',
    'DTYPE',
    'NUMBER_SENSORS',
    'MAX_ITERATIONS',
    'CONVERGENCE_EPS',
    'DAMPING_FACTOR',
    'INITIAL_STEP_SIZE',
    'MAX_STEP_SIZE'
])
