"""
Configuration objects for TBMD module.
"""

from .base import BaseConfig
from .decomposition import DecompositionConfig, GeometryAwareDecompositionConfig
from .sensor_placement import SensorPlacementConfig, GeometricSensorConfig
from .reconstruction import (
    ReconstructionConfig, 
    CompressiveSensingConfig, 
    ExtensionCompressiveSensingConfig, 
    GeometryAwareReconstructionConfig
)
from .digital_twin import DigitalTwinConfig
from .experiments import ExperimentConfig
from .modal_processor import ModalProcessorConfig

__all__ = [
    "BaseConfig",
    "DecompositionConfig",
    "GeometryAwareDecompositionConfig",
    "SensorPlacementConfig",
    "GeometricSensorConfig",
    "ReconstructionConfig",
    "CompressiveSensingConfig",
    "ExtensionCompressiveSensingConfig",
    "GeometryAwareReconstructionConfig",
    "DigitalTwinConfig",
    "ExperimentConfig",
    "ModalProcessorConfig"
]

# Instantiate default configs for quick access
_decomposition_config = DecompositionConfig()
_sensor_placement_config = SensorPlacementConfig()
_reconstruction_config = ReconstructionConfig()
_digital_twin_config = DigitalTwinConfig()
_experiment_config = ExperimentConfig()
