"""
Configuration objects for TBMD module.
"""

from .core import (
    BaseConfig,
    TBMDConfig,
    SensorPlacementConfig,
    FullPipelineConfig,
)
from .factory import (
    create_tbmd_config_from_dict,
    create_sensor_placement_config_from_dict,
    create_pipeline_config_from_dict,
    create_default_pipeline_config,
)

__all__ = [
    "BaseConfig",
    "TBMDConfig",
    "SensorPlacementConfig",
    "FullPipelineConfig",
    "create_tbmd_config_from_dict",
    "create_sensor_placement_config_from_dict",
    "create_pipeline_config_from_dict",
    "create_default_pipeline_config",
]

# Instantiate default configs for quick access
_tbmd_config = TBMDConfig()
_sensor_placement_config = SensorPlacementConfig()
_full_pipeline_config = FullPipelineConfig()
