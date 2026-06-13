"""Reconstruction modules for full-field recovery from sensor measurements."""

from TBMD.config import CompressiveSensingConfig
from TBMD.config import (
    GeometryAwareReconstructionConfig as GeometryAwareCSConfig,  # Alias for backward compatibility
)

from .geometry_aware import GeometryAwareTensorCS
from .tensor_compressive_sensing import (
    TensorBasedCompressiveSensing,
    TensorCompressiveSensing,
    TensorCSReconstructor,
)

__all__ = [
    "TensorCompressiveSensing",
    "TensorBasedCompressiveSensing",
    "TensorCSReconstructor",
    "GeometryAwareTensorCS",
    "CompressiveSensingConfig",
    "GeometryAwareCSConfig",
]
