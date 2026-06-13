"""
Decomposition module - Tucker/HOSVD tensor decomposition
"""

from ..modal_processor.modes import BatchModalProcessor
from .geometry_aware import GeometryAwareConfig as GeometryAwareDecompositionConfig
from .geometry_aware import GeometryAwareTuckerDecomposer
from .hosvd import (
    CPUStrategy,
    DecomposerState,
    DecompositionConfig,
    GPUStrategy,
    InvalidRankError,
    ProcessingStrategy,
    ReconstructionResult,
    StateError,
    TensorDecompositionError,
    TensorProcessor,
    TensorReconstructor,
    TensorValidator,
    TensorVisualizer,
    TuckerDecomposer,
    TuckerDecomposerCore,
    TuckerDecomposerInterface,
    ValidationError,
)
from .hosvd import DecompositionResult as HOSVDDecompositionResult

__all__ = [
    # Base classes
    "DecompositionConfig",
    # Main interfaces
    "TuckerDecomposer",
    "TuckerDecomposerInterface",
    "BatchModalProcessor",
    "GeometryAwareTuckerDecomposer",
    "GeometryAwareDecompositionConfig",
    # Core components
    "TuckerDecomposerCore",
    "TensorProcessor",
    "TensorValidator",
    "TensorReconstructor",
    "TensorVisualizer",
    # Strategies
    "ProcessingStrategy",
    "CPUStrategy",
    "GPUStrategy",
    # State & Results
    "DecomposerState",
    "HOSVDDecompositionResult",
    "ReconstructionResult",
    # Exceptions
    "TensorDecompositionError",
    "InvalidRankError",
    "StateError",
    "ValidationError",
]
