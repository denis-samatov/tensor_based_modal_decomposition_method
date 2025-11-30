"""
Decomposition module - Tucker/HOSVD tensor decomposition
"""


from .hosvd import (
    TuckerDecomposer,
    TuckerDecomposerInterface,
    TuckerDecomposerCore,
    TensorProcessor,
    TensorValidator,
    TensorReconstructor,
    TensorVisualizer,
    ProcessingStrategy,
    CPUStrategy,
    GPUStrategy,
    DecomposerState,
    DecompositionResult as HOSVDDecompositionResult,
    ReconstructionResult,
    TensorDecompositionError,
    InvalidRankError,
    StateError,
    ValidationError,
    DecompositionConfig
)
from ..modal_processor.modes import BatchModalProcessor
from .geometry_aware import GeometryAwareTuckerDecomposer

__all__ = [
    # Base classes

    'DecompositionConfig',
    # Main interfaces
    'TuckerDecomposer',
    'TuckerDecomposerInterface',
    'BatchModalProcessor',
    'GeometryAwareTuckerDecomposer',
    # Core components
    'TuckerDecomposerCore',
    'TensorProcessor',
    'TensorValidator',
    'TensorReconstructor',
    'TensorVisualizer',
    # Strategies
    'ProcessingStrategy',
    'CPUStrategy',
    'GPUStrategy',
    # State & Results
    'DecomposerState',
    'HOSVDDecompositionResult',
    'ReconstructionResult',
    # Exceptions
    'TensorDecompositionError',
    'InvalidRankError',
    'StateError',
    'ValidationError',
]

