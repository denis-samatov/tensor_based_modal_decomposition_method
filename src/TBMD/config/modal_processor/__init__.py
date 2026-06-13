"""Configuration for modal tensor processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class ProcessingStrategy(Enum):
    """Strategy for processing modal tensors."""

    SEQUENTIAL = "sequential"
    BATCH = "batch"
    MEMORY_EFFICIENT = "memory_efficient"


@dataclass
class ModalProcessorConfig:
    """Configuration for modal tensor processing."""

    device: str = "cpu"
    return_numpy: bool = True
    processing_strategy: ProcessingStrategy = ProcessingStrategy.BATCH
    batch_size: Optional[int] = None
    memory_limit_gb: float = 4.0
    enable_progress_logging: bool = True
    validation_enabled: bool = True
    numerical_precision: torch.dtype = torch.float32

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
