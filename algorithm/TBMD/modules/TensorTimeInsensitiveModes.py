"""
Time-Insensitive Modal Tensor Processing Module

This module implements the computation of time-insensitive modes according to the formula:
M_{:,n} = A × G_{:,n} × B  (Equation 12)

Where:
- M_{:,n} is the n-th time-insensitive mode
- A, B are spatial factor matrices
- G_{:,n} is the n-th slice of the core tensor along the time dimension

The previous implementation incorrectly applied all spatial factors at once.
This implementation correctly computes each mode according to the mathematical formula.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import tensorly as tl
import torch
from torch import nn

from ..utils.utils import get_torch_device, to_torch_tensor

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
TensorLike = Union[np.ndarray, torch.Tensor]
SubjectDict = Dict[str, TensorLike]
FactorList = List[TensorLike]
SubjectFactorsDict = Dict[str, FactorList]


class ProcessingError(Exception):
    """Base exception for modal tensor processing errors."""
    pass


class ValidationError(ProcessingError):
    """Raised when input validation fails."""
    pass


class ComputationError(ProcessingError):
    """Raised when computation fails."""
    pass


class DimensionMismatchError(ValidationError):
    """Raised when tensor dimensions don't match expected values."""
    pass


class ProcessingStrategy(Enum):
    """A strategy for processing modal tensors."""
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    MEMORY_EFFICIENT = "memory_efficient"


@dataclass
class ModalProcessorConfig:
    """A configuration class for modal tensor processing.

    Attributes:
        device (str): The device to use for processing. Defaults to 'cpu'.
        return_numpy (bool): If `True`, returns the result as a NumPy array.
            Defaults to `True`.
        processing_strategy (ProcessingStrategy): The processing strategy to
            use. Defaults to `ProcessingStrategy.BATCH`.
        batch_size (Optional[int]): The batch size for batch processing.
        memory_limit_gb (float): The memory limit in gigabytes for memory-
            efficient processing. Defaults to 4.0.
        enable_progress_logging (bool): If `True`, enables progress logging.
            Defaults to `True`.
        validation_enabled (bool): If `True`, enables input validation.
            Defaults to `True`.
        numerical_precision (torch.dtype): The numerical precision for
            processing. Defaults to `torch.float32`.
    """
    device: str = 'cpu'
    return_numpy: bool = True
    processing_strategy: ProcessingStrategy = ProcessingStrategy.BATCH
    batch_size: Optional[int] = None
    memory_limit_gb: float = 4.0
    enable_progress_logging: bool = True
    validation_enabled: bool = True
    numerical_precision: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")


class TensorValidator:
    """Validates tensor inputs for modal processing."""
    
    @staticmethod
    def validate_core_tensor(core: TensorLike) -> None:
        """Validate the properties of the core tensor.

        Parameters
        ----------
        core : TensorLike
            The core tensor to validate.
        """
        if core is None:
            raise ValidationError("Core tensor cannot be None")
        
        if isinstance(core, torch.Tensor):
            core_array = core.detach().cpu().numpy()
        else:
            core_array = core
        
        if core_array.ndim < 3:
            raise DimensionMismatchError(
                f"Core tensor must have at least 3 dimensions, got {core_array.ndim}"
            )
        
        if np.any(np.array(core_array.shape) <= 0):
            raise DimensionMismatchError(
                f"All core tensor dimensions must be positive, got shape {core_array.shape}"
            )
    
    @staticmethod
    def validate_factors(factors: FactorList, core_shape: Tuple[int, ...]) -> None:
        """Validate the compatibility of factor matrices with the core tensor.

        Parameters
        ----------
        factors : FactorList
            The list of factor matrices.
        core_shape : Tuple[int, ...]
            The shape of the core tensor.
        """
        if not factors:
            raise ValidationError("Factors list cannot be empty")
        
        if len(factors) != len(core_shape):
            raise DimensionMismatchError(
                f"Number of factors ({len(factors)}) must match core tensor dimensions ({len(core_shape)})"
            )
        
        for i, factor in enumerate(factors):
            if factor is None:
                raise ValidationError(f"Factor {i} cannot be None")
            
            if isinstance(factor, torch.Tensor):
                factor_array = factor.detach().cpu().numpy()
            else:
                factor_array = factor
            
            if factor_array.ndim != 2:
                raise DimensionMismatchError(
                    f"Factor {i} must be 2D matrix, got {factor_array.ndim}D"
                )
            
            if factor_array.shape[1] != core_shape[i]:
                raise DimensionMismatchError(
                    f"Factor {i} shape {factor_array.shape} incompatible with core dimension {core_shape[i]}"
                )
    
    @staticmethod
    def validate_subject_data(cores: Union[SubjectDict, TensorLike], 
                            factors: Union[SubjectFactorsDict, FactorList]) -> None:
        """Validate the consistency of subject data.

        Parameters
        ----------
        cores : Union[SubjectDict, TensorLike]
            The core tensors for the subjects.
        factors : Union[SubjectFactorsDict, FactorList]
            The factor matrices for the subjects.
        """
        if isinstance(cores, dict) != isinstance(factors, dict):
            raise ValidationError(
                "cores and factors must both be dictionaries or both be single-subject format"
            )
        
        if isinstance(cores, dict) and isinstance(factors, dict):
            if set(cores.keys()) != set(factors.keys()):
                raise ValidationError(
                    f"Subject keys mismatch: cores={set(cores.keys())}, factors={set(factors.keys())}"
                )


class TimeInsensitiveModeComputer:
    """Computes time-insensitive modes.

    This class implements the computation of time-insensitive modes according
    to Equation (12) from the TBMD paper: `M_{:,n} = A × G_{:,n} × B`.

    Args:
        config (ModalProcessorConfig): The configuration for the modal
            processor.
    """
    
    def __init__(self, config: ModalProcessorConfig):
        self.config = config
        self.device = get_torch_device(config.device)
        
    def compute_single_mode(self, 
                          spatial_factors: List[torch.Tensor], 
                          core_slice: torch.Tensor) -> torch.Tensor:
        """Computes a single time-insensitive mode.

        This method implements Equation (12) from the TBMD paper, which for a
        3D case is `M_{:,n} = A × G_{:,n} × B^T`. For higher dimensions, a
        generalized tensor contraction is used.

        Args:
            spatial_factors (List[torch.Tensor]): A list of spatial factor
                matrices (e.g., [A, B, ...]), excluding the temporal factor.
            core_slice (torch.Tensor): The n-th slice of the core tensor,
                `G_{:,n}`.

        Returns:
            torch.Tensor: The computed mode, `M_{:,n}`.
        """
        try:
            if len(spatial_factors) == 2:  # 3D tensor case
                A, B = spatial_factors
                # M_{:,n} = A × G_{:,n} × B^T
                # Using einsum for clarity: 'ij,jk,lk->il'
                mode = torch.einsum('ij,jk,lk->il', A, core_slice, B)
            else:
                # General case: apply factors sequentially using tensorly's mode_dot
                try:
                    # Use tensorly's mode_dot for proper n-mode operations
                    from tensorly.tenalg import mode_dot
                    
                    mode = core_slice
                    
                    # Apply each spatial factor using tensorly's mode_dot
                    for i, factor in enumerate(spatial_factors):
                        # tensorly's mode_dot expects (tensor, matrix, mode)
                        # where mode is the dimension to contract along
                        mode = mode_dot(mode, factor, mode=i)
                        
                except Exception as e:
                    # Fallback to simplified manual implementation
                    mode = core_slice
                    
                    # Apply each spatial factor to the corresponding mode
                    for i, factor in enumerate(spatial_factors):
                        # For the first factor, contract with the first dimension
                        if i == 0:
                            # Factor shape: (output_size, input_size)
                            # Mode shape: (input_size, ...)
                            mode = torch.tensordot(factor, mode, dims=([1], [0]))
                        else:
                            # For subsequent factors, the dimension index doesn't change
                            # because the previous contraction replaced one dimension with another
                            mode = torch.tensordot(factor, mode, dims=([1], [i]))
            
            return mode.to(dtype=self.config.numerical_precision)
            
        except Exception as e:
            # Add debug information for better error diagnosis
            factor_shapes = [f.shape for f in spatial_factors] if spatial_factors else []
            raise ComputationError(
                f"Failed to compute mode: {e}. "
                f"Core slice shape: {core_slice.shape}, "
                f"Factor shapes: {factor_shapes}"
            ) from e
    
    def compute_all_modes(self, 
                         core: torch.Tensor, 
                         factors: List[torch.Tensor]) -> torch.Tensor:
        """Computes all time-insensitive modes for a single subject.

        Args:
            core (torch.Tensor): The core tensor from the Tucker decomposition.
            factors (List[torch.Tensor]): The factor matrices from the Tucker
                decomposition.

        Returns:
            torch.Tensor: The modal tensor, with shape `[...spatial_dims...,
            n_modes]`.
        """
        # Spatial factors (all except the last temporal factor)
        spatial_factors = factors[:-1]
        time_dim = core.shape[-1]
        
        if self.config.processing_strategy == ProcessingStrategy.BATCH:
            return self._compute_modes_batch(core, spatial_factors, time_dim)
        elif self.config.processing_strategy == ProcessingStrategy.MEMORY_EFFICIENT:
            return self._compute_modes_memory_efficient(core, spatial_factors, time_dim)
        else:  # SEQUENTIAL
            return self._compute_modes_sequential(core, spatial_factors, time_dim)
    
    def _compute_modes_batch(self, 
                           core: torch.Tensor, 
                           spatial_factors: List[torch.Tensor], 
                           time_dim: int) -> torch.Tensor:
        """Perform a batch computation of all modes.

        This is the fastest method, but uses more memory.

        Parameters
        ----------
        core : torch.Tensor
            The core tensor.
        spatial_factors : List[torch.Tensor]
            A list of spatial factor matrices.
        time_dim : int
            The time dimension.

        Returns
        -------
        torch.Tensor
            The computed modes.
        """
        modes = []
        
        # Process in batches if batch_size is specified
        batch_size = self.config.batch_size or time_dim
        
        for start_idx in range(0, time_dim, batch_size):
            end_idx = min(start_idx + batch_size, time_dim)
            batch_modes = []
            
            for n in range(start_idx, end_idx):
                core_slice = core[..., n]
                mode = self.compute_single_mode(spatial_factors, core_slice)
                batch_modes.append(mode)
            
            if batch_modes:
                modes.extend(batch_modes)
                
                if self.config.enable_progress_logging:
                    logger.debug(f"Processed modes {start_idx}-{end_idx-1}/{time_dim}")
        
        return torch.stack(modes, dim=-1)
    
    def _compute_modes_sequential(self, 
                                core: torch.Tensor, 
                                spatial_factors: List[torch.Tensor], 
                                time_dim: int) -> torch.Tensor:
        """Perform a sequential computation.

        This is slower, but uses less memory.

        Parameters
        ----------
        core : torch.Tensor
            The core tensor.
        spatial_factors : List[torch.Tensor]
            A list of spatial factor matrices.
        time_dim : int
            The time dimension.

        Returns
        -------
        torch.Tensor
            The computed modes.
        """
        modes = []
        
        for n in range(time_dim):
            core_slice = core[..., n]
            mode = self.compute_single_mode(spatial_factors, core_slice)
            modes.append(mode)
            
            if self.config.enable_progress_logging and (n + 1) % 10 == 0:
                logger.debug(f"Processed mode {n + 1}/{time_dim}")
        
        return torch.stack(modes, dim=-1)
    
    def _compute_modes_memory_efficient(self, 
                                      core: torch.Tensor, 
                                      spatial_factors: List[torch.Tensor], 
                                      time_dim: int) -> torch.Tensor:
        """Perform a memory-efficient computation with gradient checkpointing.

        Parameters
        ----------
        core : torch.Tensor
            The core tensor.
        spatial_factors : List[torch.Tensor]
            A list of spatial factor matrices.
        time_dim : int
            The time dimension.

        Returns
        -------
        torch.Tensor
            The computed modes.
        """
        def compute_batch_fn(start_idx: int, end_idx: int) -> List[torch.Tensor]:
            batch_modes = []
            for n in range(start_idx, end_idx):
                core_slice = core[..., n]
                mode = self.compute_single_mode(spatial_factors, core_slice)
                batch_modes.append(mode)
            return batch_modes
        
        # Estimate memory usage and adjust batch size
        estimated_batch_size = self._estimate_optimal_batch_size(core, spatial_factors)
        
        modes = []
        for start_idx in range(0, time_dim, estimated_batch_size):
            end_idx = min(start_idx + estimated_batch_size, time_dim)
            
            # Use gradient checkpointing for memory efficiency
            if torch.is_grad_enabled():
                batch_modes = torch.utils.checkpoint.checkpoint(
                    compute_batch_fn, start_idx, end_idx, use_reentrant=False
                )
            else:
                batch_modes = compute_batch_fn(start_idx, end_idx)
            
            modes.extend(batch_modes)
        
        return torch.stack(modes, dim=-1)
    
    def _estimate_optimal_batch_size(self, 
                                   core: torch.Tensor, 
                                   spatial_factors: List[torch.Tensor]) -> int:
        """Estimate the optimal batch size based on memory constraints.

        Parameters
        ----------
        core : torch.Tensor
            The core tensor.
        spatial_factors : List[torch.Tensor]
            A list of spatial factor matrices.

        Returns
        -------
        int
            The estimated optimal batch size.
        """
        # Rough estimation based on tensor sizes
        core_memory = core.numel() * core.element_size()
        factors_memory = sum(f.numel() * f.element_size() for f in spatial_factors)
        
        # Estimate memory per mode computation
        mode_shape = spatial_factors[0].shape[0]  # Assuming square output
        if len(spatial_factors) > 1:
            mode_shape *= spatial_factors[1].shape[0]
        
        memory_per_mode = mode_shape * 4  # 4 bytes for float32
        
        # Target memory usage
        target_memory = self.config.memory_limit_gb * 1024**3
        available_memory = target_memory - core_memory - factors_memory
        
        estimated_batch_size = max(1, int(available_memory // memory_per_mode))
        return min(estimated_batch_size, core.shape[-1])


class ModalTensorProcessor:
    """The main processor for computing time-insensitive modal tensors.

    Args:
        config (Optional[ModalProcessorConfig]): A configuration for the
            processor. If `None`, default settings are used.
    """
    
    def __init__(self, config: Optional[ModalProcessorConfig] = None):
        self.config = config or ModalProcessorConfig()
        self.validator = TensorValidator()
        self.computer = TimeInsensitiveModeComputer(self.config)
        self.device = get_torch_device(self.config.device)
        
        logger.info(f"Initialized ModalTensorProcessor with device: {self.device}")
    
    def process_single_subject(self, 
                             core: TensorLike, 
                             factors: FactorList) -> Union[np.ndarray, torch.Tensor]:
        """Processes a single subject to compute a time-insensitive modal tensor.

        Args:
            core (TensorLike): The core tensor from the Tucker decomposition.
            factors (FactorList): The factor matrices from the Tucker
                decomposition.

        Returns:
            Union[np.ndarray, torch.Tensor]: The computed modal tensor.
        """
        if self.config.validation_enabled:
            self.validator.validate_core_tensor(core)
            
        # Convert to tensors on target device
        core_tensor = to_torch_tensor(core, self.device)
        factor_tensors = [to_torch_tensor(f, self.device) for f in factors]
        
        if self.config.validation_enabled:
            self.validator.validate_factors(factor_tensors, core_tensor.shape)
        
        # Compute modal tensor
        modal_tensor = self.computer.compute_all_modes(core_tensor, factor_tensors)
        
        if self.config.enable_progress_logging:
            logger.info(f"Computed modal tensor with shape: {modal_tensor.shape}")
        
        # Return in requested format
        if self.config.return_numpy:
            return tl.to_numpy(modal_tensor)
        else:
            return modal_tensor


class BatchModalProcessor:
    """Processes multiple subjects efficiently.

    Args:
        config (Optional[ModalProcessorConfig]): A configuration for the
            processor. If `None`, default settings are used.
    """
    
    def __init__(self, config: Optional[ModalProcessorConfig] = None):
        self.config = config or ModalProcessorConfig()
        self.processor = ModalTensorProcessor(self.config)
    
    def process_multiple_subjects(self, 
                                cores: Union[SubjectDict, TensorLike], 
                                factors: Union[SubjectFactorsDict, FactorList]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Processes multiple subjects to compute their modal tensors.

        Args:
            cores (Union[SubjectDict, TensorLike]): The core tensors for the
                subjects.
            factors (Union[SubjectFactorsDict, FactorList]): The factor matrices
                for the subjects.

        Returns:
            Dict[str, Union[np.ndarray, torch.Tensor]]: A dictionary of the
            computed modal tensors.
        """
        if self.config.validation_enabled:
            self.processor.validator.validate_subject_data(cores, factors)
        
        results = {}
        
        # Handle single subject case
        if not isinstance(cores, dict):
            modal_tensor = self.processor.process_single_subject(cores, factors)
            results["single_subject"] = modal_tensor
            return results
        
        # Handle multiple subjects
        total_subjects = len(cores)
        for i, subject in enumerate(cores.keys()):
            try:
                modal_tensor = self.processor.process_single_subject(
                    cores[subject], factors[subject]
                )
                results[subject] = modal_tensor
                
                if self.config.enable_progress_logging:
                    logger.info(f"Processed subject '{subject}' ({i+1}/{total_subjects})")
                    
            except Exception as e:
                logger.error(f"Failed to process subject '{subject}': {e}")
                raise ComputationError(f"Subject '{subject}' processing failed") from e
        
        return results


class ModalTensorStacker:
    """Stacks modal tensors from multiple subjects.

    Args:
        config (Optional[ModalProcessorConfig]): A configuration for the
            stacker. If `None`, default settings are used.
    """
    
    def __init__(self, config: Optional[ModalProcessorConfig] = None):
        self.config = config or ModalProcessorConfig()
        self.device = get_torch_device(self.config.device)
    
    def stack_modal_tensors(self, 
                          modal_tensors: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
        """Stacks modal tensors from multiple subjects along the time dimension.

        Args:
            modal_tensors (Dict[str, Union[np.ndarray, torch.Tensor]]): A
                dictionary of modal tensors from multiple subjects.

        Returns:
            Union[np.ndarray, torch.Tensor]: A stacked tensor containing all
            time slices.
        """
        if not modal_tensors:
            raise ValidationError("modal_tensors is empty")
        
        # Convert all tensors to the target device
        tensors_on_device = []
        total_time_slices = 0
        
        for subject, tensor in modal_tensors.items():
            tensor_device = to_torch_tensor(tensor, self.device)
            tensors_on_device.append(tensor_device)
            total_time_slices += tensor_device.shape[-1]
            
            if self.config.enable_progress_logging:
                logger.debug(f"Subject '{subject}': {tensor_device.shape[-1]} time slices")
        
        # Pre-allocate output tensor for efficiency
        first_tensor = tensors_on_device[0]
        spatial_shape = first_tensor.shape[:-1]
        output_shape = spatial_shape + (total_time_slices,)
        
        stacked_tensor = torch.zeros(
            output_shape, 
            dtype=self.config.numerical_precision, 
            device=self.device
        )
        
        # Efficiently fill the stacked tensor
        current_idx = 0
        for tensor in tensors_on_device:
            time_slices = tensor.shape[-1]
            stacked_tensor[..., current_idx:current_idx + time_slices] = tensor
            current_idx += time_slices
        
        if self.config.enable_progress_logging:
            logger.info(f"Stacked tensor shape: {stacked_tensor.shape}")
        
        # Return in requested format
        if self.config.return_numpy:
            return tl.to_numpy(stacked_tensor)
        else:
            return stacked_tensor


# Convenience functions for backward compatibility
def compute_modal_tensor(core: np.ndarray,
                        factors: List[np.ndarray],
                        device: str = 'cpu',
                        return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """Computes the modal tensor for a single subject.

    .. deprecated:: 0.1.0
       Use :py:meth:`ModalTensorProcessor.process_single_subject` instead.

    Args:
        core (np.ndarray): The core tensor from the Tucker decomposition.
        factors (List[np.ndarray]): The factor matrices from the Tucker
            decomposition.
        device (str, optional): The device to use for processing. Defaults to
            'cpu'.
        return_numpy (bool, optional): If `True`, returns the result as a NumPy
            array. Defaults to `True`.

    Returns:
        Union[np.ndarray, torch.Tensor]: The computed modal tensor.
    """
    warnings.warn(
        "compute_modal_tensor is deprecated. Use ModalTensorProcessor.process_single_subject() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    config = ModalProcessorConfig(device=device, return_numpy=return_numpy)
    processor = ModalTensorProcessor(config)
    return processor.process_single_subject(core, factors)


def process_all_subjects(cores: Union[Dict[str, Union[np.ndarray, torch.Tensor]], 
                                   Union[np.ndarray, torch.Tensor]],
                        factors: Union[Dict[str, List[Union[np.ndarray, torch.Tensor]]], 
                                     List[Union[np.ndarray, torch.Tensor]]],
                        device: str = 'cpu',
                        return_numpy: bool = True) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Processes multiple subjects to compute their modal tensors.

    .. deprecated:: 0.1.0
       Use :py:meth:`BatchModalProcessor.process_multiple_subjects` instead.

    Args:
        cores (Union[Dict[str, Union[np.ndarray, torch.Tensor]], Union[np.ndarray,
            torch.Tensor]]): The core tensors for the subjects.
        factors (Union[Dict[str, List[Union[np.ndarray, torch.Tensor]]],
            List[Union[np.ndarray, torch.Tensor]]]): The factor matrices for the
            subjects.
        device (str, optional): The device to use for processing. Defaults to
            'cpu'.
        return_numpy (bool, optional): If `True`, returns the result as a NumPy
            array. Defaults to `True`.

    Returns:
        Dict[str, Union[np.ndarray, torch.Tensor]]: A dictionary of the
        computed modal tensors.
    """
    warnings.warn(
        "process_all_subjects is deprecated. Use BatchModalProcessor.process_multiple_subjects() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    config = ModalProcessorConfig(device=device, return_numpy=return_numpy)
    processor = BatchModalProcessor(config)
    return processor.process_multiple_subjects(cores, factors)


def stack_all_modes(modal_tensors: Dict[str, Union[np.ndarray, torch.Tensor]],
                   device: str = 'cpu',
                   return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """Stacks modal tensors from multiple subjects.

    .. deprecated:: 0.1.0
       Use :py:meth:`ModalTensorStacker.stack_modal_tensors` instead.

    Args:
        modal_tensors (Dict[str, Union[np.ndarray, torch.Tensor]]): A dictionary
            of modal tensors from multiple subjects.
        device (str, optional): The device to use for processing. Defaults to
            'cpu'.
        return_numpy (bool, optional): If `True`, returns the result as a NumPy
            array. Defaults to `True`.

    Returns:
        Union[np.ndarray, torch.Tensor]: A stacked tensor containing all time
        slices.
    """
    warnings.warn(
        "stack_all_modes is deprecated. Use ModalTensorStacker.stack_modal_tensors() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    config = ModalProcessorConfig(device=device, return_numpy=return_numpy)
    stacker = ModalTensorStacker(config)
    return stacker.stack_modal_tensors(modal_tensors)
