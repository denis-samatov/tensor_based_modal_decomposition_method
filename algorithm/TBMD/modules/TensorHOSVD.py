import tensorly as tl
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures 
import torch
import logging
import os
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from typing import Union, List, Dict, Optional, Tuple, TypeVar, Generic, Protocol
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from TBMD.utils.utils import to_torch_tensor, get_torch_device

# Create logger (logging configuration should be done at application entry point)
logger = logging.getLogger(__name__)

# Type definitions
TensorType = TypeVar('TensorType', bound=torch.Tensor)

class TensorLike(Protocol):
    """Protocol for tensor-like objects"""
    shape: Tuple[int, ...]
    
    def norm(self) -> float:
        """Compute tensor norm"""
        ...

# Custom exceptions
class TensorDecompositionError(Exception):
    """Base exception for tensor decomposition errors"""
    pass

class InvalidRankError(TensorDecompositionError):
    """Invalid rank values provided"""
    pass

class StateError(TensorDecompositionError):
    """Invalid state for operation"""
    pass

class ValidationError(TensorDecompositionError):
    """Input validation failed"""
    pass

# Constants
DEFAULT_EPSILON = 1e-2
DEFAULT_MIDDLE_SLICE_INDEX = lambda shape: shape[2] // 2
MIN_RANK = 1
MAX_THREADS = 8

# State management
class DecomposerState(Enum):
    """States of the decomposer"""
    INITIALIZED = "initialized"
    DECOMPOSED = "decomposed" 
    RECONSTRUCTED = "reconstructed"

@dataclass
class DecompositionResult:
    """Result of Tucker decomposition"""
    core: torch.Tensor
    factors: List[torch.Tensor]
    
@dataclass
class ReconstructionResult:
    """Result of tensor reconstruction"""
    tensor: torch.Tensor
    error: float

# Processing strategies
class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies"""
    
    @abstractmethod
    def process_decomposition(self, tensors: Dict[str, torch.Tensor], 
                            decomposer_func) -> Dict[str, DecompositionResult]:
        """Process decomposition for multiple tensors"""
        pass
    
    @abstractmethod
    def process_reconstruction(self, cores: Dict[str, torch.Tensor],
                             factors: Dict[str, List[torch.Tensor]],
                             original_tensors: Dict[str, torch.Tensor]) -> Dict[str, ReconstructionResult]:
        """Process reconstruction for multiple tensors"""
        pass

class CPUStrategy(ProcessingStrategy):
    """CPU-based parallel processing strategy"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(MAX_THREADS, os.cpu_count() or 4)
    
    def process_decomposition(self, tensors: Dict[str, torch.Tensor], 
                            decomposer_func) -> Dict[str, DecompositionResult]:
        """Process decomposition using ThreadPoolExecutor"""
        results = {}
        
        def decompose_single(item: Tuple[str, torch.Tensor]) -> Tuple[str, DecompositionResult]:
            key, tensor = item
            core, factors = decomposer_func(tensor)
            return key, DecompositionResult(core=core, factors=factors)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(decompose_single, item) for item in tensors.items()]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, result = future.result()
                    results[key] = result
                except Exception as e:
                    logger.error(f"Decomposition failed for tensor: {e}")
                    raise TensorDecompositionError(f"Decomposition failed: {e}")
        
        return results
    
    def process_reconstruction(self, cores: Dict[str, torch.Tensor],
                             factors: Dict[str, List[torch.Tensor]],
                             original_tensors: Dict[str, torch.Tensor]) -> Dict[str, ReconstructionResult]:
        """Process reconstruction using ThreadPoolExecutor"""
        results = {}
        
        def reconstruct_single(key: str) -> Tuple[str, ReconstructionResult]:
            reconstructed = tucker_to_tensor((cores[key], factors[key]))
            error = float(tl.norm(original_tensors[key] - reconstructed) / tl.norm(original_tensors[key]))
            return key, ReconstructionResult(tensor=reconstructed, error=error)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(reconstruct_single, key) for key in cores.keys()]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, result = future.result()
                    results[key] = result
                except Exception as e:
                    logger.error(f"Reconstruction failed for tensor {key}: {e}")
                    raise TensorDecompositionError(f"Reconstruction failed: {e}")
        
        return results

class GPUStrategy(ProcessingStrategy):
    """GPU-based sequential processing strategy with CPU fallback"""
    
    def __init__(self, fallback_to_cpu: bool = True):
        self.fallback_to_cpu = fallback_to_cpu
        self._cpu_strategy = None
    
    def _get_cpu_strategy(self):
        """Lazy initialization of CPU strategy for fallback"""
        if self._cpu_strategy is None:
            self._cpu_strategy = CPUStrategy()
        return self._cpu_strategy
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Clear MPS cache if available (PyTorch 2.0+)
            try:
                torch.mps.empty_cache()
            except AttributeError:
                # Fallback for older PyTorch versions
                pass
    
    def _is_memory_error(self, error: Exception) -> bool:
        """Check if error is related to GPU memory"""
        error_str = str(error).lower()
        return any(keyword in error_str for keyword in [
            'out of memory', 'memory', 'mps backend out of memory',
            'cuda out of memory', 'allocation'
        ])
    
    def _get_gpu_memory_info(self) -> tuple:
        """Get GPU memory usage information"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have direct memory query methods, return estimates
            return 0, 0  # Can't easily query MPS memory
        return 0, 0
    
    def _should_use_cpu_fallback(self, tensor_size_mb: float, threshold_mb: float = 1000) -> bool:
        """Determine if tensor is too large and should use CPU"""
        return tensor_size_mb > threshold_mb
    
    def process_decomposition(self, tensors: Dict[str, torch.Tensor], 
                            decomposer_func) -> Dict[str, DecompositionResult]:
        """Process decomposition sequentially on GPU with CPU fallback"""
        results = {}
        
        for key, tensor in tensors.items():
            try:
                # Clear memory before processing
                self._clear_gpu_memory()
                core, factors = decomposer_func(tensor)
                results[key] = DecompositionResult(core=core, factors=factors)
            except Exception as e:
                if self._is_memory_error(e) and self.fallback_to_cpu:
                    logger.warning(f"GPU memory error for tensor {key}, falling back to CPU: {e}")
                    self._clear_gpu_memory()
                    # Move tensor to CPU and process
                    cpu_tensor = tensor.cpu()
                    try:
                        core, factors = decomposer_func(cpu_tensor)
                        results[key] = DecompositionResult(core=core, factors=factors)
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed for tensor {key}: {cpu_e}")
                        raise TensorDecompositionError(f"Both GPU and CPU decomposition failed: {e}")
                else:
                    logger.error(f"GPU decomposition failed for tensor {key}: {e}")
                    raise TensorDecompositionError(f"GPU decomposition failed: {e}")
        
        return results
    
    def process_reconstruction(self, cores: Dict[str, torch.Tensor],
                             factors: Dict[str, List[torch.Tensor]],
                             original_tensors: Dict[str, torch.Tensor]) -> Dict[str, ReconstructionResult]:
        """Process reconstruction sequentially on GPU with CPU fallback"""
        results = {}
        
        for key in cores.keys():
            try:
                # Clear memory before processing
                self._clear_gpu_memory()
                reconstructed = tucker_to_tensor((cores[key], factors[key]))
                error = float(tl.norm(original_tensors[key] - reconstructed) / tl.norm(original_tensors[key]))
                results[key] = ReconstructionResult(tensor=reconstructed, error=error)
            except Exception as e:
                if self._is_memory_error(e) and self.fallback_to_cpu:
                    logger.warning(f"GPU memory error for tensor {key}, falling back to CPU: {e}")
                    self._clear_gpu_memory()
                    # Move tensors to CPU and process
                    try:
                        cpu_core = cores[key].cpu()
                        cpu_factors = [f.cpu() for f in factors[key]]
                        cpu_original = original_tensors[key].cpu()
                        
                        reconstructed = tucker_to_tensor((cpu_core, cpu_factors))
                        error = float(tl.norm(cpu_original - reconstructed) / tl.norm(cpu_original))
                        results[key] = ReconstructionResult(tensor=reconstructed, error=error)
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed for tensor {key}: {cpu_e}")
                        raise TensorDecompositionError(f"Both GPU and CPU reconstruction failed: {e}")
                else:
                    logger.error(f"GPU reconstruction failed for tensor {key}: {e}")
                    raise TensorDecompositionError(f"GPU reconstruction failed: {e}")
        
        return results

# Validation utilities
class TensorValidator:
    """Utilities for tensor validation"""
    
    @staticmethod
    def validate_ranks(ranks: Optional[Union[int, List[int]]], tensor_shape: Tuple[int, ...]) -> List[int]:
        """Validate and normalize ranks"""
        if ranks is None:
            if not tensor_shape:
                raise ValidationError("Cannot determine ranks for empty tensor shape")
            return [min(tensor_shape)] * len(tensor_shape)
        
        if isinstance(ranks, int):
            if ranks < MIN_RANK:
                raise InvalidRankError(f"Rank must be >= {MIN_RANK}, got {ranks}")
            if ranks > min(tensor_shape):
                raise InvalidRankError(f"Rank {ranks} exceeds minimum tensor dimension {min(tensor_shape)}")
            return [ranks] * len(tensor_shape)
        
        if isinstance(ranks, list):
            if len(ranks) != len(tensor_shape):
                raise InvalidRankError(f"Ranks list length {len(ranks)} must match tensor modes {len(tensor_shape)}")
            
            for i, rank in enumerate(ranks):
                if rank < MIN_RANK:
                    raise InvalidRankError(f"Rank at position {i} must be >= {MIN_RANK}, got {rank}")
                if rank > tensor_shape[i]:
                    raise InvalidRankError(f"Rank {rank} at position {i} exceeds dimension {tensor_shape[i]}")
            
            return ranks
        
        raise ValidationError(f"Ranks must be None, int, or list of ints, got {type(ranks)}")
    
    @staticmethod
    def validate_epsilon(epsilon: float) -> float:
        """Validate epsilon parameter"""
        if not isinstance(epsilon, (int, float)):
            raise ValidationError(f"Epsilon must be numeric, got {type(epsilon)}")
        if epsilon <= 0:
            raise ValidationError(f"Epsilon must be positive, got {epsilon}")
        return float(epsilon)
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, min_dims: int = 2) -> None:
        """Validate tensor shape"""
        if len(tensor.shape) < min_dims:
            raise ValidationError(f"Tensor must have at least {min_dims} dimensions, got {len(tensor.shape)}")
        if any(dim <= 0 for dim in tensor.shape):
            raise ValidationError(f"All tensor dimensions must be positive, got shape {tensor.shape}")

# Core classes
class TensorProcessor:
    """Handles tensor management and device operations"""
    
    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.device = get_torch_device(device)
        self.dtype = dtype
        logger.info(f"TensorProcessor initialized with device: {self.device}, dtype: {self.dtype}")
    
    def process_tensors(self, tensors: Union[torch.Tensor, np.ndarray, tl.tensor, 
                                          Dict[str, Union[torch.Tensor, np.ndarray, tl.tensor]]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process and convert tensors to the target device and dtype"""
        try:
            if isinstance(tensors, dict):
                processed = {}
                for key, tensor in tensors.items():
                    processed_tensor = to_torch_tensor(tensor, device=self.device, dtype=self.dtype)
                    TensorValidator.validate_tensor_shape(processed_tensor)
                    processed[key] = processed_tensor
                return processed
            else:
                processed_tensor = to_torch_tensor(tensors, device=self.device, dtype=self.dtype)
                TensorValidator.validate_tensor_shape(processed_tensor)
                return processed_tensor
        except Exception as e:
            raise ValidationError(f"Failed to process tensors: {e}")

class TuckerDecomposerCore:
    """Handles Tucker decomposition operations"""
    
    def __init__(self, ranks: Optional[Union[int, List[int]]] = None,
                 epsilon: float = DEFAULT_EPSILON,
                 random_state: Optional[int] = None):
        self.ranks = ranks
        self.epsilon = TensorValidator.validate_epsilon(epsilon)
        self.random_state = random_state
        logger.info(f"TuckerDecomposerCore initialized with epsilon: {self.epsilon}")
    
    def decompose_single(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decompose a single tensor"""
        TensorValidator.validate_tensor_shape(tensor)
        ranks = TensorValidator.validate_ranks(self.ranks, tensor.shape)
        
        try:
            return tucker(tensor, rank=ranks, init='svd', tol=self.epsilon, random_state=self.random_state)
        except Exception as e:
            raise TensorDecompositionError(f"Tucker decomposition failed: {e}")
    
    def decompose_collection(self, tensors: Dict[str, torch.Tensor], 
                           strategy: ProcessingStrategy) -> Dict[str, DecompositionResult]:
        """Decompose a collection of tensors using the specified strategy"""
        if not tensors:
            raise ValidationError("Cannot decompose empty tensor collection")
        
        return strategy.process_decomposition(tensors, self.decompose_single)

class TensorReconstructor:
    """Handles tensor reconstruction operations"""
    
    @staticmethod
    def reconstruct_single(core: torch.Tensor, factors: List[torch.Tensor], 
                          original: torch.Tensor) -> ReconstructionResult:
        """Reconstruct a single tensor and compute error"""
        try:
            reconstructed = tucker_to_tensor((core, factors))
            error = float(tl.norm(original - reconstructed) / tl.norm(original))
            return ReconstructionResult(tensor=reconstructed, error=error)
        except Exception as e:
            raise TensorDecompositionError(f"Reconstruction failed: {e}")
    
    @staticmethod
    def reconstruct_collection(cores: Dict[str, torch.Tensor],
                             factors: Dict[str, List[torch.Tensor]],
                             original_tensors: Dict[str, torch.Tensor],
                             strategy: ProcessingStrategy) -> Dict[str, ReconstructionResult]:
        """Reconstruct a collection of tensors using the specified strategy"""
        if not cores or not factors or not original_tensors:
            raise ValidationError("Cannot reconstruct with empty inputs")
        
        return strategy.process_reconstruction(cores, factors, original_tensors)

class TensorVisualizer:
    """Handles tensor visualization"""
    
    @staticmethod
    def visualize_single(original: torch.Tensor, reconstructed: torch.Tensor, 
                        title: str = "Tensor Comparison") -> None:
        """Visualize original vs reconstructed tensor"""
        if len(original.shape) != 3:
            raise ValidationError("Can only visualize 3D tensors")
        
        middle_idx = DEFAULT_MIDDLE_SLICE_INDEX(original.shape)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original[:, :, middle_idx].detach().cpu().numpy(), cmap="gray")
        plt.title(f"Original {title}")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed[:, :, middle_idx].detach().cpu().numpy(), cmap="gray")
        plt.title(f"Reconstructed {title}")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_collection(original_tensors: Dict[str, torch.Tensor],
                           reconstructed_tensors: Dict[str, torch.Tensor],
                           subjects: Optional[List[str]] = None) -> None:
        """Visualize a collection of tensors"""
        subjects = subjects or list(original_tensors.keys())
        
        for subject in subjects:
            if subject not in original_tensors or subject not in reconstructed_tensors:
                logger.warning(f"Subject {subject} not found in tensor collections")
                continue
            
            TensorVisualizer.visualize_single(
                original_tensors[subject], 
                reconstructed_tensors[subject],
                f"Subject {subject}"
            )

# Main interface class
class TuckerDecomposerInterface:
    """
    Main interface for Tucker decomposition with improved architecture.
    
    This class provides a clean API while delegating responsibilities to specialized components.
    """
    
    def __init__(self, 
                tensors: Union[torch.Tensor, np.ndarray, tl.tensor, Dict[str, Union[torch.Tensor, np.ndarray, tl.tensor]]],
                ranks: Optional[Union[int, List[int]]] = None,
                epsilon: float = DEFAULT_EPSILON,
                random_state: Optional[int] = None,
                device: str = 'cpu',
                dtype: torch.dtype = torch.float32,
                max_workers: Optional[int] = None):
        """
        Initialize Tucker decomposer with improved architecture.
        
        Args:
            tensors: Input tensor(s) for decomposition
            ranks: Tucker ranks (None for auto, int for uniform, list for per-mode)
            epsilon: Convergence tolerance
            random_state: Random seed for reproducibility
            device: Computing device ('cpu', 'cuda', 'mps')
            dtype: Tensor data type
            max_workers: Maximum workers for parallel processing
        """
        # Initialize componentsы
        self.processor = TensorProcessor(device, dtype)
        self.decomposer = TuckerDecomposerCore(ranks, epsilon, random_state)
        
        # Process input tensors
        self.tensors = self.processor.process_tensors(tensors)
        self.is_collection = isinstance(self.tensors, dict)
        
        # Initialize state
        self.state = DecomposerState.INITIALIZED
        
        # Choose processing strategy
        if self.processor.device.type == 'cpu':
            self.strategy = CPUStrategy(max_workers)
        else:
            self.strategy = GPUStrategy()
        
        # Results storage
        self._cores: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
        self._factors: Optional[Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]] = None
        self._reconstructed: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
        self._errors: Optional[Union[float, Dict[str, float]]] = None
        
        logger.info(f"TuckerDecomposerInterface initialized in {'collection' if self.is_collection else 'single'} mode")
    
    def decompose(self) -> None:
        """Perform Tucker decomposition"""
        if self.state != DecomposerState.INITIALIZED:
            raise StateError(f"Cannot decompose in state {self.state.value}")
        
        try:
            if self.is_collection:
                results = self.decomposer.decompose_collection(self.tensors, self.strategy)
                self._cores = {k: v.core for k, v in results.items()}
                self._factors = {k: v.factors for k, v in results.items()}
            else:
                core, factors = self.decomposer.decompose_single(self.tensors)
                self._cores = core
                self._factors = factors
            
            self.state = DecomposerState.DECOMPOSED
            logger.info("Decomposition completed successfully")
        
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            raise
    
    def reconstruct(self) -> None:
        """Reconstruct tensors from decomposition"""
        if self.state != DecomposerState.DECOMPOSED:
            raise StateError(f"Cannot reconstruct in state {self.state.value}. Call decompose() first.")
        
        try:
            if self.is_collection:
                results = TensorReconstructor.reconstruct_collection(
                    self._cores, self._factors, self.tensors, self.strategy
                )
                self._reconstructed = {k: v.tensor for k, v in results.items()}
                self._errors = {k: v.error for k, v in results.items()}
            else:
                result = TensorReconstructor.reconstruct_single(
                    self._cores, self._factors, self.tensors
                )
                self._reconstructed = result.tensor
                self._errors = result.error
            
            self.state = DecomposerState.RECONSTRUCTED
            logger.info("Reconstruction completed successfully")
        
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise
    
    def visualize(self, subjects: Optional[List[str]] = None) -> None:
        """Visualize results"""
        if self.state != DecomposerState.RECONSTRUCTED:
            raise StateError(f"Cannot visualize in state {self.state.value}. Call reconstruct() first.")
        
        try:
            if self.is_collection:
                TensorVisualizer.visualize_collection(
                    self.tensors, self._reconstructed, subjects
                )
            else:
                TensorVisualizer.visualize_single(self.tensors, self._reconstructed)
        
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise
    
    # Property accessors with validation
    @property
    def cores(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get decomposition cores"""
        if self.state == DecomposerState.INITIALIZED:
            raise StateError("Call decompose() first")
        return self._cores
    
    @property
    def factors(self) -> Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """Get decomposition factors"""
        if self.state == DecomposerState.INITIALIZED:
            raise StateError("Call decompose() first")
        return self._factors
    
    @property
    def reconstructed_tensors(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get reconstructed tensors"""
        if self.state != DecomposerState.RECONSTRUCTED:
            raise StateError("Call reconstruct() first")
        return self._reconstructed
    
    @property
    def reconstruction_errors(self) -> Union[float, Dict[str, float]]:
        """Get reconstruction errors"""
        if self.state != DecomposerState.RECONSTRUCTED:
            raise StateError("Call reconstruct() first")
        return self._errors
    
    def set_ranks(self, ranks: Optional[Union[int, List[int]]]) -> None:
        """Update ranks (only allowed in INITIALIZED state)"""
        if self.state != DecomposerState.INITIALIZED:
            raise StateError("Cannot change ranks after decomposition")
        self.decomposer.ranks = ranks

# Maintain backward compatibility
TuckerDecomposer = TuckerDecomposerInterface