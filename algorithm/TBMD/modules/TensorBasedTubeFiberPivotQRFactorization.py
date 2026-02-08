"""
Tensor-based Tube Fiber-Pivot QR Factorization Module

This module implements Algorithm 2 from the tensor-based modal decomposition paper,
providing numerically stable and performant QR factorization with tube pivoting
for N-dimensional tensors with optional uniform sensor distribution.

Key improvements:
- Numerically stable Householder transformations
- Separated responsibilities following SOLID principles
- Comprehensive input validation
- Optimized performance for large tensors
- Scientific constants instead of magic numbers
"""

import numpy as np
import torch
import tensorly as tl
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.utils import to_torch_tensor, get_torch_device


@dataclass
class TensorQRConfig:
    """A configuration class for tensor QR decomposition.

    This class defines constants for numerical stability, distribution
    penalties, and performance optimization.

    Attributes:
        MACHINE_EPSILON_FACTOR (float): A tolerance factor for float32.
        HOUSEHOLDER_THRESHOLD (float): A threshold for Householder vector
            computation.
        ORTHOGONALITY_TOLERANCE (float): A tolerance for orthogonality checks.
        CONDITION_NUMBER_THRESHOLD (float): The maximum acceptable condition
            number.
        SLICE_PENALTY_WEIGHT (float): The weight for inter-slice balance
            penalties.
        DISTRIBUTION_PENALTY_WEIGHT (float): The weight for spatial
            distribution penalties.
        SIMILARITY_GROUPING_DECIMALS (int): The precision for similarity
            grouping.
        CHUNK_SIZE (int): The size for batched operations.
    """
    
    # Numerical stability constants
    MACHINE_EPSILON_FACTOR: float = 1e-6  # Realistic tolerance for float32
    HOUSEHOLDER_THRESHOLD: float = 1e-6   # Threshold for Householder vector computation
    ORTHOGONALITY_TOLERANCE: float = 1e-4  # Realistic tolerance for float32 with accumulation
    CONDITION_NUMBER_THRESHOLD: float = 1e12  # Maximum acceptable condition number
    
    # Distribution penalty weights (configurable for research)
    SLICE_PENALTY_WEIGHT: float = 0.8      # Weight for inter-slice balance penalty
    DISTRIBUTION_PENALTY_WEIGHT: float = 0.5  # Weight for spatial distribution penalty
    SIMILARITY_GROUPING_DECIMALS: int = 3   # Precision for similarity grouping
    
    # Performance optimization constants
    CHUNK_SIZE: int = 1000                  # Size for batched operations
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 < self.SLICE_PENALTY_WEIGHT <= 1):
            raise ValueError("SLICE_PENALTY_WEIGHT must be in (0, 1]")
        if not (0 < self.DISTRIBUTION_PENALTY_WEIGHT <= 1):
            raise ValueError("DISTRIBUTION_PENALTY_WEIGHT must be in (0, 1]")
        if self.CONDITION_NUMBER_THRESHOLD < 1:
            raise ValueError("CONDITION_NUMBER_THRESHOLD must be >= 1")


class TensorValidator:
    """Validates inputs for tensor QR decomposition.

    This class ensures that the inputs meet the numerical stability and
    algorithmic requirements for the decomposition.
    """
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, min_dims: int = 3) -> None:
        """Validates the input tensor for QR decomposition.

        Args:
            tensor (torch.Tensor): The input tensor to validate.
            min_dims (int, optional): The minimum required dimensions.
                Defaults to 3.

        Raises:
            ValueError: If the tensor does not meet the requirements.
        """
        if tensor.ndim < min_dims:
            raise ValueError(f"Tensor must have at least {min_dims} dimensions, got {tensor.ndim}")
            
        if not torch.isfinite(tensor).all():
            raise ValueError("Tensor contains NaN or infinite values")
            
        if tensor.numel() == 0:
            raise ValueError("Tensor cannot be empty")
            
        # Check for numerical rank deficiency
        min_dim = min(tensor.shape)
        if min_dim > 0:
            # Use SVD to check rank for small tensors
            if tensor.numel() < 10000:  # Only for small tensors due to computational cost
                try:
                    flat_tensor = tensor.flatten(-2, -1)
                    if flat_tensor.shape[-1] > 0:
                        _, s, _ = torch.svd(flat_tensor[..., :min(flat_tensor.shape[-2:])].float())
                        if s.min() < TensorQRConfig.MACHINE_EPSILON_FACTOR:
                            print("Warning: Tensor may be numerically rank-deficient")
                except RuntimeError:
                    pass  # SVD might fail for some tensor shapes
    
    @staticmethod
    def validate_sensor_count(N: int, k: int) -> None:
        """Validates the sensor count parameter.

        Args:
            N (int): The number of sensors.
            k (int): The tube dimension.
        """
        import numpy as np
        
        # Convert to Python int if it's a numpy integer type
        if hasattr(N, 'item'):  # numpy scalar
            N = N.item()
        
        # Check if it's an integer type (Python int or numpy integer)
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError(f"N must be a positive integer, got {N} (type: {type(N)})")
        
        # Ensure N is Python int for consistency
        N = int(N)
        
        if N > k:
            print(f"Warning: N ({N}) exceeds tube dimension k ({k}). "
                  f"Only {k} sensors can be effectively placed.")
    
    @staticmethod
    def validate_rejection_domain(rejection_domain: torch.Tensor, spatial_shape: Tuple[int, ...]) -> None:
        """Validates the rejection domain mask.

        Args:
            rejection_domain (torch.Tensor): The rejection domain mask.
            spatial_shape (Tuple[int, ...]): The spatial shape of the tensor.
        """
        if rejection_domain.shape != spatial_shape:
            raise ValueError(f"Rejection domain shape {rejection_domain.shape} "
                           f"must match spatial shape {spatial_shape}")
        if rejection_domain.dtype != torch.bool:
            raise ValueError("Rejection domain must have boolean dtype")


class NumericallyStableOperations:
    """Performs numerically stable operations for tensor QR decomposition.

    This class implements stable algorithms to avoid catastrophic cancellation
    and maintain numerical precision during the computation.

    Args:
        config (TensorQRConfig): The configuration with numerical parameters.
        device (torch.device): The device for computations.
        dtype (torch.dtype): The data type for computations.
    """
    
    def __init__(self, config: TensorQRConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
    
    def compute_householder_vector(self, v: torch.Tensor) -> torch.Tensor:
        """Computes the Householder reflection vector.

        This method implements the formula from Algorithm 2 of the TBMD paper.

        Args:
            v (torch.Tensor): The input vector (tube) for the Householder
                transformation.

        Returns:
            torch.Tensor: The normalized Householder vector `u`.
        """
        if v.numel() == 0:
            return torch.zeros_like(v)
            
        # Compute σ = ||v||₂ 
        sigma = torch.norm(v, p=2)
        
        if sigma < self.config.HOUSEHOLDER_THRESHOLD:
            return torch.zeros_like(v)
        
        # Compute sign(v[0]) with numerical stability
        if torch.abs(v[0]) < self.config.MACHINE_EPSILON_FACTOR:
            sign_v0 = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            sign_v0 = torch.sign(v[0])
        
        # Compute u = v + sign(v[0]) * σ * e₁
        u = v.clone()
        u[0] = u[0] + sign_v0 * sigma
        
        # Compute denominator: √(2σ(σ + |v[0]|))
        denom = torch.sqrt(2 * sigma * (sigma + torch.abs(v[0])))
        
        if denom < self.config.MACHINE_EPSILON_FACTOR:
            return torch.zeros_like(v)
            
        # Return normalized vector: u / √(2σ(σ + |v[0]|))
        return u / denom
    
    def check_orthogonality(self, Q: torch.Tensor) -> Tuple[bool, float]:
        """Checks the orthogonality of a matrix with numerical tolerance.

        Args:
            Q (torch.Tensor): The matrix to check.

        Returns:
            Tuple[bool, float]: A tuple containing a boolean indicating if the
            matrix is orthogonal and the maximum deviation from orthogonality.
        """
        if Q.numel() == 0:
            return True, 0.0
            
        QTQ = Q.T @ Q
        I = torch.eye(Q.shape[1], device=self.device, dtype=self.dtype)
        deviation = torch.max(torch.abs(QTQ - I)).item()
        
        is_orthogonal = deviation < self.config.ORTHOGONALITY_TOLERANCE
        return is_orthogonal, deviation
    
    def estimate_condition_number(self, tensor: torch.Tensor) -> float:
        """Estimates the condition number for numerical stability monitoring.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            float: The estimated condition number.
        """
        try:
            # Use SVD on a representative slice for efficiency
            if tensor.ndim >= 2:
                representative_slice = tensor.flatten(-2, -1)
                if representative_slice.shape[-1] > 0:
                    _, s, _ = torch.svd(representative_slice[..., :min(representative_slice.shape[-2:])].float())
                    if s.numel() > 1 and s.min() > 0:
                        return (s.max() / s.min()).item()
        except RuntimeError:
            pass
        return 1.0  # Default safe value


class OptimizedPivotSelector:
    """An optimized pivot selector with vectorized operations.

    This class implements the pivot selection from Algorithm 2 with performance
    optimizations for large tensors and uniform distribution constraints.

    Args:
        config (TensorQRConfig): The configuration with numerical parameters.
        device (torch.device): The device for computations.
        dtype (torch.dtype): The data type for computations.
    """
    
    def __init__(self, config: TensorQRConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self._cached_max_norms = {}
    
    def select_pivot(self, 
                    R: torch.Tensor, 
                    d: int, 
                    available: torch.Tensor,
                    distribution_state: Optional[Dict] = None) -> Tuple[int, ...]:
        """Selects the optimal pivot position.

        This method uses numerical stability and efficiency considerations to
        select the best pivot.

        Args:
            R (torch.Tensor): The current R matrix.
            d (int): The current decomposition step.
            available (torch.Tensor): A boolean mask of available positions.
            distribution_state (Optional[Dict]): The state for uniform
                distribution.

        Returns:
            Tuple[int, ...]: A tuple of pivot indices.
        """
        # Compute residual norms efficiently
        norms = self._compute_residual_norms(R, d)
        
        # Apply availability mask
        norms = torch.where(available, norms, 
                          torch.tensor(float('-inf'), device=self.device, dtype=self.dtype))
        
        # Apply distribution penalties if needed
        if distribution_state is not None:
            norms = self._apply_distribution_penalties(norms, distribution_state)
        
        # Select best pivot
        flat_idx = torch.argmax(norms).item()
        return np.unravel_index(flat_idx, norms.shape)
    
    def _compute_residual_norms(self, R: torch.Tensor, d: int) -> torch.Tensor:
        """Compute residual norms efficiently using vectorized operations.

        Parameters
        ----------
        R : torch.Tensor
            The current R matrix.
        d : int
            The current decomposition step.

        Returns
        -------
        torch.Tensor
            The computed residual norms.
        """
        cache_key = (R.data_ptr(), d)
        
        if cache_key not in self._cached_max_norms:
            # Vectorized computation of residual norms
            residual = R[..., d:]
            norms = torch.sum(torch.abs(residual), dim=-1)
            self._cached_max_norms[cache_key] = norms
        
        return self._cached_max_norms[cache_key]
    
    def _apply_distribution_penalties(self, norms: torch.Tensor, state: Dict) -> torch.Tensor:
        """Apply distribution penalties using vectorized operations.

        Parameters
        ----------
        norms : torch.Tensor
            The norms to apply penalties to.
        state : Dict
            The distribution state.

        Returns
        -------
        torch.Tensor
            The norms with applied penalties.
        """
        penalties = torch.zeros_like(norms)
        
        # Cache maximum norm value for penalty scaling
        max_norm = torch.max(norms)
        
        # Vectorized slice penalty computation
        if 'slice_counts' in state and len(norms.shape) >= 3:
            penalties += self._compute_slice_penalties(norms, state['slice_counts'], max_norm)
        
        # Vectorized distribution penalty computation
        if 'sensor_placement' in state:
            penalties += self._compute_distribution_penalties(norms, state['sensor_placement'], max_norm)
        
        return norms - penalties
    
    def _compute_slice_penalties(self, norms: torch.Tensor, slice_counts: Dict[int, int], max_norm: torch.Tensor) -> torch.Tensor:
        """Compute slice balance penalties efficiently.

        Parameters
        ----------
        norms : torch.Tensor
            The norms to compute penalties for.
        slice_counts : Dict[int, int]
            The number of sensors in each slice.
        max_norm : torch.Tensor
            The maximum norm value.

        Returns
        -------
        torch.Tensor
            The computed slice penalties.
        """
        if len(norms.shape) < 3:
            return torch.zeros_like(norms)
            
        total_sensors = sum(slice_counts.values())
        
        if total_sensors == 0:
            return torch.zeros_like(norms)
            
        z_dim = norms.shape[2]
        target_per_slice = total_sensors / z_dim
        
        # Convert slice_counts to tensor efficiently
        counts = torch.zeros(z_dim, device=norms.device, dtype=norms.dtype)
        
        if slice_counts:
            # Filter keys that are within range
            valid_items = [(k, v) for k, v in slice_counts.items() if 0 <= k < z_dim]
            if valid_items:
                indices, values = zip(*valid_items)
                indices_tensor = torch.tensor(indices, device=norms.device, dtype=torch.long)
                values_tensor = torch.tensor(values, device=norms.device, dtype=norms.dtype)
                counts[indices_tensor] = values_tensor

        # Vectorized imbalance calculation
        imbalance = torch.clamp(counts - target_per_slice, min=0)

        # Calculate penalty values
        penalty_values = imbalance * self.config.SLICE_PENALTY_WEIGHT * max_norm

        # Apply penalties to the last dimension, consistent with the original loop
        # Loop over range(norms.shape[2]) and assignment to penalties[..., z]
        # implies mapping the calculated penalties to the first z_dim indices of the last dimension.
        last_dim = norms.shape[-1]

        # Prepare vector for broadcasting along the last dimension
        final_penalties = torch.zeros(last_dim, device=norms.device, dtype=norms.dtype)
        copy_len = min(z_dim, last_dim)
        final_penalties[:copy_len] = penalty_values[:copy_len]

        # Reshape for broadcasting
        shape = [1] * norms.ndim
        shape[-1] = last_dim

        return final_penalties.view(*shape).expand_as(norms).clone()
    
    def _compute_distribution_penalties(self, norms: torch.Tensor, sensor_placement: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:
        """Compute spatial distribution penalties efficiently.

        Parameters
        ----------
        norms : torch.Tensor
            The norms to compute penalties for.
        sensor_placement : torch.Tensor
            The sensor placement matrix.
        max_norm : torch.Tensor
            The maximum norm value.

        Returns
        -------
        torch.Tensor
            The computed spatial distribution penalties.
        """
        penalties = torch.zeros_like(norms)
        total_sensors = torch.sum(sensor_placement).item()
        
        if total_sensors == 0:
            return penalties
        
        # Vectorized computation for each spatial dimension
        for dim in range(len(norms.shape)):
            # Sum along all other dimensions to get density per slice in this dimension
            sum_dims = list(range(len(norms.shape)))
            sum_dims.remove(dim)
            
            if sum_dims:
                density_per_slice = torch.sum(sensor_placement, dim=tuple(sum_dims)) / total_sensors
                
                # Broadcast penalty across the dimension
                penalty_shape = [1] * len(norms.shape)
                penalty_shape[dim] = norms.shape[dim]
                
                penalty_values = density_per_slice.view(penalty_shape) * self.config.DISTRIBUTION_PENALTY_WEIGHT * max_norm
                penalties += penalty_values
        
        return penalties


class UniformDistributionManager:
    """Manages uniform sensor distribution with efficient region grouping.

    This class implements spatial and temporal distribution constraints for
    optimal sensor placement.

    Args:
        config (TensorQRConfig): The configuration with numerical parameters.
        spatial_shape (Tuple[int, ...]): The spatial shape of the tensor.
        device (torch.device): The device for computations.
    """
    
    def __init__(self, config: TensorQRConfig, spatial_shape: Tuple[int, ...], device: torch.device):
        self.config = config
        self.spatial_shape = spatial_shape
        self.device = device
        
        # Distribution tracking
        self.slice_counts: Dict[int, int] = {}
        self.dimension_counts: Dict[int, Dict[int, int]] = {
            dim: {} for dim in range(len(spatial_shape))
        }
        
        # Similar regions for efficient grouping
        self.similar_regions: Dict[tuple, List[Tuple[int, ...]]] = {}
        self.region_lookup: Dict[Tuple[int, ...], tuple] = {}
        
        # Initialize slice tracking for 3D+ tensors
        if len(spatial_shape) >= 3:
            for z in range(spatial_shape[2]):
                self.slice_counts[z] = 0
    
    def identify_similar_regions(self, tensor: torch.Tensor) -> None:
        """Identifies similar regions for distribution constraints.

        Args:
            tensor (torch.Tensor): The input tensor for region analysis.
        """
        if len(self.spatial_shape) < 3:
            return
        
        # Use first temporal slice as pattern descriptor
        pattern_tensor = tensor[..., 0].detach().cpu().numpy()
        
        # Efficient vectorized implementation
        # Reshape to (N, Z) to apply rounding in one go
        reshaped = pattern_tensor.reshape(-1, self.spatial_shape[2])
        rounded = np.round(reshaped, decimals=self.config.SIMILARITY_GROUPING_DECIMALS)

        # Convert rows to tuples (efficient iteration using tolist)
        patterns = [tuple(row) for row in rounded.tolist()]

        # Populate dictionary
        Y = self.spatial_shape[1]
        for i, pattern in enumerate(patterns):
            # i = x * Y + y
            x, y = divmod(i, Y)
            self.similar_regions.setdefault(pattern, []).append((x, y))
            self.region_lookup[(x, y)] = pattern
    
    def update_sensor_placement(self, pivot: Tuple[int, ...]) -> None:
        """Updates distribution tracking after sensor placement.

        Args:
            pivot (Tuple[int, ...]): The pivot coordinates.
        """
        # Update dimension counts
        for dim, idx in enumerate(pivot):
            self.dimension_counts[dim][idx] = self.dimension_counts[dim].get(idx, 0) + 1
        
        # Update slice counts for 3D+ tensors
        if len(self.spatial_shape) >= 3 and len(pivot) >= 3:
            z = pivot[2]
            self.slice_counts[z] = self.slice_counts.get(z, 0) + 1
    
    def mark_similar_regions_unavailable(self, pivot: Tuple[int, ...], available: torch.Tensor) -> None:
        """Marks similar regions as unavailable.

        This is a corrected version with less aggressive locking.

        Args:
            pivot (Tuple[int, ...]): The pivot coordinates.
            available (torch.Tensor): The availability mask.
        """
        if len(self.spatial_shape) < 3 or len(pivot) < 2:
            return
        
        x, y = pivot[0], pivot[1]
        pattern = self.region_lookup.get((x, y))
        
        if pattern and pattern in self.similar_regions:
            similar_positions = self.similar_regions[pattern]
            
            # ИЗМЕНЕНИЕ: блокировать только ближайшие позиции
            blocked_count = 0
            max_blocks_per_region = max(1, len(similar_positions) // 4)  # Максимум 25%
            
            for px, py in similar_positions:
                if (px, py) != (x, y) and blocked_count < max_blocks_per_region:
                    # Блокировать только тот же z-уровень
                    if len(pivot) >= 3:
                        z = pivot[2]
                        available[px, py, z] = False
                        blocked_count += 1
    
    def get_distribution_state(self, sensor_placement: torch.Tensor) -> Dict:
        """Returns the current distribution state for penalty computation.

        Args:
            sensor_placement (torch.Tensor): The sensor placement tensor.

        Returns:
            Dict: The distribution state.
        """
        return {
            'slice_counts': self.slice_counts.copy(),
            'dimension_counts': {k: v.copy() for k, v in self.dimension_counts.items()},
            'sensor_placement': sensor_placement
        }


class TensorTubeQRDecomposition:
    """An improved tensor-based QR factorization with tube pivoting.

    This class implements Algorithm 2 from the TBMD paper, with enhancements
    for numerical stability, performance, and validation.

    Args:
        tensor (Union[np.ndarray, torch.Tensor, tl.tensor]): The input tensor.
        N (int): The number of sensors to select.
        rejection_domain (Optional[Union[np.ndarray, torch.Tensor]]): A mask of
            positions that cannot host a sensor.
        random_state (Optional[int]): A seed for reproducible results.
        check_orthogonality (bool): If `True`, verifies that Q remains
            orthonormal. Defaults to `False`.
        device (str): The PyTorch device ('cpu', 'cuda', 'mps'). Defaults to 'cpu'.
        dtype (torch.dtype): The PyTorch data type. Defaults to `torch.float32`.
        uniform_distribution (bool): If `True`, enforces spatial distribution
            constraints. Defaults to `False`.
        config (Optional[TensorQRConfig]): A configuration object with
            algorithm parameters.
    """
    
    def __init__(
        self,
        tensor: Union[np.ndarray, torch.Tensor, tl.tensor],
        N: int,
        rejection_domain: Optional[Union[np.ndarray, torch.Tensor]] = None,
        random_state: Optional[int] = None,
        check_orthogonality: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        uniform_distribution: bool = False,
        config: Optional[TensorQRConfig] = None,
    ) -> None:
        # Configuration and reproducibility
        self.config = config or TensorQRConfig()
        self._setup_reproducibility(random_state)
        
        # Device and data type setup
        self.device = get_torch_device(device)
        self.dtype = dtype
        
        # Convert and validate input tensor
        self.tensor = to_torch_tensor(tensor, device=self.device, dtype=self.dtype)
        TensorValidator.validate_tensor(self.tensor)
        
        # Extract tensor properties
        self.spatial_shape: Tuple[int, ...] = self.tensor.shape[:-1]
        self.k: int = self.tensor.shape[-1]
        
        # Validate and store sensor count
        TensorValidator.validate_sensor_count(N, self.k)
        self.N = N
        
        # Setup availability mask
        self.available = self._setup_availability_mask(rejection_domain)
        
        # Initialize specialized components
        self.numerical_ops = NumericallyStableOperations(self.config, self.device, self.dtype)
        self.pivot_selector = OptimizedPivotSelector(self.config, self.device, self.dtype)
        self.uniform_distribution = uniform_distribution
        
        if self.uniform_distribution:
            self.distribution_manager = UniformDistributionManager(
                self.config, self.spatial_shape, self.device
            )
            self.distribution_manager.identify_similar_regions(self.tensor)
        
        # Algorithm state
        self.check_orthogonality = check_orthogonality
        self._reset_results()
        
        # Monitor numerical health
        condition_number = self.numerical_ops.estimate_condition_number(self.tensor)
        if condition_number > self.config.CONDITION_NUMBER_THRESHOLD:
            print(f"Warning: High condition number ({condition_number:.2e}) detected. "
                  f"Results may be numerically unstable.")
    
    def _setup_reproducibility(self, random_state: Optional[int]) -> None:
        """Setup reproducible random number generation."""
        if random_state is not None:
            np.random.seed(random_state)
            tl.check_random_state(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
                torch.cuda.manual_seed_all(random_state)
    
    def _setup_availability_mask(self, rejection_domain: Optional[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        """Setup and validate availability mask for sensor placement."""
        if rejection_domain is None:
            return torch.ones(self.spatial_shape, dtype=torch.bool, device=self.device)
        
        rejection_tensor = to_torch_tensor(rejection_domain, dtype=torch.bool, device=self.device)
        TensorValidator.validate_rejection_domain(rejection_tensor, self.spatial_shape)
        return rejection_tensor.clone()
    
    def _reset_results(self) -> None:
        """Reset algorithm results for fresh computation."""
        self.P: Optional[torch.Tensor] = None  # Sensor placement indicator
        self.Q: Optional[torch.Tensor] = None  # Orthogonal matrix
        self.R: Optional[torch.Tensor] = None  # Upper triangular result
        self._orthogonality_history: List[float] = []
    
    def factorize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs tensor QR factorization with tube pivoting.

        This method implements Algorithm 2 with numerical stability and
        performance optimizations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple `(P, Q, R)`,
            where `P` is a binary tensor of sensor positions, `Q` is a `k x k`
            orthogonal matrix, and `R` is the transformed tensor.

        Raises:
            RuntimeError: If the algorithm fails due to numerical issues.
        """
        try:
            # Initialize algorithm state
            self.R = self.tensor.clone()
            self.Q = torch.eye(self.k, device=self.device, dtype=self.dtype)
            self.P = torch.zeros(self.spatial_shape, device=self.device, dtype=torch.int32)
            self._orthogonality_history.clear()
            
            # Reset availability and distribution tracking
            available = self.available.clone()
            if self.uniform_distribution:
                self.distribution_manager.slice_counts = {z: 0 for z in range(self.spatial_shape[2])} if len(self.spatial_shape) >= 3 else {}
                self.distribution_manager.dimension_counts = {dim: {} for dim in range(len(self.spatial_shape))}
            
            # Main factorization loop
            successful_steps = 0
            for d in range(min(self.N, self.k)):
                try:
                    success = self._factorization_step(d, available)
                    if not success:
                        print(f"Warning: Factorization stopped early at step {d} due to numerical issues")
                        break
                    successful_steps += 1
                    
                except RuntimeError as e:
                    print(f"Warning: Numerical error at step {d}: {e}")
                    break
            
            if successful_steps == 0:
                raise RuntimeError("Factorization failed: no successful steps completed")
            
            # Final orthogonality check
            is_orthogonal, deviation = self.numerical_ops.check_orthogonality(self.Q)
            if not is_orthogonal:
                print(f"Warning: Final Q matrix not orthogonal (deviation: {deviation:.2e})")
            
            actual_rank = torch.sum(self.P).item()
            print(f"QR Factorization completed:")
            print(f"  Requested sensors: {self.N}")
            print(f"  Actual rank: {actual_rank}")
            print(f"  Success rate: {actual_rank/self.N*100:.1f}%")
            print(f"  Early stops: {self.N - successful_steps}")
            
            if actual_rank < self.N * 0.5:  # Менее 50% успеха
                print("WARNING: Low rank achieved - consider relaxing thresholds!")
            
            return self.P, self.Q, self.R
            
        except Exception as e:
            self._reset_results()
            raise RuntimeError(f"Factorization failed: {e}")
    
    def _factorization_step(self, d: int, available: torch.Tensor) -> bool:
        """Perform a single step of the QR factorization algorithm.
        
        Parameters
        ----------
        d : int
            The current step index.
        available : torch.Tensor
            The availability mask for sensor placement.
            
        Returns
        -------
        bool
            `True` if the step completed successfully, `False` otherwise.
        """
        # Get distribution state for pivot selection
        distribution_state = None
        if self.uniform_distribution:
            distribution_state = self.distribution_manager.get_distribution_state(self.P)
        
        # Select pivot position
        pivot = self.pivot_selector.select_pivot(self.R, d, available, distribution_state)
        
        # Extract tube at pivot position
        tube = self.R[pivot + (slice(d, None),)]
        
        # Check for numerical insignificance
        tube_norm = torch.norm(tube)
        if tube_norm < self.config.MACHINE_EPSILON_FACTOR:
            return False  # Skip insignificant tube
        
        # Update sensor placement and availability
        self.P[pivot] = 1
        available[pivot] = False
        
        # Update distribution tracking
        if self.uniform_distribution:
            self.distribution_manager.update_sensor_placement(pivot)
            self.distribution_manager.mark_similar_regions_unavailable(pivot, available)
        
        # Compute Householder vector
        u = self.numerical_ops.compute_householder_vector(tube)
        u_norm = torch.norm(u)
        
        if u_norm < self.config.HOUSEHOLDER_THRESHOLD:
            return False  # Skip degenerate transformation
        
        # Apply Householder transformation to R
        self._apply_householder_to_R(u, d)
        
        # Apply Householder transformation to Q
        self._apply_householder_to_Q(u, d)
        
        # Check orthogonality if requested
        if self.check_orthogonality:
            is_orthogonal, deviation = self.numerical_ops.check_orthogonality(self.Q)
            self._orthogonality_history.append(deviation)
            
            if not is_orthogonal:
                print(f"Warning: Q lost orthogonality at step {d} (deviation: {deviation:.2e})")
        
        return True
    
    def _apply_householder_to_R(self, u: torch.Tensor, d: int) -> None:
        """Apply the Householder transformation to the R matrix efficiently.

        This method applies H = I - 2uu^T to each tube (fiber) R[x, y, d:k].
        According to Algorithm 2, R_{x,y,d:k} is updated as:
        R_{x,y,d:k} - 2u^T * R_{x,y,d:k}.

        For each spatial position (x, y), the tube t = R[x, y, d:k] is
        transformed as: t_new = t - 2 * (u^T @ t) * u.

        Parameters
        ----------
        u : torch.Tensor
            The Householder vector.
        d : int
            The current decomposition step.
        """
        # Work with submatrix R[..., d:] for efficiency
        sub_R = self.R[..., d:]
        original_shape = sub_R.shape
        
        # Reshape to (spatial_size, k-d) for vectorized operations
        flat_R = sub_R.reshape(-1, self.k - d)  # (spatial_size, k-d)
        
        # For each tube (row in flat_R), compute (u^T @ tube) * u
        # This is vectorized: flat_R @ u gives us u^T @ tube for each tube
        uT_tubes = flat_R @ u  # (spatial_size,) - dot product of u with each tube
        
        # Apply transformation: tube := tube - 2 * (u^T @ tube) * u
        # Broadcasting: (spatial_size, 1) * (1, k-d) -> (spatial_size, k-d)
        flat_R -= 2 * uT_tubes.unsqueeze(1) * u.unsqueeze(0)
        
        # Reshape back to original tensor structure
        self.R[..., d:] = flat_R.reshape(original_shape)
    
    def _apply_householder_to_Q(self, u: torch.Tensor, d: int) -> None:
        """Apply the Householder transformation to the Q matrix efficiently.

        This method applies H = I - 2uu^T to Q from the right, so Q becomes Q*H.
        According to Algorithm 2, Q_{:,d:k} is updated as:
        Q_{:,d:k} - 2 * Q_{:,d:k} * u * u^T.

        Parameters
        ----------
        u : torch.Tensor
            The Householder vector.
        d : int
            The current decomposition step.
        """
        # Work with submatrix Q[:, d:] for efficiency
        Q_block = self.Q[:, d:]
        
        # Apply transformation: Q := Q - 2 * (Q @ u) @ u^T
        Qu = Q_block @ u  # (k,) vector - Q times u
        Q_block -= 2 * Qu.unsqueeze(1) * u.unsqueeze(0)  # Q := Q - 2*(Q@u)@u^T
    
    def check_factorization(self, tol: float = 1e-6) -> Tuple[bool, float, Dict[str, float]]:
        """Performs a validation of the factorization quality.

        Args:
            tol (float, optional): The tolerance for validation checks.
                Defaults to 1e-6.

        Returns:
            Tuple[bool, float, Dict[str, float]]: A tuple containing a boolean
            indicating if the factorization is valid, the relative error, and a
            dictionary of metrics.
        """
        if any(x is None for x in (self.P, self.Q, self.R)):
            raise ValueError("Run factorize() first")
        
        metrics = {}
        
        # Check orthogonality
        is_orthogonal, ortho_deviation = self.numerical_ops.check_orthogonality(self.Q)
        metrics['orthogonality_deviation'] = ortho_deviation
        
        # Check reconstruction error
        RQ_T = torch.tensordot(self.R, self.Q.T, dims=([-1], [0]))
        reconstruction_error = torch.norm(self.tensor - RQ_T) / torch.norm(self.tensor)
        metrics['relative_reconstruction_error'] = float(reconstruction_error.item())
        
        # Check sensor count
        actual_sensors = torch.sum(self.P).item()
        metrics['sensor_count'] = actual_sensors
        metrics['sensor_efficiency'] = actual_sensors / self.N
        
        # Overall validation
        is_valid = (is_orthogonal and 
                   reconstruction_error < tol and 
                   actual_sensors > 0)
        
        return is_valid, float(reconstruction_error.item()), metrics
    
    def get_algorithm_info(self) -> Dict[str, any]:
        """Returns information about the algorithm's state and performance.

        Returns:
            Dict[str, any]: A dictionary of the algorithm's state and
            performance metrics.
        """
        info = {
            'tensor_shape': self.tensor.shape,
            'spatial_shape': self.spatial_shape,
            'tube_dimension': self.k,
            'requested_sensors': self.N,
            'uniform_distribution': self.uniform_distribution,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'config': self.config.__dict__,
        }
        
        if self.P is not None:
            info['actual_sensors'] = torch.sum(self.P).item()
            
        if self._orthogonality_history:
            info['orthogonality_history'] = self._orthogonality_history
            info['max_orthogonality_deviation'] = max(self._orthogonality_history)
            
        return info
    
    def visualize_sensor_placement(self, figsize: Optional[Tuple[int, int]] = None) -> None:
        """Visualizes sensor placement with enhanced graphics and statistics.

        Args:
            figsize (Optional[Tuple[int, int]]): The figure size in inches.
        """
        if self.P is None:
            raise ValueError("Run factorize() first")
        
        p = self.P.detach().cpu().numpy()
        
        # Determine visualization approach based on tensor dimensionality
        if p.ndim == 2:
            self._visualize_2d_placement(p, figsize)
        elif p.ndim >= 3:
            self._visualize_3d_placement(p, figsize)
        else:
            print(f"Cannot visualize {p.ndim}D sensor placement")
    
    def _visualize_2d_placement(self, p: np.ndarray, figsize: Optional[Tuple[int, int]]) -> None:
        """Visualize 2D sensor placement.

        Parameters
        ----------
        p : np.ndarray
            The sensor placement matrix.
        figsize : Optional[Tuple[int, int]]
            The figure size (width, height) in inches.
        """
        if figsize is None:
            figsize = (max(8, p.shape[1] // 10), max(6, p.shape[0] // 10))
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("black")
        ax.imshow(np.zeros(p.shape), cmap="gray", origin="upper")
        
        # Plot sensors
        sensor_positions = np.argwhere(p == 1)
        if sensor_positions.size > 0:
            ax.scatter(sensor_positions[:, 1], sensor_positions[:, 0], 
                      s=50, c="red", marker="o", alpha=0.8, label="Sensors")
        
        ax.set_title(f"Sensor Placement (N={self.N}, actual={torch.sum(self.P).item()})", 
                    color="white", fontsize=14)
        ax.axis("off")
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def _visualize_3d_placement(self, p: np.ndarray, figsize: Optional[Tuple[int, int]]) -> None:
        """Visualize 3D+ sensor placement with slice analysis.

        Parameters
        ----------
        p : np.ndarray
            The sensor placement matrix.
        figsize : Optional[Tuple[int, int]]
            The figure size (width, height) in inches.
        """
        # Project to 2D for main visualization
        sensor_map = np.max(p, axis=tuple(range(2, p.ndim)))
        
        if figsize is None:
            figsize = (12, 5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Main sensor map
        ax1.set_facecolor("black")
        ax1.imshow(np.zeros(sensor_map.shape), cmap="gray", origin="upper")
        
        sensor_positions = np.argwhere(sensor_map >= 1)
        if sensor_positions.size > 0:
            # Color code by number of sensors at each (x,y) position
            colors = [np.sum(p[pos[0], pos[1], :]) for pos in sensor_positions]
            scatter = ax1.scatter(sensor_positions[:, 1], sensor_positions[:, 0], 
                                s=50, c=colors, cmap="Reds", marker="o", alpha=0.8)
            plt.colorbar(scatter, ax=ax1, label="Sensors per position")
        
        ax1.set_title(f"Sensor Placement Map (N={self.N})", color="white", fontsize=12)
        ax1.axis("off")
        
        # Slice distribution histogram
        slice_counts = [np.sum(p[..., z]) for z in range(p.shape[2])]
        ax2.bar(range(p.shape[2]), slice_counts, alpha=0.7, color="skyblue")
        ax2.set_title("Sensors per Slice", fontsize=12)
        ax2.set_xlabel("Slice Index")
        ax2.set_ylabel("Sensor Count")
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_per_slice = np.mean(slice_counts)
        std_per_slice = np.std(slice_counts)
        ax2.axhline(y=mean_per_slice, color='red', linestyle='--', 
                   label=f'Mean: {mean_per_slice:.1f}±{std_per_slice:.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


# Backward compatibility alias
TensorTubeQRDecomposition = TensorTubeQRDecomposition


class TensorHOSVD:
    """
    A class to perform Higher-Order Singular Value Decomposition (HOSVD) on a tensor.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Initializes the TensorHOSVD class.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to decompose.
        """
        self.tensor = tensor

    def decompose(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Performs HOSVD on the input tensor.

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            A tuple containing the core tensor and a list of factor matrices.
        """
        core, factors = tl.decomposition.tucker(
            self.tensor,
            rank=self.tensor.shape,
            init="svd",
            svd="numpy_svd",
            random_state=12345,
        )
        return core, factors