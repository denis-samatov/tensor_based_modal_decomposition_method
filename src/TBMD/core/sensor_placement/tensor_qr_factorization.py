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

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import torch

from TBMD.config import SensorPlacementConfig
from TBMD.core.utils.misc import get_torch_device, to_torch_tensor


class TensorValidator:
    """
    Comprehensive validation for tensor QR decomposition inputs.

    Ensures numerical stability and algorithm requirements are met.
    """

    @staticmethod
    def validate_tensor(tensor: torch.Tensor, min_dims: int = 3) -> None:
        """
        Validate input tensor for QR decomposition.

        Args:
            tensor: Input tensor to validate
            min_dims: Minimum required dimensions

        Raises:
            ValueError: If tensor doesn't meet requirements
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
                        _, s, _ = torch.linalg.svd(
                            flat_tensor[..., : min(flat_tensor.shape[-2:])].float(),
                            full_matrices=False,
                        )
                        if s.min() < SensorPlacementConfig().machine_epsilon_factor:
                            print("Warning: Tensor may be numerically rank-deficient")
                except RuntimeError:
                    pass  # SVD might fail for some tensor shapes

    @staticmethod
    def validate_sensor_count(N: int, k: int) -> None:
        """Validate sensor count parameter."""
        import numpy as np

        # Convert to Python int if it's a numpy integer type
        if hasattr(N, "item"):  # numpy scalar
            N = N.item()

        # Check if it's an integer type (Python int or numpy integer)
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError(f"N must be a positive integer, got {N} (type: {type(N)})")

        # Ensure N is Python int for consistency
        N = int(N)

        if N > k:
            print(
                f"Warning: N ({N}) exceeds tube dimension k ({k}). "
                f"Only {k} sensors can be effectively placed."
            )

    @staticmethod
    def validate_rejection_domain(
        rejection_domain: torch.Tensor, spatial_shape: Tuple[int, ...]
    ) -> None:
        """Validate rejection domain mask."""
        if rejection_domain.shape != spatial_shape:
            raise ValueError(
                f"Rejection domain shape {rejection_domain.shape} "
                f"must match spatial shape {spatial_shape}"
            )
        if rejection_domain.dtype != torch.bool:
            raise ValueError("Rejection domain must have boolean dtype")


class NumericallyStableOperations:
    """
    Numerically stable mathematical operations for tensor QR decomposition.

    Implements stable algorithms to avoid catastrophic cancellation and
    maintain numerical precision throughout the computation.
    """

    def __init__(self, config: SensorPlacementConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype

    def compute_householder_vector(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Householder reflection vector according to Algorithm 2.

        Implements the exact formula from the paper:
        σ ← ||t||₂
        u ← (t + sign(t₁)σe^d) / √(2σ(σ + |t₁|))

        Args:
            v: Input vector (tube) for Householder transformation

        Returns:
            Normalized Householder vector u
        """
        if v.numel() == 0:
            return torch.zeros_like(v)

        # Compute σ = ||v||₂
        sigma = torch.norm(v, p=2)

        if sigma < self.config.householder_threshold:
            return torch.zeros_like(v)

        # Compute sign(v[0]) with numerical stability
        if torch.abs(v[0]) < self.config.machine_epsilon_factor:
            sign_v0 = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            sign_v0 = torch.sign(v[0])

        # Compute u = v + sign(v[0]) * σ * e₁
        u = v.clone()
        u[0] = u[0] + sign_v0 * sigma

        # Compute denominator: √(2σ(σ + |v[0]|))
        denom = torch.sqrt(2 * sigma * (sigma + torch.abs(v[0])))

        if denom < self.config.machine_epsilon_factor:
            return torch.zeros_like(v)

        # Return normalized vector: u / √(2σ(σ + |v[0]|))
        return u / denom

    def check_orthogonality(self, Q: torch.Tensor) -> Tuple[bool, float]:
        """
        Check orthogonality of matrix Q with numerical tolerance.

        Args:
            Q: Matrix to check for orthogonality

        Returns:
            Tuple of (is_orthogonal, max_deviation)
        """
        if Q.numel() == 0:
            return True, 0.0

        QTQ = Q.T @ Q
        I = torch.eye(Q.shape[1], device=self.device, dtype=self.dtype)
        deviation = torch.max(torch.abs(QTQ - I)).item()

        is_orthogonal = deviation < self.config.orthogonality_tolerance
        return is_orthogonal, deviation

    def estimate_condition_number(self, tensor: torch.Tensor) -> float:
        """
        Estimate condition number for numerical stability monitoring.

        Args:
            tensor: Input tensor

        Returns:
            Estimated condition number
        """
        try:
            # Use SVD on a representative slice for efficiency
            if tensor.ndim >= 2:
                representative_slice = tensor.flatten(-2, -1)
                if representative_slice.shape[-1] > 0:
                    _, s, _ = torch.linalg.svd(
                        representative_slice[..., : min(representative_slice.shape[-2:])].float(),
                        full_matrices=False,
                    )
                    if s.numel() > 1 and s.min() > 0:
                        return (s.max() / s.min()).item()
        except RuntimeError:
            pass
        return 1.0  # Default safe value


class OptimizedPivotSelector:
    """
    Optimized pivot selection with vectorized operations and efficient penalties.

    Implements Algorithm 2's pivot selection with performance optimizations
    for large tensors and uniform distribution constraints.
    """

    def __init__(self, config: SensorPlacementConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype

    def select_pivot(
        self,
        R: torch.Tensor,
        d: int,
        available: torch.Tensor,
        distribution_state: Optional[Dict] = None,
    ) -> Tuple[int, ...]:
        """
        Select optimal pivot position with numerical stability and efficiency.

        Args:
            R: Current R matrix
            d: Current decomposition step
            available: Boolean mask of available positions
            distribution_state: State for uniform distribution (optional)

        Returns:
            Tuple of pivot indices
        """
        # Guard against empty availability
        if not available.any():
            raise RuntimeError(
                f"No available locations at step {d}. All cells are forbidden or already selected."
            )

        # Compute residual norms efficiently
        norms = self._compute_residual_norms(R, d)

        # Apply availability mask
        norms = torch.where(
            available, norms, torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)
        )

        # Apply distribution penalties if needed
        if distribution_state is not None:
            norms = self._apply_distribution_penalties(norms, distribution_state)

        # Select best pivot
        flat_idx = torch.argmax(norms).item()
        return np.unravel_index(flat_idx, norms.shape)

    def _compute_residual_norms(self, R: torch.Tensor, d: int) -> torch.Tensor:
        """
        Compute residual norms using L2 norm (Algorithm 2 standard).

        Note: R is mutated in-place during factorization, so we cannot
        cache by data_ptr(). Computing fresh each time ensures correct pivots.
        Uses L2 norm for alignment with DEIM literature and numerical stability.
        """
        residual = R[..., d:]
        # Use L2 norm (Frobenius norm along last dimension) per Algorithm 2
        norms = torch.linalg.norm(residual, dim=-1)
        return norms

    def _apply_distribution_penalties(self, norms: torch.Tensor, state: Dict) -> torch.Tensor:
        """Apply distribution penalties using vectorized operations."""
        penalties = torch.zeros_like(norms)

        # Cache maximum norm value for penalty scaling
        max_norm = torch.max(norms)

        # Vectorized slice penalty computation
        if "slice_counts" in state and len(norms.shape) >= 3:
            penalties += self._compute_slice_penalties(norms, state["slice_counts"], max_norm)

        # Vectorized distribution penalty computation
        if "sensor_placement" in state:
            penalties += self._compute_distribution_penalties(
                norms, state["sensor_placement"], max_norm
            )

        return norms - penalties

    def _compute_slice_penalties(
        self, norms: torch.Tensor, slice_counts: Dict[int, int], max_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute slice balance penalties efficiently."""
        if len(norms.shape) < 3:
            return torch.zeros_like(norms)

        penalties = torch.zeros_like(norms)
        total_sensors = sum(slice_counts.values())

        if total_sensors == 0:
            return penalties

        target_per_slice = total_sensors / norms.shape[2]

        for z in range(norms.shape[2]):
            current_count = slice_counts.get(z, 0)
            imbalance = max(0, current_count - target_per_slice)

            if imbalance > 0:
                penalty_value = imbalance * self.config.slice_penalty_weight * max_norm
                penalties[..., z] += penalty_value

        return penalties

    def _compute_distribution_penalties(
        self, norms: torch.Tensor, sensor_placement: torch.Tensor, max_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial distribution penalties efficiently."""
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

                penalty_values = (
                    density_per_slice.view(penalty_shape)
                    * self.config.distribution_penalty_weight
                    * max_norm
                )
                penalties += penalty_values

        return penalties


class UniformDistributionManager:
    """
    Manages uniform sensor distribution with efficient region grouping.

    Implements spatial and temporal distribution constraints for optimal
    sensor placement across the tensor domain.
    """

    def __init__(
        self, config: SensorPlacementConfig, spatial_shape: Tuple[int, ...], device: torch.device
    ):
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
        self.region_ids = (
            torch.full(spatial_shape[:2], -1, dtype=torch.long, device=device)
            if len(spatial_shape) >= 2
            else None
        )

        # Initialize slice tracking for 3D+ tensors
        if len(spatial_shape) >= 3:
            for z in range(spatial_shape[2]):
                self.slice_counts[z] = 0

    def identify_similar_regions(self, tensor: torch.Tensor) -> None:
        """
        Efficiently identify similar regions for distribution constraints.

        Args:
            tensor: Input tensor for region analysis
        """
        if len(self.spatial_shape) < 3:
            return

        # Use first temporal slice as pattern descriptor
        pattern_tensor = tensor[..., 0].detach().cpu().numpy()

        pattern_to_id: Dict[tuple, int] = {}

        # Vectorized pattern computation
        for x in range(self.spatial_shape[0]):
            for y in range(self.spatial_shape[1]):
                if len(self.spatial_shape) >= 3:
                    z_values = pattern_tensor[x, y, :]
                    pattern = tuple(
                        np.round(z_values, decimals=self.config.similarity_grouping_decimals)
                    )

                    self.similar_regions.setdefault(pattern, []).append((x, y))
                    self.region_lookup[(x, y)] = pattern
                    if pattern not in pattern_to_id:
                        pattern_to_id[pattern] = len(pattern_to_id)
                    if self.region_ids is not None:
                        self.region_ids[x, y] = pattern_to_id[pattern]

        if self.region_ids is not None:
            self.similar_regions = {
                pattern_to_id[pattern]: torch.tensor(coords, dtype=torch.long, device=self.device)
                for pattern, coords in self.similar_regions.items()
            }
            self.region_lookup = {
                coords: pattern_to_id[pattern] for coords, pattern in self.region_lookup.items()
            }

    def update_sensor_placement(self, pivot: Tuple[int, ...]) -> None:
        """Update distribution tracking after sensor placement."""
        # Update dimension counts
        for dim, idx in enumerate(pivot):
            self.dimension_counts[dim][idx] = self.dimension_counts[dim].get(idx, 0) + 1

        # Update slice counts for 3D+ tensors
        if len(self.spatial_shape) >= 3 and len(pivot) >= 3:
            z = pivot[2]
            self.slice_counts[z] = self.slice_counts.get(z, 0) + 1

    def mark_similar_regions_unavailable(
        self, pivot: Tuple[int, ...], available: torch.Tensor
    ) -> None:
        """Mark nearby similar regions unavailable with conservative blocking."""
        if len(self.spatial_shape) < 3 or len(pivot) < 2:
            return

        x, y = pivot[0], pivot[1]
        region_id = self.region_lookup.get((x, y))

        if region_id is not None and region_id in self.similar_regions:
            similar_positions = self.similar_regions[region_id]

            # Block only the nearest positions.
            blocked_count = 0
            max_blocks_per_region = max(1, len(similar_positions) // 4)  # Maximum 25%

            for row in similar_positions:
                px = int(row[0].item()) if hasattr(row[0], "item") else int(row[0])
                py = int(row[1].item()) if hasattr(row[1], "item") else int(row[1])
                if (px, py) != (x, y) and blocked_count < max_blocks_per_region:
                    # Block only the same z-level
                    if len(pivot) >= 3:
                        z = pivot[2]
                        available[px, py, z] = False
                        blocked_count += 1

    def get_distribution_state(self, sensor_placement: torch.Tensor) -> Dict:
        """Get current distribution state for penalty computation."""
        return {
            "slice_counts": self.slice_counts.copy(),
            "dimension_counts": {k: v.copy() for k, v in self.dimension_counts.items()},
            "sensor_placement": sensor_placement,
        }


class TensorTubeQRDecomposition:
    """
    Improved Tensor-based QR factorization with tube pivoting.

    Implements Algorithm 2 from the tensor-based modal decomposition paper
    with enhanced numerical stability, performance optimization, and
    comprehensive validation.

    Key improvements over original implementation:
    - Numerically stable Householder transformations
    - Vectorized penalty computations
    - Separated concerns following SOLID principles
    - Comprehensive input validation
    - Scientific constants instead of magic numbers
    - Detailed documentation with mathematical references

    Mathematical Background:
    The algorithm performs QR factorization on tensor tubes (fibers along the last dimension)
    using Householder reflections with pivot selection for optimal sensor placement.

    References:
    - Algorithm 2: Tensor-based tube fiber-pivot QR factorization
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.)
    """

    def __init__(
        self,
        tensor: Union[np.ndarray, torch.Tensor, tl.tensor],
        N: Optional[int] = None,
        rejection_domain: Optional[Union[np.ndarray, torch.Tensor]] = None,
        random_state: Optional[int] = None,
        check_orthogonality: Optional[bool] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        uniform_distribution: Optional[bool] = None,
        config: Optional[SensorPlacementConfig] = None,
    ) -> None:
        """
        Initialize tensor QR decomposition with enhanced validation and configuration.

        Args:
            tensor: Input tensor of shape (..., k) where k is the tube dimension
            N: Number of sensors (pivot tubes) to select. If None, uses config.n_sensors
            rejection_domain: Boolean mask of positions that cannot host a sensor
            random_state: Seed for reproducible results. If None, uses config.random_state
            check_orthogonality: Whether to verify Q remains orthonormal. If None, uses config
            device: PyTorch device ("cpu", "cuda", "mps"). If None, uses config.device
            dtype: PyTorch data type. If None, uses config.dtype
            uniform_distribution: Whether to enforce spatial distribution constraints. If None, uses config
            config: Configuration object with algorithm parameters

        Raises:
            ValueError: If inputs don't meet algorithm requirements

        Note:
            All parameters are optional and are read from config if not passed explicitly.
            Priority: explicit argument > config > default value.
        """
        # Configuration first (needed for defaults)
        self.config = config or SensorPlacementConfig()

        # Resolve all parameters: explicit argument > config > default
        effective_random_state = (
            random_state
            if random_state is not None
            else (self.config.random_state or self.config.seed)
        )
        self._setup_reproducibility(effective_random_state)

        # Resolve device and dtype: explicit > config > default
        effective_device = device if device is not None else (self.config.device or "cpu")
        effective_dtype = (
            dtype
            if dtype is not None
            else (torch.float64 if self.config.dtype == "float64" else torch.float32)
        )

        # Device and data type setup
        self.device = get_torch_device(effective_device)
        self.dtype = effective_dtype

        # Convert and validate input tensor
        self.tensor = to_torch_tensor(tensor, device=self.device, dtype=self.dtype)
        TensorValidator.validate_tensor(self.tensor)

        # Extract tensor properties
        self.spatial_shape: Tuple[int, ...] = self.tensor.shape[:-1]
        self.k: int = self.tensor.shape[-1]

        # Resolve N: explicit > config
        effective_N = N if N is not None else self.config.n_sensors

        # Validate and store sensor count
        TensorValidator.validate_sensor_count(effective_N, self.k)
        self.N = effective_N

        # Setup availability mask
        self.available = self._setup_availability_mask(rejection_domain)

        # Initialize specialized components
        self.numerical_ops = NumericallyStableOperations(self.config, self.device, self.dtype)
        self.pivot_selector = OptimizedPivotSelector(self.config, self.device, self.dtype)

        # Resolve uniform_distribution and check_orthogonality from config
        self.uniform_distribution = (
            uniform_distribution
            if uniform_distribution is not None
            else self.config.uniform_distribution
        )
        self.check_orthogonality = (
            check_orthogonality
            if check_orthogonality is not None
            else self.config.check_orthogonality
        )

        # Initialize distribution manager if needed
        if self.uniform_distribution:
            self.distribution_manager = UniformDistributionManager(
                self.config, self.spatial_shape, self.device
            )
            self.distribution_manager.identify_similar_regions(self.tensor)

        self._reset_results()

        # Monitor numerical health
        condition_number = self.numerical_ops.estimate_condition_number(self.tensor)
        if condition_number > self.config.condition_number_threshold:
            print(
                f"Warning: High condition number ({condition_number:.2e}) detected. "
                f"Results may be numerically unstable."
            )

    def _setup_reproducibility(self, random_state: Optional[int]) -> None:
        """Setup reproducible random number generation."""
        if random_state is not None:
            np.random.seed(random_state)
            tl.check_random_state(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
                torch.cuda.manual_seed_all(random_state)

    def _setup_availability_mask(
        self, rejection_domain: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Setup and validate availability mask for sensor placement.

        Note: rejection_domain marks cells that CANNOT host sensors (True = forbidden).
        We invert it to get availability mask (True = available).
        """
        if rejection_domain is None:
            return torch.ones(self.spatial_shape, dtype=torch.bool, device=self.device)

        rejection_tensor = to_torch_tensor(rejection_domain, dtype=torch.bool, device=self.device)
        TensorValidator.validate_rejection_domain(rejection_tensor, self.spatial_shape)

        # Invert: rejection marks forbidden, availability marks allowed
        available = ~rejection_tensor

        # Validate that at least some cells are available
        if not available.any():
            raise ValueError("rejection_domain forbids all cells - no valid sensor locations")

        return available

    def _reset_results(self) -> None:
        """Reset algorithm results for fresh computation."""
        self.P: Optional[torch.Tensor] = None  # Sensor placement indicator
        self.Q: Optional[torch.Tensor] = None  # Orthogonal matrix
        self.R: Optional[torch.Tensor] = None  # Upper triangular result
        self._orthogonality_history: List[float] = []

    def factorize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform tensor QR factorization with tube pivoting.

        Implements Algorithm 2 with numerical stability improvements and
        performance optimizations.

        Returns:
            Tuple of (P, Q, R) where:
            - P: Binary tensor indicating sensor positions
            - Q: k×k orthogonal matrix
            - R: Transformed tensor with upper-triangular structure

        Raises:
            RuntimeError: If algorithm fails due to numerical issues
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
                self.distribution_manager.slice_counts = (
                    {z: 0 for z in range(self.spatial_shape[2])}
                    if len(self.spatial_shape) >= 3
                    else {}
                )
                self.distribution_manager.dimension_counts = {
                    dim: {} for dim in range(len(self.spatial_shape))
                }

            # Main factorization loop
            successful_steps = 0
            for d in range(min(self.N, self.k)):
                try:
                    success = self._factorization_step(d, available)
                    if not success:
                        print(
                            f"Warning: Factorization stopped early at step {d} due to numerical issues"
                        )
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
            print("QR Factorization completed:")
            print(f"  Requested sensors: {self.N}")
            print(f"  Actual rank: {actual_rank}")
            print(f"  Success rate: {actual_rank / self.N * 100:.1f}%")
            print(f"  Early stops: {self.N - successful_steps}")

            if actual_rank < self.N * 0.5:  # Less than 50% success
                print("WARNING: Low rank achieved - consider relaxing thresholds!")

            return self.P, self.Q, self.R

        except Exception as e:
            self._reset_results()
            raise RuntimeError(f"Factorization failed: {e}")

    def _factorization_step(self, d: int, available: torch.Tensor) -> bool:
        """
        Perform a single step of the QR factorization algorithm.

        Args:
            d: Current step index
            available: Availability mask for sensor placement

        Returns:
            True if step completed successfully, False otherwise
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
        if tube_norm < self.config.machine_epsilon_factor:
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

        if u_norm < self.config.householder_threshold:
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
        """Apply Householder transformation to R matrix efficiently.

        Applies H = I - 2uu^T to each tube (fiber) R[x,y,d:k].
        According to Algorithm 2: R_{x,y,d:k} ← R_{x,y,d:k} - 2u^T R_{x,y,d:k}

        For each spatial position (x,y), the tube t = R[x,y,d:k] is transformed as:
        t_new = t - 2 * (u^T @ t) * u
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
        """Apply Householder transformation to Q matrix efficiently.

        Applies H = I - 2uu^T to Q from the right: Q := Q*H
        According to Algorithm 2: Q_{:,d:k} ← Q_{:,d:k} - 2Q_{:,d:k}uu^T
        """
        # Work with submatrix Q[:, d:] for efficiency
        Q_block = self.Q[:, d:]

        # Apply transformation: Q := Q - 2 * (Q @ u) @ u^T
        Qu = Q_block @ u  # (k,) vector - Q times u
        Q_block -= 2 * Qu.unsqueeze(1) * u.unsqueeze(0)  # Q := Q - 2*(Q@u)@u^T

    def check_factorization(self, tol: float = 1e-6) -> Tuple[bool, float, Dict[str, float]]:
        """
        Comprehensive validation of factorization quality.

        Args:
            tol: Tolerance for validation checks

        Returns:
            Tuple of (is_valid, relative_error, metrics_dict)
        """
        if any(x is None for x in (self.P, self.Q, self.R)):
            raise ValueError("Run factorize() first")

        metrics = {}

        # Check orthogonality
        is_orthogonal, ortho_deviation = self.numerical_ops.check_orthogonality(self.Q)
        metrics["orthogonality_deviation"] = ortho_deviation

        # Check reconstruction error
        RQ_T = torch.tensordot(self.R, self.Q.T, dims=([-1], [0]))
        reconstruction_error = torch.norm(self.tensor - RQ_T) / torch.norm(self.tensor)
        metrics["relative_reconstruction_error"] = float(reconstruction_error.item())

        # Check sensor count
        actual_sensors = torch.sum(self.P).item()
        metrics["sensor_count"] = actual_sensors
        metrics["sensor_efficiency"] = actual_sensors / self.N

        # Overall validation
        is_valid = is_orthogonal and reconstruction_error < tol and actual_sensors > 0

        return is_valid, float(reconstruction_error.item()), metrics

    def get_algorithm_info(self) -> Dict[str, any]:
        """Get comprehensive information about the algorithm state and performance."""
        info = {
            "tensor_shape": self.tensor.shape,
            "spatial_shape": self.spatial_shape,
            "tube_dimension": self.k,
            "requested_sensors": self.N,
            "uniform_distribution": self.uniform_distribution,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "config": self.config.__dict__,
        }

        if self.P is not None:
            info["actual_sensors"] = torch.sum(self.P).item()

        if self._orthogonality_history:
            info["orthogonality_history"] = self._orthogonality_history
            info["max_orthogonality_deviation"] = max(self._orthogonality_history)

        return info

    def visualize_sensor_placement(self, figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Visualize sensor placement with enhanced graphics and statistics.

        Args:
            figsize: Figure size (width, height) in inches
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
        """Visualize 2D sensor placement."""
        if figsize is None:
            figsize = (max(8, p.shape[1] // 10), max(6, p.shape[0] // 10))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("black")
        ax.imshow(np.zeros(p.shape), cmap="gray", origin="upper")

        # Plot sensors
        sensor_positions = np.argwhere(p == 1)
        if sensor_positions.size > 0:
            ax.scatter(
                sensor_positions[:, 1],
                sensor_positions[:, 0],
                s=50,
                c="red",
                marker="o",
                alpha=0.8,
                label="Sensors",
            )

        ax.set_title(
            f"Sensor Placement (N={self.N}, actual={torch.sum(self.P).item()})",
            color="white",
            fontsize=14,
        )
        ax.axis("off")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _visualize_3d_placement(self, p: np.ndarray, figsize: Optional[Tuple[int, int]]) -> None:
        """Visualize 3D+ sensor placement with slice analysis."""
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
            scatter = ax1.scatter(
                sensor_positions[:, 1],
                sensor_positions[:, 0],
                s=50,
                c=colors,
                cmap="Reds",
                marker="o",
                alpha=0.8,
            )
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
        ax2.axhline(
            y=mean_per_slice,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_per_slice:.1f}±{std_per_slice:.1f}",
        )
        ax2.legend()

        plt.tight_layout()
        plt.show()


# Backward compatibility alias
TensorBasedTubeFiberPivotQRFactorization = TensorTubeQRDecomposition
TensorQRConfig = SensorPlacementConfig
