"""
Geometry-Aware Tensor QR Factorization for Sensor Placement.

This module extends the standard Tensor Tube QR factorization with geometric
information to improve sensor placement on unstructured meshes. Key enhancements:

1. **Geometric Weights**: Priority to cells with high spatial gradients (fronts,
   vortices, boundaries) using field gradient information.

2. **Proximity Penalties**: Penalize placing sensors too close together to ensure
   good spatial coverage and avoid redundancy.

3. **Mesh-Aware Distribution**: Use graph distance (not just Euclidean) to enforce
   uniform coverage across the mesh topology.

4. **Adaptive Threshold**: Automatically determine minimum sensor spacing based on
   mesh characteristic length and sensor count.

Mathematical Formulation
------------------------
Modified pivot selection criterion:

    pivot = argmax_i { ||R[i, d:]||₂ + β * w_grad[i] + η * w_amp[i] + ζ * w_energy[i]
                       - γ * w_prox[i] - δ * w_dist[i] }

where:
    - ||R[i, d:]||₂: residual norm (standard QR criterion)
    - w_grad[i]: geometric weight (gradient magnitude)
    - w_amp[i]: amplitude weight (RMS of field values)
    - w_energy[i]: local energy weight (own + neighbors)
    - w_prox[i]: proximity penalty to existing sensors
    - w_dist[i]: distribution penalty (slice/region balance)
    - β, η, ζ, γ, δ: tunable weights

References
----------
- Algorithm 2 (base QR): Tensor-based tube fiber-pivot QR factorization
- Chaturantabut & Sorensen (2010): Nonlinear model reduction via DEIM
- Manohar et al. (2018): Data-driven sparse sensor placement
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorly as tl
import torch

from TBMD.config import SensorPlacementConfig

from ..geometry.graph import (
    GeometricWeightComputer,
    MeshGeometry,
    estimate_characteristic_length,
)
from ..utils.misc import get_torch_device, to_torch_tensor
from .tensor_qr_factorization import (
    NumericallyStableOperations,
    TensorValidator,
)

logger = logging.getLogger(__name__)


@dataclass
class GeometricQRConfig(SensorPlacementConfig):
    """An extended configuration for geometry-aware QR factorization.

    This class inherits from `SensorPlacementConfig` and adds geometry-specific
    parameters.

    Attributes:
        gradient_weight (float): The weight for geometric gradient importance.
            Defaults to 0.5.
        proximity_weight (float): The weight for the proximity penalty. Higher
            values encourage more spacing between sensors. Defaults to 1.0.
        distribution_weight (float): The weight for distribution uniformity.
            Defaults to 0.5.
        min_distance_factor (float): The minimum sensor spacing as a multiple
            of the characteristic mesh length. Defaults to 2.0.
        gradient_method (str): The method for computing spatial gradients
            ('fd' or 'graph'). Defaults to 'graph'.
        adaptive_weights (bool): If `True`, automatically normalizes and
            scales weights based on the data. Defaults to `True`.
        use_graph_distance (bool): If `True`, uses graph geodesic distance;
            otherwise, uses Euclidean distance. Defaults to `False`.
    """

    # Geometry-specific weights
    gradient_weight: float = 0.5
    proximity_weight: float = 1.0
    distribution_weight: float = 0.5
    amplitude_weight: float = 1.0  # NEW: prioritize high-amplitude regions
    energy_weight: float = 0.5  # NEW: local energy importance

    # Sensor spacing
    min_distance_factor: float = 2.0

    # Methods
    gradient_method: str = "graph"
    adaptive_weights: bool = True
    use_graph_distance: bool = False


class GeometryAwarePivotSelector:
    """An enhanced pivot selector that incorporates geometric information.

    This selector combines residual norms from QR factorization with field
    gradient weights, proximity penalties, and mesh topology awareness to
    improve pivot selection.

    Args:
        config (GeometricQRConfig): The configuration with geometric parameters.
        mesh (MeshGeometry): The mesh geometry information.
        field_data (Optional[torch.Tensor]): Field data for computing
            gradients. If `None`, gradients are not used.
        device (torch.device): The PyTorch device.
        dtype (torch.dtype): The data type.
    """

    def __init__(
        self,
        config: GeometricQRConfig,
        mesh: MeshGeometry,
        field_data: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.mesh = mesh
        self.device = device
        self.dtype = dtype

        # Initialize geometric weight computer
        self.geo_computer = GeometricWeightComputer(mesh)

        # Compute gradient weights if field data provided
        if field_data is not None:
            self._compute_gradient_weights(field_data)
        else:
            self.gradient_weights = None

        # Estimate characteristic length
        self.h_char = estimate_characteristic_length(mesh)
        self.min_distance = config.min_distance_factor * self.h_char

        logger.info(
            f"GeometryAwarePivotSelector: h_char={self.h_char:.4f}, "
            f"min_distance={self.min_distance:.4f}"
        )

        # Track placed sensors
        self.placed_sensors: List[int] = []
        self.field_data = field_data

        # Cache for efficiency
        self._norm_cache = {}
        self.current_min_dists = None

    def _compute_gradient_weights(self, field_data: torch.Tensor) -> None:
        """Compute gradient-based geometric weights."""
        # Convert to numpy for geometry utilities
        if isinstance(field_data, torch.Tensor):
            field_np = field_data.detach().cpu().numpy()
        else:
            field_np = field_data

        # Handle spatial dimensions: (H, W, ..., T) -> (N_cells, T)
        N_cells = len(self.mesh.coordinates)

        if field_np.ndim > 2:
            field_np = field_np.reshape(N_cells, -1)
        elif field_np.ndim == 2 and field_np.shape[0] != N_cells:
            # Check if it is a single snapshot (spatial_1, spatial_2)
            if np.prod(field_np.shape) == N_cells:
                field_np = field_np.reshape(N_cells, 1)

        # Ensure 2D: (N_cells, N_time)
        if field_np.ndim == 1:
            field_np = field_np[:, np.newaxis]
        elif field_np.ndim > 2:
            # Flatten spatial dimensions: (d1, d2, ..., T) -> (N_cells, T)
            field_np = field_np.reshape(-1, field_np.shape[-1])

        # Compute gradient magnitude
        grad_mag = self.geo_computer.compute_gradient_weights(
            field_np, method=self.config.gradient_method
        )

        # Normalize to [0, 1]
        if grad_mag.max() > 0:
            grad_mag = grad_mag / grad_mag.max()

        # Convert to torch
        self.gradient_weights = torch.from_numpy(grad_mag).to(device=self.device, dtype=self.dtype)

        logger.info(
            f"Computed gradient weights: min={grad_mag.min():.4f}, "
            f"max={grad_mag.max():.4f}, mean={grad_mag.mean():.4f}"
        )

    def _compute_amplitude_weights(self, field_data: torch.Tensor) -> torch.Tensor:
        """
        Compute amplitude-based weights to prioritize high-energy regions.

        Parameters
        ----------
        field_data : torch.Tensor
            Field data (N_cells, N_time) or (N_cells,).

        Returns
        -------
        torch.Tensor
            Amplitude weights (N_cells,), normalized to [0, 1].
        """
        # Convert to torch if needed
        if not isinstance(field_data, torch.Tensor):
            field_data = torch.from_numpy(field_data).to(device=self.device, dtype=self.dtype)

        # Compute temporal mean and RMS for each cell
        if field_data.ndim == 1:
            amplitude = torch.abs(field_data)
        else:
            # RMS over time: sqrt(mean(field^2))
            # Use dim=-1 to handle both (N_cells, T) and (Nx, Ny, Nz, T)
            amplitude = torch.sqrt(torch.mean(field_data**2, dim=-1))

        # Normalize to [0, 1]
        if amplitude.max() > 0:
            amplitude = amplitude / amplitude.max()

        logger.info(
            f"Computed amplitude weights: min={amplitude.min():.4f}, "
            f"max={amplitude.max():.4f}, mean={amplitude.mean():.4f}"
        )

        return amplitude

    def _compute_energy_weights(self, field_data: torch.Tensor) -> torch.Tensor:
        """
        Compute local energy weights using mesh neighborhood.

        Energy at cell i = |f_i|^2 + sum_j(|f_j|^2) for neighbors j.

        Parameters
        ----------
        field_data : torch.Tensor
            Field data (N_cells, N_time) or (N_cells,).

        Returns
        -------
        torch.Tensor
            Energy weights (N_cells,), normalized to [0, 1].
        """
        # Convert to numpy for sparse operations
        if isinstance(field_data, torch.Tensor):
            field_np = field_data.detach().cpu().numpy()
        else:
            field_np = field_data

        # Compute local energy
        if field_np.ndim == 1:
            field_energy = field_np**2
        else:
            # Mean energy over time
            field_energy = np.mean(field_np**2, axis=-1)

        # Flatten if necessary for matrix multiplication
        if field_energy.ndim > 1:
            field_energy = field_energy.flatten()

        # Add neighbor energy (using adjacency matrix)
        adj = self.mesh.adjacency_matrix
        neighbor_energy = adj @ field_energy  # Sparse matrix-vector multiply

        # Total energy: own + neighbors
        total_energy = field_energy + neighbor_energy

        # Normalize to [0, 1]
        if total_energy.max() > 0:
            total_energy = total_energy / total_energy.max()

        # Convert to torch
        energy_weights = torch.from_numpy(total_energy).to(device=self.device, dtype=self.dtype)

        logger.info(
            f"Computed energy weights: min={total_energy.min():.4f}, "
            f"max={total_energy.max():.4f}, mean={total_energy.mean():.4f}"
        )

        return energy_weights

    def select_pivot(
        self,
        R: torch.Tensor,
        d: int,
        available: torch.Tensor,
        distribution_state: Optional[Dict] = None,
    ) -> Tuple[int, ...]:
        """Selects a pivot with geometric enhancements.

        Args:
            R (torch.Tensor): The current R matrix from the QR decomposition.
            d (int): The current decomposition step.
            available (torch.Tensor): A boolean mask of available positions.
            distribution_state (Optional[Dict]): The state for distribution
                tracking.

        Returns:
            Tuple[int, ...]: The multi-index of the selected pivot.
        """
        # 1. Compute residual norms (standard QR criterion)
        norms = self._compute_residual_norms(R, d)

        # 2. Apply availability mask
        norms = torch.where(
            available, norms, torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)
        )

        # 3. Add geometric gradient weights
        if self.gradient_weights is not None:
            # Reshape gradient weights to match spatial dimensions
            grad_reshaped = self.gradient_weights.view(norms.shape)
            max_norm = torch.max(norms[available])

            if max_norm > 0 and self.config.gradient_weight > 0:
                # Scale gradients to be comparable to norms
                scaled_gradients = self.config.gradient_weight * max_norm * grad_reshaped
                norms = norms + scaled_gradients

        # 3b. Add amplitude-based weights (NEW!)
        if self.field_data is not None and self.config.amplitude_weight > 0:
            amplitude_weights = self._compute_amplitude_weights(self.field_data)
            amp_reshaped = amplitude_weights.view(norms.shape)
            max_norm = torch.max(norms[available])

            if max_norm > 0:
                # Scale amplitude weights to be comparable to norms
                scaled_amplitude = self.config.amplitude_weight * max_norm * amp_reshaped
                norms = norms + scaled_amplitude
                logger.debug(f"Added amplitude weights (weight={self.config.amplitude_weight:.2f})")

        # 3c. Add energy-based weights (NEW!)
        if self.field_data is not None and self.config.energy_weight > 0:
            energy_weights = self._compute_energy_weights(self.field_data)
            energy_reshaped = energy_weights.view(norms.shape)
            max_norm = torch.max(norms[available])

            if max_norm > 0:
                # Scale energy weights to be comparable to norms
                scaled_energy = self.config.energy_weight * max_norm * energy_reshaped
                norms = norms + scaled_energy
                logger.debug(f"Added energy weights (weight={self.config.energy_weight:.2f})")

        # 4. Apply proximity penalties
        if len(self.placed_sensors) > 0 and self.config.proximity_weight > 0:
            prox_penalty = self._compute_proximity_penalty().view(norms.shape)
            max_norm = torch.max(norms[available])

            if max_norm > 0:
                scaled_penalty = self.config.proximity_weight * max_norm * prox_penalty
                norms = norms - scaled_penalty

        # 5. Apply distribution penalties (existing mechanism)
        if distribution_state is not None and self.config.distribution_weight > 0:
            # Use existing distribution penalty but with custom weight
            dist_penalty = self._compute_distribution_penalties(norms, distribution_state)
            norms = norms - self.config.distribution_weight * dist_penalty

        # 6. Select best pivot
        flat_idx = torch.argmax(norms).item()
        pivot = np.unravel_index(flat_idx, norms.shape)

        # 7. Update sensor tracking
        self.placed_sensors.append(flat_idx)

        # Update distances incrementally
        if self.current_min_dists is not None:
            _, self.current_min_dists = self.geo_computer.update_proximity_penalty(
                flat_idx, self.current_min_dists, self.min_distance
            )

        return pivot

    def _compute_residual_norms(self, R: torch.Tensor, d: int) -> torch.Tensor:
        """
        Compute residual norms (standard QR).

        Note: R is mutated in-place during Householder steps, so we cannot
        cache by data_ptr(). We compute norms fresh each time, or could cache
        by (iteration_number, d) if performance becomes critical.
        """
        residual = R[..., d:]
        norms = torch.sum(torch.abs(residual), dim=-1)
        return norms

    def _compute_proximity_penalty(self) -> torch.Tensor:
        """
        Compute penalty for placing sensors too close to existing ones.

        Returns penalty for each spatial location based on distance to
        nearest existing sensor.
        """
        if len(self.placed_sensors) == 0:
            # No penalty if no sensors placed yet
            spatial_size = int(np.prod(self.mesh.coordinates.shape[:-1]))
            return torch.zeros(spatial_size, device=self.device, dtype=self.dtype)

        if self.current_min_dists is not None:
            # Use cached distances (O(N) update is done in select_pivot)
            penalty = np.exp(-self.current_min_dists / (self.min_distance + 1e-10))
            return torch.from_numpy(penalty).to(device=self.device, dtype=self.dtype)

        # Fallback (should not be reached if reset() is called)
        sensor_positions = np.array(self.placed_sensors)
        penalty = self.geo_computer.compute_proximity_penalty(sensor_positions, self.min_distance)
        return torch.from_numpy(penalty).to(device=self.device, dtype=self.dtype)

    def _compute_distribution_penalties(self, norms: torch.Tensor, state: Dict) -> torch.Tensor:
        """Compute distribution balance penalties (existing logic)."""
        penalties = torch.zeros_like(norms)
        max_norm = torch.max(norms)

        # Slice balance penalty
        if "slice_counts" in state and len(norms.shape) >= 3:
            slice_counts = state["slice_counts"]
            total_sensors = sum(slice_counts.values())

            if total_sensors > 0:
                target_per_slice = total_sensors / norms.shape[2]

                for z in range(norms.shape[2]):
                    current_count = slice_counts.get(z, 0)
                    imbalance = max(0, current_count - target_per_slice)

                    if imbalance > 0:
                        penalty_value = imbalance * self.config.slice_penalty_weight * max_norm
                        penalties[..., z] += penalty_value

        return penalties

    def reset(self) -> None:
        """Reset state for new factorization."""
        self.placed_sensors.clear()
        self._norm_cache.clear()
        # Initialize with infinity
        N_cells = len(self.mesh.coordinates)
        self.current_min_dists = np.full(N_cells, np.inf)


class GeometryAwareTensorQR:
    """A geometry-aware Tensor QR factorization for optimal sensor placement.

    This class extends `TensorTubeQRDecomposition` with geometric information
    from unstructured meshes to improve the quality of sensor placement.

    Args:
        tensor (Union[np.ndarray, torch.Tensor, tl.tensor]): The input tensor.
        mesh (MeshGeometry): The mesh geometry with adjacency and Laplacian.
        N (int): The number of sensors to place.
        field_data (Optional[Union[np.ndarray, torch.Tensor]]): Field data for
            gradient computation. If `None`, the input tensor is used.
        rejection_domain (Optional[Union[np.ndarray, torch.Tensor]]): A boolean
            mask of forbidden positions.
        random_state (Optional[int]): The random seed for reproducibility.
        check_orthogonality (bool): If `True`, verifies the orthogonality of Q
            during factorization. Defaults to `False`.
        device (str): The PyTorch device to use. Defaults to "cpu".
        dtype (torch.dtype): The data type for tensors. Defaults to
            `torch.float32`.
        config (Optional[GeometricQRConfig]): The configuration with geometric
            parameters.
    """

    def __init__(
        self,
        tensor: Union[np.ndarray, torch.Tensor, tl.tensor],
        mesh: MeshGeometry,
        N: int,
        field_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        rejection_domain: Optional[Union[np.ndarray, torch.Tensor]] = None,
        random_state: Optional[int] = None,
        check_orthogonality: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        config: Optional[GeometricQRConfig] = None,
    ):
        """Initializes the GeometryAwareTensorQR."""
        self.config = config or GeometricQRConfig()
        self.mesh = mesh

        # Setup device
        self.device = get_torch_device(device)
        self.dtype = dtype

        # Convert tensor
        self.tensor = to_torch_tensor(tensor, device=self.device, dtype=self.dtype)
        # For geometry-aware QR, we support both 2D (spatial_cells, time) and 3D tensors
        min_dims = 2 if self.tensor.ndim == 2 else 3
        TensorValidator.validate_tensor(self.tensor, min_dims=min_dims)

        # Extract properties
        self.spatial_shape = self.tensor.shape[:-1]
        self.k = self.tensor.shape[-1]

        # Validate mesh compatibility
        if self.tensor.ndim == 2:
            # 2D tensor: first dimension is spatial (already flattened)
            expected_cells = self.tensor.shape[0]
        else:
            # 3D+ tensor: spatial dimensions are all except last
            expected_cells = int(np.prod(self.spatial_shape))

        if mesh.adjacency_matrix.shape[0] != expected_cells:
            raise ValueError(
                f"Mesh has {mesh.adjacency_matrix.shape[0]} cells but "
                f"tensor spatial size is {expected_cells}. Tensor shape: {self.tensor.shape}"
            )

        # Validate sensor count
        TensorValidator.validate_sensor_count(N, self.k)
        self.N = N

        # Setup availability mask
        if rejection_domain is None:
            if self.tensor.ndim == 2:
                # 2D tensor: availability mask is 1D
                self.available = torch.ones(
                    self.tensor.shape[0], dtype=torch.bool, device=self.device
                )
            else:
                self.available = torch.ones(
                    self.spatial_shape, dtype=torch.bool, device=self.device
                )
        else:
            rejection_tensor = to_torch_tensor(
                rejection_domain, dtype=torch.bool, device=self.device
            )
            TensorValidator.validate_rejection_domain(rejection_tensor, self.spatial_shape)
            # Invert: rejection_domain marks FORBIDDEN cells, available marks ALLOWED cells
            self.available = ~rejection_tensor

            # Check that at least some cells are available
            if not self.available.any():
                raise ValueError(
                    "rejection_domain forbids all cells - no valid sensor locations available"
                )

        # Prepare field data for gradients
        if field_data is None:
            field_data = self.tensor
        else:
            field_data = to_torch_tensor(field_data, device=self.device, dtype=self.dtype)

        # Initialize geometry-aware pivot selector
        self.pivot_selector = GeometryAwarePivotSelector(
            config=self.config,
            mesh=mesh,
            field_data=field_data,
            device=self.device,
            dtype=self.dtype,
        )

        # Initialize numerical operations (from base class)
        self.numerical_ops = NumericallyStableOperations(self.config, self.device, self.dtype)

        # Algorithm state
        self.check_orthogonality = check_orthogonality
        self._reset_results()

        # Setup reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

    def _reset_results(self) -> None:
        """Reset algorithm results."""
        self.P: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.R: Optional[torch.Tensor] = None
        self._orthogonality_history: List[float] = []

    def factorize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs geometry-aware QR factorization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - P (torch.Tensor): The binary sensor placement mask.
                - Q (torch.Tensor): The orthogonal matrix (k × k).
                - R (torch.Tensor): The transformed tensor.
        """
        try:
            # Initialize
            self.R = self.tensor.clone()
            self.Q = torch.eye(self.k, device=self.device, dtype=self.dtype)
            self.P = torch.zeros(self.spatial_shape, device=self.device, dtype=torch.int32)
            self._orthogonality_history.clear()

            # Reset pivot selector
            self.pivot_selector.reset()

            # Reset availability
            available = self.available.clone()

            # Main factorization loop
            successful_steps = 0
            for d in range(min(self.N, self.k)):
                try:
                    success = self._factorization_step(d, available)
                    if not success:
                        logger.warning(f"Factorization stopped early at step {d}")
                        break
                    successful_steps += 1
                except RuntimeError as e:
                    logger.warning(f"Numerical error at step {d}: {e}")
                    break

            if successful_steps == 0:
                raise RuntimeError("Factorization failed: no successful steps")

            # Final checks
            is_orthogonal, deviation = self.numerical_ops.check_orthogonality(self.Q)
            if not is_orthogonal:
                logger.warning(f"Final Q not orthogonal (deviation: {deviation:.2e})")

            actual_rank = torch.sum(self.P).item()
            logger.info(f"Geometry-aware QR completed: {actual_rank}/{self.N} sensors placed")

            return self.P, self.Q, self.R

        except Exception as e:
            self._reset_results()
            raise RuntimeError(f"Factorization failed: {e}")

    def _factorization_step(self, d: int, available: torch.Tensor) -> bool:
        """Perform one factorization step."""
        # Select pivot using geometry-aware selector
        pivot = self.pivot_selector.select_pivot(self.R, d, available, None)

        # Extract tube
        tube = self.R[pivot + (slice(d, None),)]

        # Check significance
        tube_norm = torch.norm(tube)
        if tube_norm < self.config.machine_epsilon_factor:
            return False

        # Update placement
        self.P[pivot] = 1
        available[pivot] = False

        # Compute Householder vector
        u = self.numerical_ops.compute_householder_vector(tube)
        u_norm = torch.norm(u)

        if u_norm < self.config.householder_threshold:
            return False

        # Apply transformations
        self._apply_householder_to_R(u, d)
        self._apply_householder_to_Q(u, d)

        # Check orthogonality
        if self.check_orthogonality:
            is_orthogonal, deviation = self.numerical_ops.check_orthogonality(self.Q)
            self._orthogonality_history.append(deviation)
            if not is_orthogonal:
                logger.warning(f"Q lost orthogonality at step {d} (dev: {deviation:.2e})")

        return True

    def _apply_householder_to_R(self, u: torch.Tensor, d: int) -> None:
        """Apply Householder transformation to R."""
        sub_R = self.R[..., d:]
        original_shape = sub_R.shape
        flat_R = sub_R.reshape(-1, self.k - d)

        uT_tubes = flat_R @ u
        flat_R -= 2 * uT_tubes.unsqueeze(1) * u.unsqueeze(0)

        self.R[..., d:] = flat_R.reshape(original_shape)

    def _apply_householder_to_Q(self, u: torch.Tensor, d: int) -> None:
        """Apply Householder transformation to Q."""
        Q_block = self.Q[:, d:]
        Qu = Q_block @ u
        Q_block -= 2 * Qu.unsqueeze(1) * u.unsqueeze(0)

    def get_sensor_coordinates(self) -> np.ndarray:
        """Returns the coordinates of the placed sensors.

        Returns:
            np.ndarray: The sensor coordinates, with shape (N_sensors,
            spatial_dim).
        """
        if self.P is None:
            raise ValueError("Call factorize() first")

        sensor_indices = torch.nonzero(self.P.flatten(), as_tuple=False).cpu().numpy().flatten()
        return self.mesh.coordinates[sensor_indices]

    def visualize_with_geometry(
        self, show_mesh: bool = True, figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Visualizes sensor placement with a mesh overlay.

        Args:
            show_mesh (bool, optional): Whether to show the mesh edges.
                Defaults to `True`.
            figsize (tuple, optional): The figure size. Defaults to (12, 6).
        """
        if self.P is None:
            raise ValueError("Call factorize() first")

        import matplotlib.pyplot as plt

        # Determine if 2D or 3D
        is_3d = len(self.spatial_shape) == 3

        if is_3d:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Sensor placement
        P_np = self.P.detach().cpu().numpy()
        sensor_pos = np.argwhere(P_np == 1)

        if is_3d:
            # 3D Scatter plot
            if len(sensor_pos) > 0:
                ax1.scatter(
                    sensor_pos[:, 0],
                    sensor_pos[:, 1],
                    sensor_pos[:, 2],
                    c="blue",
                    s=50,
                    marker="x",
                    label="Sensors",
                )

                # Optional: Plot mesh boundary or points for context
                # This can be heavy for large meshes, so maybe just bounding box
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_zlabel("Z")
        elif P_np.ndim == 2:
            # 2D Image plot
            ax1.imshow(P_np, cmap="Reds", alpha=0.6, origin="lower")
            if len(sensor_pos) > 0:
                ax1.scatter(
                    sensor_pos[:, 1],
                    sensor_pos[:, 0],
                    c="blue",
                    s=100,
                    marker="x",
                    linewidths=2,
                    label="Sensors",
                )
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")

        ax1.set_title(f"Geometry-Aware Sensor Placement (N={torch.sum(self.P).item()})")
        ax1.legend()

        # Plot 2: Gradient weights if available
        if self.pivot_selector.gradient_weights is not None:
            grad_weights = self.pivot_selector.gradient_weights.detach().cpu().numpy()

            if is_3d:
                # For 3D, maybe scatter plot of high gradient points?
                # Or just skip complex visualization
                grad_weights = grad_weights.reshape(self.spatial_shape)
                # Simple scatter of points with high weight
                threshold = np.percentile(grad_weights, 95)
                high_grad_indices = np.argwhere(grad_weights > threshold)

                if len(high_grad_indices) > 0:
                    p = ax2.scatter(
                        high_grad_indices[:, 0],
                        high_grad_indices[:, 1],
                        high_grad_indices[:, 2],
                        c=grad_weights[grad_weights > threshold],
                        cmap="viridis",
                        alpha=0.1,
                    )
                    plt.colorbar(p, ax=ax2, label="Gradient Magnitude (>95%)")

                # Overlay sensors
                if len(sensor_pos) > 0:
                    ax2.scatter(
                        sensor_pos[:, 0],
                        sensor_pos[:, 1],
                        sensor_pos[:, 2],
                        c="red",
                        s=50,
                        marker="x",
                        label="Sensors",
                    )

            elif len(self.spatial_shape) == 2:
                grad_weights = grad_weights.reshape(self.spatial_shape)
                im = ax2.imshow(grad_weights, cmap="viridis", origin="lower")
                plt.colorbar(im, ax=ax2, label="Gradient Magnitude")

                # Overlay sensors
                if len(sensor_pos) > 0:
                    ax2.scatter(
                        sensor_pos[:, 1],
                        sensor_pos[:, 0],
                        c="red",
                        s=100,
                        marker="x",
                        linewidths=2,
                        label="Sensors",
                    )

            ax2.set_title("Spatial Gradient Weights")
            if is_3d:
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                ax2.set_zlabel("Z")
        else:
            ax2.text(
                0.5, 0.5, "No gradient data", ha="center", va="center", transform=ax2.transAxes
            )
            ax2.set_title("Gradient Weights (N/A)")

        plt.tight_layout()
        plt.show()
