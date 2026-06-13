"""Configuration objects for tensor decomposition."""

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from ..base import BaseConfig


@dataclass
class DecompositionConfig(BaseConfig):
    """Configuration for HOSVD/Tucker decomposition."""

    # Decomposition parameters
    ranks: Optional[Union[int, List[int]]] = None  # [spatial_rank, temporal_rank] or single int
    method: Literal["hosvd", "tucker", "st_hosvd"] = "hosvd"

    # Truncation thresholds
    energy_threshold: float = 0.99  # Energy threshold for automatic rank selection
    singular_value_threshold: float = 1e-10  # Singular value threshold

    # Optimization
    max_iterations: int = 100  # For iterative methods
    convergence_tol: float = 1e-6

    # Data centering and scaling
    center_data: bool = False
    normalize: bool = False

    # Numerical constraints used by hosvd.py
    min_rank: int = 1  # Minimum allowed rank
    epsilon: float = 1e-2  # Tucker convergence epsilon

    # Parallelism
    max_workers: Optional[int] = None  # Number of parallel workers (used in hosvd.py)

    # Additional fields used by hosvd.py
    random_state: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self._validate()

    def _validate(self):
        """Validate parameter ranges."""
        if self.ranks is not None:
            if isinstance(self.ranks, list):
                if len(self.ranks) != 2:
                    # Allow more than 2 ranks for general Tucker, but warn or just pass if not strict
                    pass
                if any(r <= 0 for r in self.ranks):
                    raise ValueError("all ranks must be positive")
            elif isinstance(self.ranks, int):
                if self.ranks <= 0:
                    raise ValueError("rank must be positive")

        if not 0 < self.energy_threshold <= 1:
            raise ValueError("energy_threshold must be in the range (0, 1]")

        if self.singular_value_threshold < 0:
            raise ValueError("singular_value_threshold must be non-negative")


@dataclass
class GeometryAwareDecompositionConfig(DecompositionConfig):
    """Configuration for geometry-aware decomposition."""

    # Geometry parameters
    alpha: float = 0.1  # Geometry regularization weight
    alpha_adaptive: bool = False  # Adaptive alpha selection
    alpha_min: float = 0.01
    alpha_max: float = 0.5

    # Graph parameters
    graph_metric: Literal["euclidean", "geodesic"] = "euclidean"
    k_neighbors: int = 6  # Number of neighbors for graph construction

    # Laplacian
    laplacian_type: Literal["unnormalized", "symmetric", "random_walk"] = "symmetric"

    # Weights
    weight_function: Literal["inverse_distance", "gaussian", "uniform"] = "gaussian"
    gaussian_sigma: Optional[float] = None  # None means automatic selection

    def _validate(self):
        """Validate geometry-aware parameters."""
        super()._validate()

        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in the range [0, 1]")

        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")

        if self.alpha_adaptive:
            if not 0 <= self.alpha_min <= self.alpha_max <= 1:
                raise ValueError("alpha bounds must satisfy 0 <= alpha_min <= alpha_max <= 1")
