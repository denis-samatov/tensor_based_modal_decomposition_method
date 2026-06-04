"""Configuration for sensor placement.

This module contains:
- SensorPlacementConfig: QR sensor placement configuration
- GeometricSensorConfig: geometry-aware sensor placement configuration

References:
- Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.)
- Algorithm 2: Tensor-based tube fiber-pivot QR factorization
"""
from dataclasses import dataclass
from typing import Optional, Literal
from ..base import BaseConfig


@dataclass
class SensorPlacementConfig(BaseConfig):
    """
    Configuration for QR-based sensor placement.
    
    Includes Tensor QR parameters and numerical stability constants.
    
    Inherits from BaseConfig, which provides:
    - seed: Optional[int] = 0 for reproducibility
    - device: Optional[str] = None for 'cuda', 'cpu', or auto
    - dtype: Literal['float32', 'float64'] = 'float32'
    - verbose: bool = True
    
    Attributes
    ----------
    n_sensors : int, default=200
        Number of sensors to place.
    uniform_distribution : bool, default=False
        Enable spatial distribution constraints.
    check_orthogonality : bool, default=False
        Check Q orthogonality at each step.
    random_state : int, optional
        Alternative seed parameter for sklearn-style API compatibility.
    
    Numerical Stability Constants
    -----------------------------
    machine_epsilon_factor : float, default=1e-6
        Practical tolerance for float32.
    householder_threshold : float, default=1e-6
        Threshold for Householder vector computation.
    orthogonality_tolerance : float, default=1e-4
        Tolerance for Q orthogonality checks.
    condition_number_threshold : float, default=1e12
        Maximum accepted tensor condition number before warning.
    
    Distribution Penalty Weights
    ----------------------------
    slice_penalty_weight : float, default=0.8
        Weight for inter-slice imbalance penalty.
    distribution_penalty_weight : float, default=0.5
        Weight for spatial distribution penalty.
    similarity_grouping_decimals : int, default=3
        Rounding precision for grouping similar regions.
    
    Examples
    --------
    >>> from TBMD.config import SensorPlacementConfig
    >>> config = SensorPlacementConfig(n_sensors=100, seed=42)
    >>> config.n_sensors
    100
    
    References
    ----------
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.)
    - Algorithm 2: Tensor-based tube fiber-pivot QR factorization
    """
    
    # Core parameters
    n_sensors: int = 200
    uniform_distribution: bool = False
    check_orthogonality: bool = False
    random_state: Optional[int] = None
    
    # Numerical stability constants
    machine_epsilon_factor: float = 1e-6  # Realistic tolerance for float32
    householder_threshold: float = 1e-6   # Threshold for Householder vector computation
    orthogonality_tolerance: float = 1e-4  # Realistic tolerance for float32 with accumulation
    condition_number_threshold: float = 1e12  # Maximum acceptable condition number
    
    # Distribution penalty weights
    slice_penalty_weight: float = 0.8      # Weight for inter-slice balance penalty
    distribution_penalty_weight: float = 0.5  # Weight for spatial distribution penalty
    similarity_grouping_decimals: int = 3   # Precision for similarity grouping
    
    def __post_init__(self):
        super().__post_init__()
        self._validate()
    
    def _validate(self):
        """Validate parameter ranges."""
        if self.n_sensors <= 0:
            raise ValueError("n_sensors must be positive")
        
        if not (0 < self.slice_penalty_weight <= 1):
            raise ValueError("slice_penalty_weight must be in (0, 1]")
        
        if not (0 < self.distribution_penalty_weight <= 1):
            raise ValueError("distribution_penalty_weight must be in (0, 1]")
        
        if self.condition_number_threshold < 1:
            raise ValueError("condition_number_threshold must be >= 1")


@dataclass
class GeometricSensorConfig(SensorPlacementConfig):
    """
    Configuration for geometry-aware sensor placement.
    
    Extends SensorPlacementConfig with geometry parameters for placement on
    irregular or unstructured grids.
    
    Attributes
    ----------
    gradient_weight : float, default=0.5
        Weight for geometric gradients.
    proximity_weight : float, default=1.0
        Penalty for proximity to already selected sensors.
    amplitude_weight : float, default=1.0
        Weight for field amplitude.
    energy_weight : float, default=0.5
        Weight for local spatial energy.
    min_distance_factor : float, default=2.0
        Minimum sensor distance as a multiplier of characteristic mesh length.
    gradient_method : {'fd', 'graph'}, default='graph'
        Spatial gradient method.
    adaptive_weights : bool, default=True
        Automatically normalize and scale weights.
    use_graph_distance : bool, default=False
        Use graph geodesic distance instead of Euclidean distance.
    k_neighbors : int, default=6
        Number of neighbors used for graph construction.
    
    Examples
    --------
    >>> from TBMD.config import GeometricSensorConfig
    >>> config = GeometricSensorConfig(
    ...     n_sensors=50,
    ...     seed=42,
    ...     gradient_weight=0.8,
    ...     proximity_weight=1.5
    ... )
    """
    
    # Geometry weights
    gradient_weight: float = 0.5
    proximity_weight: float = 1.0
    amplitude_weight: float = 1.0
    energy_weight: float = 0.5
    
    # Distance parameters
    min_distance_factor: float = 2.0  # min_distance = factor * h_char
    
    # Computation methods
    gradient_method: Literal['fd', 'graph'] = 'graph'
    adaptive_weights: bool = True
    use_graph_distance: bool = False
    
    # Graph parameters
    k_neighbors: int = 6
    
    def _validate(self):
        """Validate geometry-aware parameters."""
        super()._validate()
        
        if self.gradient_weight < 0:
            raise ValueError("gradient_weight must be >= 0")
        
        if self.proximity_weight < 0:
            raise ValueError("proximity_weight must be >= 0")
        
        if self.amplitude_weight < 0:
            raise ValueError("amplitude_weight must be >= 0")
        
        if self.energy_weight < 0:
            raise ValueError("energy_weight must be >= 0")
        
        if self.min_distance_factor <= 0:
            raise ValueError("min_distance_factor must be > 0")
        
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
