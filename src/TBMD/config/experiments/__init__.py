"""
Configuration for TBMD Experiments
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal, Union
import numpy as np
from ..base import BaseConfig


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration class for experiment parameters."""
    
    # --- Core parameters (inherited from BaseConfig) ---
    # seed: int = 0
    # device: str = 'cpu' (BaseConfig defaults to auto-detec or user provided)
    # verbose: bool = True
    
    # Experiment specifics
    solver_method: str = "triangular"
    
    # Compressive sensing parameters
    max_iter: int = 1000
    epsilon: float = 1e-2
    lambd: float = 0.95
    delta_0: float = 0.1
    delta_max: float = 1.0
    
    # Noise parameters
    noise_level: float = 0.0
    num_noise_samples: int = 0
    noise_threshold: float = 1e-6  # Threshold for determining "zero" values when adding noise
    
    # Analysis parameters
    confidence_level: float = 0.95
    convergence_tol: float = 1e-7
    subject_axis: bool = False
    
    # Validation parameters
    # Note: numpy arrays are mutable and generally tricky as defaults, 
    # but for config dataclasses it's often accepted if initialized carefully.
    # BaseConfig typically handles simple types.
    valid_mask: Optional[Union[np.ndarray, List]] = None
    wells: Optional[Dict[str, List[Tuple[int, int]]]] = None
    
    def __post_init__(self):
        super().__post_init__()  # Initialize BaseConfig (sets device, seed, etc.)
        self._validate()
        
    def _validate(self):
        """Validate configuration parameters."""
        if self.confidence_level not in [0.90, 0.95, 0.99]:
            print(f"Warning: confidence_level {self.confidence_level} not in [0.90, 0.95, 0.99]. Using 0.95.")
            self.confidence_level = 0.95
