"""Configuration for the digital twin workflow."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ..base import BaseConfig


@dataclass
class DigitalTwinConfig(BaseConfig):
    """Configuration for the digital twin workflow."""

    # Architecture
    n_spatial_modes: int = 40
    n_temporal_modes: int = 20
    n_sensors: int = 30

    # Modal coefficient forecaster
    forecaster_type: Literal["linear", "mlp", "lstm", "persistence"] = "linear"

    # Proxy model for scenario analysis with well controls
    proxy_model_type: Optional[Literal["linear_dynamics", "neural", "physics_informed"]] = None
    proxy_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])

    # Forecaster parameters
    forecaster_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "seq_length": 5,  # For LSTM/Transformer
            "learning_rate": 0.001,
            "weight_decay": 1e-5,  # L2 regularization
        }
    )

    # Proxy model parameters
    proxy_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "regularization": 1e-4,  # For LinearDynamics
            "learning_rate": 1e-3,  # For Neural
            "batch_size": 32,  # For Neural
        }
    )

    # Training
    train_test_split: float = 0.8
    validation_split: float = 0.2
    batch_size: int = 32
    epochs: int = 300
    early_stopping_patience: int = 20

    # Data Loading
    shuffle_training_data: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    # Operating modes
    online_learning: bool = False
    uncertainty_estimation: bool = False
    anomaly_detection: bool = False

    # Optimization
    use_data_assimilation: bool = True
    assimilation_method: Literal["kalman", "ensemble", "variational"] = "ensemble"

    # Performance
    cache_decomposition: bool = True
    parallel_scenarios: bool = True
    max_workers: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self._validate()

    def _validate(self):
        """Validate parameter ranges."""
        if self.n_spatial_modes <= 0:
            raise ValueError("n_spatial_modes must be positive")

        if self.n_temporal_modes <= 0:
            raise ValueError("n_temporal_modes must be positive")

        if self.n_sensors <= 0:
            raise ValueError("n_sensors must be positive")

        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be in (0, 1)")

        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be in (0, 1)")

        # validation_split is applied to the training subset, so only its own range is checked here.

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
