"""Configuration objects for forecasting models."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..base import BaseConfig


@dataclass
class ForecasterConfig(BaseConfig):
    """Base configuration for forecasting models."""

    # Architecture
    model_type: Literal["linear", "mlp", "lstm", "transformer"] = "lstm"
    in_dim: Optional[int] = None  # Input dimension set during initialization
    out_dim: Optional[int] = None  # Output dimension set during initialization
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    seq_length: int = 5  # For LSTM/Transformer

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"

    # Training
    num_epochs: int = 300
    batch_size: int = 32
    val_split: float = 0.2
    early_stopping_patience: int = 20

    # Data Loading
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    # Delta Forecasting (forecast incremental change instead of absolute values)
    delta_forecast: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._validate()

    def _validate(self):
        """Validate parameter ranges."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be in the range [0, 1]")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0 < self.val_split < 1:
            raise ValueError("val_split must be in the range (0, 1)")


@dataclass
class LinearForecasterConfig(ForecasterConfig):
    """Configuration for the linear forecaster."""

    model_type: Literal["linear"] = "linear"

    # Linear models do not use these parameters
    hidden_size: int = 0
    num_layers: int = 0
    dropout: float = 0.0
    seq_length: int = 1

    # Simplified training for the linear model
    num_epochs: int = 1
    early_stopping_patience: int = 0

    def _validate(self):
        """Skip neural-network-specific validation for the linear model."""
        pass


@dataclass
class MLPForecasterConfig(ForecasterConfig):
    """Configuration for the MLP forecaster."""

    model_type: Literal["mlp"] = "mlp"

    # MLP defaults
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    seq_length: int = 1

    # MLP usually needs more epochs to converge
    num_epochs: int = 500
    early_stopping_patience: int = 20


@dataclass
class LSTMForecasterConfig(ForecasterConfig):
    """Configuration for the LSTM forecaster."""

    model_type: Literal["lstm"] = "lstm"

    # LSTM parameters
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0  # Applied only when num_layers > 1
    seq_length: int = 5

    # LSTM training
    num_epochs: int = 300
    early_stopping_patience: int = 20

    # Scheduled Sampling
    use_scheduled_sampling: bool = False
    ss_unroll_steps: int = 5
    ss_decay_rate: float = 0.01
    ss_min_prob: float = 0.0


@dataclass
class LatentModalForecasterConfig(BaseConfig):
    """
    Configuration for Latent Modal Forecaster.

    This forecaster operates in the Tucker-decomposed latent modal space:
    1. Decomposes the input tensor via HOSVD/Tucker → G, A, B, C
    2. Trains a sub-forecaster on temporal coefficients C
    3. Predicts c_{t+1} in latent space
    4. Reconstructs full spatial state via G, A, B, ĉ_{t+1}
    """

    # Tucker decomposition parameters
    ranks: Optional[list] = None  # Tucker ranks [R1, R2, R3] or int for uniform
    epsilon: float = 1e-2  # Tucker convergence tolerance
    random_state: Optional[int] = 0  # For reproducible decomposition

    # Train/test split
    train_ratio: float = 0.8  # Temporal train/test split ratio

    # Sub-forecaster selection
    forecaster_type: Literal["linear", "mlp", "lstm"] = "mlp"

    # === Tier 1 improvements ===
    spatial_mean_centering: bool = True  # Subtract temporal mean before decomposition
    latent_normalization: bool = True  # Standardize latent variables before sub-forecaster
    delta_forecast: bool = True  # If True, forecast Δc = c_{t+1} - c_t instead of c_{t+1}
    projection_refinement_steps: int = (
        0  # Number of iterative projection refinement steps (0 = disabled)
    )
    projection_refinement_alpha: float = 1.0  # Step size for projection refinement (0, 1]

    # Sub-forecaster configs (only the selected type is used)
    mlp_hidden_size: int = 128
    mlp_num_layers: int = 2
    mlp_dropout: float = 0.3
    mlp_num_epochs: int = 500
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 1e-5
    mlp_batch_size: int = 32
    mlp_val_split: float = 0.2
    mlp_early_stopping_patience: int = 30

    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_seq_length: int = 5
    lstm_num_epochs: int = 300
    lstm_learning_rate: float = 1e-3
    lstm_batch_size: int = 32
    lstm_val_split: float = 0.2
    lstm_early_stopping_patience: int = 20

    # Scheduled Sampling
    lstm_use_scheduled_sampling: bool = False
    lstm_ss_unroll_steps: int = 5
    lstm_ss_decay_rate: float = 0.01
    lstm_ss_min_prob: float = 0.0

    def _validate(self):
        """Validate latent modal forecaster parameters."""
        if not 0.1 <= self.train_ratio <= 0.99:
            raise ValueError(f"train_ratio must be in [0.1, 0.99], got {self.train_ratio}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.forecaster_type not in ("linear", "mlp", "lstm"):
            raise ValueError(
                f"forecaster_type must be 'linear', 'mlp', or 'lstm', got {self.forecaster_type}"
            )
        if self.projection_refinement_steps < 0:
            raise ValueError(
                f"projection_refinement_steps must be >= 0, got {self.projection_refinement_steps}"
            )

    def __post_init__(self):
        super().__post_init__()
        self._validate()


@dataclass
class MultiResolutionTBMDConfig(BaseConfig):
    """
    Configuration for Multi-Resolution Cascaded Tucker Forecaster.

    Decomposes the tensor at multiple resolution levels:
    - Level 1: Captures dominant/smooth energy modes
    - Level 2+: Captures residual detail/turbulence modes

    Each level has its own Tucker ranks and sub-forecaster.
    Predictions are summed in the spatial domain.
    """

    # Per-level configuration
    level_ranks: List[list] = field(default_factory=lambda: [[64, 64, 5], [64, 64, 15]])
    level_forecaster_types: List[str] = field(default_factory=lambda: ["linear", "linear"])

    # Train/test split (shared across levels)
    train_ratio: float = 0.8

    # Tucker parameters
    epsilon: float = 1e-2
    random_state: Optional[int] = 0

    # Tier 1 improvements (applied per-level)
    spatial_mean_centering: bool = True
    latent_normalization: bool = True
    delta_forecast: bool = True
    projection_refinement_steps: int = 0
    projection_refinement_alpha: float = 1.0

    # Sub-forecaster defaults
    mlp_hidden_size: int = 128
    mlp_num_layers: int = 2
    mlp_dropout: float = 0.3
    mlp_num_epochs: int = 500
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 1e-5
    mlp_batch_size: int = 32
    mlp_val_split: float = 0.2
    mlp_early_stopping_patience: int = 30

    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_seq_length: int = 5
    lstm_num_epochs: int = 300
    lstm_learning_rate: float = 1e-3
    lstm_batch_size: int = 32
    lstm_val_split: float = 0.2
    lstm_early_stopping_patience: int = 20

    def _validate(self):
        if len(self.level_ranks) < 1:
            raise ValueError("At least one level is required")
        if len(self.level_ranks) != len(self.level_forecaster_types):
            raise ValueError("level_ranks and level_forecaster_types must have the same length")
        for ft in self.level_forecaster_types:
            if ft not in ("linear", "mlp", "lstm"):
                raise ValueError(f"Invalid forecaster_type: {ft}")

    def __post_init__(self):
        super().__post_init__()
        self._validate()


# Backward compatibility with DigitalTwinConfig
def create_forecaster_config_from_dict(
    config_dict: dict, model_type: str = "lstm"
) -> ForecasterConfig:
    """
    Create a ForecasterConfig from a dictionary.

    Args:
        config_dict: Parameter dictionary.
        model_type: Model type ('linear', 'mlp', 'lstm').

    Returns:
        Matching ForecasterConfig instance.
    """
    if model_type == "linear":
        return LinearForecasterConfig(**config_dict)
    elif model_type == "mlp":
        return MLPForecasterConfig(**config_dict)
    elif model_type == "lstm":
        return LSTMForecasterConfig(**config_dict)
    else:
        return ForecasterConfig(**config_dict)
