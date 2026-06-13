"""Tests for configuration modules."""

import pytest
import torch

from TBMD.config import (
    BaseConfig,
    DecompositionConfig,
    DigitalTwinConfig,
    GeometryAwareDecompositionConfig,
    ReconstructionConfig,
    SensorPlacementConfig,
)


class TestBaseConfig:
    """Tests for BaseConfig."""

    def test_default_values(self):
        """Test default values."""
        config = BaseConfig()
        assert config.backend == "pytorch"
        assert config.dtype == "float32"
        assert config.seed == 0
        assert config.deterministic is True

    def test_auto_device_selection(self):
        """Test automatic device selection."""
        config = BaseConfig(device=None)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = BaseConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "backend" in config_dict
        assert "device" in config_dict


class TestDecompositionConfig:
    """Tests for DecompositionConfig."""

    def test_valid_ranks(self):
        """Test valid ranks."""
        config = DecompositionConfig(ranks=[50, 20])
        assert config.ranks == [50, 20]

    def test_invalid_ranks_negative(self):
        """Test negative ranks."""
        with pytest.raises(ValueError, match="all ranks must be positive"):
            DecompositionConfig(ranks=[-1, 20])

    def test_invalid_energy_threshold(self):
        """Test invalid energy threshold."""
        with pytest.raises(ValueError, match="energy_threshold"):
            DecompositionConfig(energy_threshold=1.5)


class TestGeometryAwareDecompositionConfig:
    """Tests for GeometryAwareDecompositionConfig."""

    def test_valid_alpha(self):
        """Test valid alpha."""
        config = GeometryAwareDecompositionConfig(alpha=0.1)
        assert config.alpha == 0.1

    def test_invalid_alpha(self):
        """Test invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be"):
            GeometryAwareDecompositionConfig(alpha=1.5)

    def test_adaptive_alpha(self):
        """Test adaptive alpha."""
        config = GeometryAwareDecompositionConfig(
            alpha_adaptive=True, alpha_min=0.01, alpha_max=0.5
        )
        assert config.alpha_adaptive is True


class TestSensorPlacementConfig:
    """Tests for SensorPlacementConfig."""

    def test_valid_n_sensors(self):
        """Test valid sensor count."""
        config = SensorPlacementConfig(n_sensors=100)
        assert config.n_sensors == 100

    def test_invalid_n_sensors(self):
        """Test invalid sensor count."""
        with pytest.raises(ValueError, match="must be positive"):
            SensorPlacementConfig(n_sensors=-10)


class TestReconstructionConfig:
    """Tests for ReconstructionConfig."""

    def test_valid_solver(self):
        """Test valid solver."""
        config = ReconstructionConfig(solver="admm")
        assert config.solver == "admm"

    def test_invalid_damping_factor(self):
        """Test invalid damping factor."""
        with pytest.raises(ValueError, match="damping_factor"):
            ReconstructionConfig(damping_factor=1.5)

    def test_convergence_parameters(self):
        """Test convergence parameters."""
        config = ReconstructionConfig(max_iterations=100, convergence_eps=1e-3)
        assert config.max_iterations == 100
        assert config.convergence_eps == 1e-3


class TestDigitalTwinConfig:
    """Tests for DigitalTwinConfig."""

    def test_valid_configuration(self):
        """Test valid configuration."""
        config = DigitalTwinConfig(n_spatial_modes=40, n_sensors=30, forecaster_type="lstm")
        assert config.n_spatial_modes == 40
        assert config.n_sensors == 30
        assert config.forecaster_type == "lstm"

    def test_train_test_split_validation(self):
        """Test train/test split validation."""
        with pytest.raises(ValueError, match="train_test_split"):
            DigitalTwinConfig(train_test_split=1.5)

    def test_forecaster_config_defaults(self):
        """Test default forecaster values."""
        config = DigitalTwinConfig()
        assert "hidden_size" in config.forecaster_config
        assert "learning_rate" in config.forecaster_config
