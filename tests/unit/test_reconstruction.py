"""Tests for reconstruction modules."""

import numpy as np
import pytest
import torch

from TBMD.config import CompressiveSensingConfig
from TBMD.core.reconstruction import TensorCompressiveSensing


class TestTensorCompressiveSensing:
    """Tests for TensorCompressiveSensing."""

    def test_initialization(self, sample_spatial_modes, sample_measurements, reconstruction_config):
        """Test initialization."""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.bool()  # P must be boolean mask

        # P should be spatial shape mask, not matrix
        # Wait, implementation says P: bool array_like, shape = A.shape[:-1]
        # A is ( spatial..., W ).
        # So P matches spatial shape.
        # But sample_spatial_modes is (2D) I x R. A has shape (I, R).
        # P should be (I,).
        P_mask = torch.zeros(I, dtype=torch.bool)
        P_mask[sensor_indices] = True

        # Y must also match spatial shape (selected entries)

        # Let's mock Y
        Y = torch.zeros(I)  # I spatial points
        # Set values at sensor locations
        # For test we can just put random values
        Y[sensor_indices] = sample_measurements[
            :, 0
        ]  # Take one time slice or simplified 1D measurement

        Y_single = torch.zeros(I)
        Y_single[sensor_indices] = torch.randn(n_sensors)

        cs = TensorCompressiveSensing(sample_spatial_modes, P_mask, Y_single)
        assert cs.cfg is not None

    def test_basic_reconstruction_admm(self, sample_spatial_modes):
        """Test basic ADMM reconstruction."""
        I = sample_spatial_modes.shape[0]
        W = sample_spatial_modes.shape[1]

        # Create synthetic ground truth signal x
        x_true = torch.randn(W)

        # Generate full field y = A @ x
        y_full = sample_spatial_modes @ x_true

        # Create mask
        n_sensors = 30
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        P_mask = torch.zeros(I, dtype=torch.bool)
        P_mask[sensor_indices] = True

        # Create measurements Y (full shape, masked locations valid)
        Y = y_full.clone()
        # (Optional: zero out others to verify they aren't used, but implementation uses P mask)

        config = CompressiveSensingConfig(max_iter=50)

        cs = TensorCompressiveSensing(sample_spatial_modes, P_mask, Y, core_cfg=config)
        x_rec, metrics = cs.solve()

        assert x_rec.shape[0] == W
        assert metrics is not None
        assert metrics.iterations <= 50

    def test_reconstruction_error(self, sample_spatial_modes):
        """Test reconstruction error calculation."""
        I = sample_spatial_modes.shape[0]
        W = sample_spatial_modes.shape[1]
        x_true = torch.randn(W)
        y_full = sample_spatial_modes @ x_true

        n_sensors = 30
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        P_mask = torch.zeros(I, dtype=torch.bool)
        P_mask[sensor_indices] = True
        Y = y_full

        cs = TensorCompressiveSensing(sample_spatial_modes, P_mask, Y)
        cs.solve()

        # Check error method
        err = cs.reconstruction_error(x_true)
        assert err >= 0.0
        assert err < 1e-4  # Should be small for noiseless synthetic case

    def test_invalid_dimensions(self):
        """Test invalid dimensions."""
        A = torch.randn(10, 5)
        P = torch.zeros(12, dtype=torch.bool)  # Wrong size
        Y = torch.zeros(10)

        with pytest.raises(ValueError, match="Shapes of P/Y must match"):
            TensorCompressiveSensing(A, P, Y)

    def test_metrics_content(self, sample_spatial_modes):
        """Test metrics content."""
        I = sample_spatial_modes.shape[0]
        n_sensors = 30
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        P_mask = torch.zeros(I, dtype=torch.bool)
        P_mask[sensor_indices] = True
        Y = torch.randn(I)

        cs = TensorCompressiveSensing(sample_spatial_modes, P_mask, Y)
        _, metrics = cs.solve()

        assert metrics.iterations > 0
        assert hasattr(metrics, "converged")
        assert hasattr(metrics, "primal_residual")
        assert hasattr(metrics, "dual_residual")
