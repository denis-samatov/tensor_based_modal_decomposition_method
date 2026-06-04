import os
import pytest
import random
import numpy as np
import torch

@pytest.fixture(autouse=True)
def set_global_determinism():
    """Enforce determinism for every test."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True) # Uncomment if strictly required, but might be brittle on some ops
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- Fixtures from original tests/audit ---

@pytest.fixture
def synthetic_tensor():
    """Generates a consistent random tensor for testing."""
    shape = (8, 6, 4)
    return torch.rand(shape, dtype=torch.float32)

@pytest.fixture
def synthetic_mesh_laplacian():
    """Generates a synthetic Laplacian for an 8-node line graph."""
    # Simple 1D Laplacian for size 8
    size = 8
    L = np.zeros((size, size))
    for i in range(size):
        L[i, i] = 2
        if i > 0: L[i, i-1] = -1
        if i < size - 1: L[i, i+1] = -1
    return torch.tensor(L, dtype=torch.float32)

# --- Fixtures from original algorithm/tests/unit ---

@pytest.fixture
def sample_tensor_small():
    """Small tensor for fast tests."""
    return torch.randn(10, 10, 5)


@pytest.fixture
def sample_tensor_medium():
    """Medium tensor for realistic tests."""
    return torch.randn(100, 50, 20)


@pytest.fixture
def sample_tensor_large():
    """Large tensor for load-oriented tests."""
    return torch.randn(500, 200, 50)


@pytest.fixture
def sample_mesh_2d():
    """Simple 2D mesh geometry."""
    # Create a simple rectangular grid
    nx, ny = 10, 10
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Simple quadrilateral connectivity
    connectivity = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            idx = i * nx + j
            connectivity.append([idx, idx + 1, idx + nx, idx + nx + 1])
    
    return {
        'coordinates': coords,
        'connectivity': np.array(connectivity)
    }


@pytest.fixture
def sample_spatial_modes():
    """Sample spatial modes."""
    return torch.randn(100, 20)  # 100 spatial points, 20 modes


@pytest.fixture
def sample_temporal_modes():
    """Sample temporal modes."""
    return torch.randn(50, 10)  # 50 temporal points, 10 modes


@pytest.fixture
def sample_measurements():
    """Sample sensor measurements."""
    return torch.randn(30, 50)  # 30 sensors, 50 time points


@pytest.fixture
def temp_data_path(tmp_path):
    """Temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def decomposition_config():
    """Base decomposition configuration."""
    from TBMD.config import DecompositionConfig
    return DecompositionConfig(
        ranks=[20, 10, 5],
        energy_threshold=0.95,
        verbose=False
    )


@pytest.fixture
def sensor_config():
    """Base sensor placement configuration."""
    from TBMD.config import SensorPlacementConfig
    return SensorPlacementConfig(
        n_sensors=30,
        verbose=False
    )


@pytest.fixture
def reconstruction_config():
    """Base reconstruction configuration."""
    from TBMD.config import ReconstructionConfig
    return ReconstructionConfig(
        solver='admm',
        max_iterations=50,
        convergence_eps=1e-2,
        verbose=False
    )
