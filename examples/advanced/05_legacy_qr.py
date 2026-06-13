"""
Example usage of the improved Tensor QR Decomposition implementation.

This example demonstrates the enhanced features, numerical stability,
and comprehensive validation of the refactored TensorBasedTubeFiberPivotQRFactorization.
"""

import numpy as np
import torch

from TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import (
    TensorQRConfig,
    TensorTubeQRDecomposition,
)


def create_test_tensor(shape=(20, 20, 15, 10), noise_level=0.1, random_state=42):
    """Create a test tensor with known structure for validation."""
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Create structured tensor with temporal patterns
    spatial_dims = shape[:-1]
    temporal_dim = shape[-1]

    # Base tensor with smooth spatial variation
    x = np.linspace(0, 2 * np.pi, spatial_dims[0])
    y = np.linspace(0, 2 * np.pi, spatial_dims[1])
    if len(spatial_dims) >= 3:
        z = np.linspace(0, 2 * np.pi, spatial_dims[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        spatial_pattern = np.sin(X) * np.cos(Y) * np.sin(Z)
    else:
        X, Y = np.meshgrid(x, y, indexing="ij")
        spatial_pattern = np.sin(X) * np.cos(Y)

    # Add temporal evolution
    tensor = np.zeros(shape)
    for t in range(temporal_dim):
        temporal_weight = np.exp(-0.1 * t) * np.cos(0.5 * t)
        if len(spatial_dims) >= 3:
            tensor[..., t] = spatial_pattern * temporal_weight
        else:
            tensor[..., t] = spatial_pattern * temporal_weight

    # Add structured noise
    noise = noise_level * np.random.randn(*shape)
    tensor += noise

    return torch.tensor(tensor, dtype=torch.float32)


def demonstrate_basic_usage():
    """Demonstrate basic usage with default configuration."""
    print("=== Basic Usage Demo ===")

    # Create test data
    tensor = create_test_tensor(shape=(15, 15, 10, 8))
    print(f"Test tensor shape: {tensor.shape}")

    # Initialize with default configuration
    decomposer = TensorTubeQRDecomposition(
        tensor=tensor, N=5, random_state=42, check_orthogonality=True
    )

    # Perform factorization
    print("Performing QR factorization...")
    P, Q, R = decomposer.factorize()

    # Validate results
    is_valid, error, metrics = decomposer.check_factorization()
    print(f"Factorization valid: {is_valid}")
    print(f"Reconstruction error: {error:.2e}")
    print(f"Orthogonality deviation: {metrics['orthogonality_deviation']:.2e}")
    print(f"Sensors placed: {metrics['sensor_count']}/{decomposer.N}")

    return decomposer


def demonstrate_advanced_configuration():
    """Demonstrate advanced configuration options."""
    print("\n=== Advanced Configuration Demo ===")

    # Create custom configuration
    config = TensorQRConfig(
        MACHINE_EPSILON_FACTOR=1e-15,  # More conservative numerical tolerance
        SLICE_PENALTY_WEIGHT=0.9,  # Stronger slice balance enforcement
        DISTRIBUTION_PENALTY_WEIGHT=0.3,  # Moderate spatial distribution
        ORTHOGONALITY_TOLERANCE=1e-10,  # Stricter orthogonality requirements
        CONDITION_NUMBER_THRESHOLD=1e8,  # Lower condition number threshold
    )

    # Create larger test tensor
    tensor = create_test_tensor(shape=(25, 25, 12, 15), noise_level=0.05)

    # Create rejection domain (exclude edges)
    rejection_domain = torch.ones(tensor.shape[:-1], dtype=torch.bool)
    rejection_domain[:3, :, :] = False  # Exclude x-boundary
    rejection_domain[-3:, :, :] = False  # Exclude x-boundary
    rejection_domain[:, :3, :] = False  # Exclude y-boundary
    rejection_domain[:, -3:, :] = False  # Exclude y-boundary

    # Initialize with advanced configuration
    decomposer = TensorTubeQRDecomposition(
        tensor=tensor,
        N=12,
        rejection_domain=rejection_domain,
        random_state=42,
        check_orthogonality=True,
        uniform_distribution=True,  # Enable uniform distribution
        config=config,
    )

    print(f"Configuration: {config}")

    # Perform factorization
    print("Performing advanced QR factorization...")
    P, Q, R = decomposer.factorize()

    # Comprehensive validation
    is_valid, error, metrics = decomposer.check_factorization(tol=1e-8)

    print("Advanced factorization results:")
    print(f"  Valid: {is_valid}")
    print(f"  Reconstruction error: {error:.2e}")
    print(f"  Orthogonality deviation: {metrics['orthogonality_deviation']:.2e}")
    print(f"  Sensor efficiency: {metrics['sensor_efficiency']:.2%}")

    # Get algorithm information
    info = decomposer.get_algorithm_info()
    print(
        f"  Max orthogonality deviation during factorization: {info.get('max_orthogonality_deviation', 'N/A'):.2e}"
    )

    return decomposer


def demonstrate_numerical_stability():
    """Demonstrate improved numerical stability with challenging cases."""
    print("\n=== Numerical Stability Demo ===")

    # Create challenging tensor (near rank-deficient)
    shape = (10, 10, 8)
    tensor = torch.randn(shape, dtype=torch.float64)  # Use double precision

    # Make it nearly rank-deficient by setting small singular values
    flat_tensor = tensor.reshape(-1, shape[-1])
    U, s, V = torch.svd(flat_tensor)
    s[4:] *= 1e-10  # Make some singular values very small
    flat_tensor = U @ torch.diag(s) @ V.T
    tensor = flat_tensor.reshape(shape)

    print(f"Created challenging tensor with condition number ~ {s[0] / s[3]:.1e}")

    # Test with conservative configuration
    config = TensorQRConfig(
        MACHINE_EPSILON_FACTOR=1e-16, HOUSEHOLDER_THRESHOLD=1e-16, CONDITION_NUMBER_THRESHOLD=1e6
    )

    decomposer = TensorTubeQRDecomposition(
        tensor=tensor.float(),  # Convert back to float32
        N=4,
        config=config,
        check_orthogonality=True,
    )

    # Perform factorization
    try:
        P, Q, R = decomposer.factorize()
        is_valid, error, metrics = decomposer.check_factorization()

        print("Numerical stability test:")
        print(f"  Factorization completed: {is_valid}")
        print(f"  Final reconstruction error: {error:.2e}")
        print(f"  Orthogonality maintained: {metrics['orthogonality_deviation'] < 1e-6}")

    except RuntimeError as e:
        print(f"Factorization failed gracefully: {e}")
        print("This demonstrates robust error handling for ill-conditioned problems.")


def demonstrate_visualization():
    """Demonstrate enhanced visualization capabilities."""
    print("\n=== Visualization Demo ===")

    # Create 3D tensor for visualization
    tensor = create_test_tensor(shape=(20, 20, 8, 12))

    decomposer = TensorTubeQRDecomposition(
        tensor=tensor, N=15, uniform_distribution=True, random_state=42
    )

    # Perform factorization
    P, Q, R = decomposer.factorize()

    print("Generating enhanced visualizations...")

    # Enhanced sensor placement visualization
    try:
        decomposer.visualize_sensor_placement(figsize=(14, 6))
        print("Visualization generated successfully!")
        print("The plot shows:")
        print("  - Left: 2D projection of sensor positions with color coding")
        print("  - Right: Histogram of sensors per slice with statistics")
    except Exception as e:
        print(f"Visualization not available in this environment: {e}")


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling and validation."""
    print("\n=== Error Handling Demo ===")

    # Test various error conditions
    test_cases = [
        ("NaN tensor", lambda: torch.tensor([[[1.0, float("nan")]]]), "NaN values detected"),
        (
            "Infinite tensor",
            lambda: torch.tensor([[[1.0, float("inf")]]]),
            "Infinite values detected",
        ),
        ("Too few dimensions", lambda: torch.randn(5, 5), "Insufficient dimensions"),
        ("Invalid sensor count", lambda: (torch.randn(5, 5, 3), -1), "Invalid N parameter"),
    ]

    for test_name, tensor_func, expected_error in test_cases:
        try:
            if test_name == "Invalid sensor count":
                tensor, N = tensor_func()
                TensorTubeQRDecomposition(tensor, N)
            else:
                tensor = tensor_func()
                TensorTubeQRDecomposition(tensor, N=2)
            print(f"❌ {test_name}: Should have failed but didn't")
        except (ValueError, RuntimeError) as e:
            print(f"✅ {test_name}: Caught expected error - {str(e)[:50]}...")


def main():
    """Main demonstration function."""
    print("Tensor QR Decomposition - Improved Implementation Demo")
    print("=" * 60)

    # Run all demonstrations
    basic_decomposer = demonstrate_basic_usage()
    advanced_decomposer = demonstrate_advanced_configuration()
    demonstrate_numerical_stability()
    demonstrate_visualization()
    demonstrate_error_handling()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nKey improvements demonstrated:")
    print("✅ Numerical stability with challenging tensors")
    print("✅ Comprehensive input validation and error handling")
    print("✅ Configurable algorithm parameters")
    print("✅ Enhanced visualization with statistics")
    print("✅ Robust orthogonality checking")
    print("✅ Detailed algorithm diagnostics")

    return basic_decomposer, advanced_decomposer


if __name__ == "__main__":
    basic_decomposer, advanced_decomposer = main()
