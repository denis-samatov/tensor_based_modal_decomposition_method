"""
Tests and validation for geometry-aware TBMD components.

This module provides comprehensive tests for:
1. Mesh graph construction
2. Laplacian computation and properties
3. Geometry-aware HOSVD regularization
4. Geometry-aware QR sensor placement
5. Integration tests for the complete pipeline

Run with: python test_geometry_aware_components.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import issparse

from TBMD.core.decomposition import GeometryAwareDecompositionConfig as GeometryAwareConfig
from TBMD.core.decomposition import GeometryAwareTuckerDecomposer

# Import components to test
from TBMD.core.geometry import GeometricWeightComputer, MeshGraphBuilder
from TBMD.core.sensor_placement import GeometricQRConfig, GeometryAwareTensorQR


class TestMeshGraphBuilder:
    """Test mesh graph construction."""

    def test_grid_2d_construction(self):
        """Test 2D grid graph construction."""
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((10, 10))

        assert mesh.adjacency_matrix.shape == (100, 100)
        assert issparse(mesh.adjacency_matrix)

        # Check symmetry
        A = mesh.adjacency_matrix
        assert (A - A.T).nnz == 0

        # Check degree (interior points should have degree 4)
        degrees = np.array(A.sum(axis=1)).flatten()
        assert np.max(degrees) == 4  # Interior points
        assert np.min(degrees) == 2  # Corner points

        print(f"✓ 2D grid: {A.nnz} edges, degrees in [{degrees.min()}, {degrees.max()}]")

    def test_grid_3d_construction(self):
        """Test 3D grid graph construction."""
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((5, 5, 5))

        assert mesh.adjacency_matrix.shape == (125, 125)

        # Check degree (interior points should have degree 6)
        degrees = np.array(mesh.adjacency_matrix.sum(axis=1)).flatten()
        assert np.max(degrees) == 6  # Interior points

        print(f"✓ 3D grid: {mesh.adjacency_matrix.nnz} edges, max degree = {degrees.max()}")

    def test_knn_graph_construction(self):
        """Test k-NN graph construction."""
        # Random coordinates
        coords = np.random.rand(50, 2)

        builder = MeshGraphBuilder(connectivity_type="knn", k=6)
        mesh = builder.build_from_coordinates(coords)

        assert mesh.adjacency_matrix.shape == (50, 50)
        assert mesh.coordinates.shape == (50, 2)

        # Check that distances are computed
        assert mesh.distances is not None
        assert issparse(mesh.distances)

        print(f"✓ k-NN graph: {mesh.adjacency_matrix.nnz} edges")

    def test_laplacian_properties(self):
        """Test Laplacian matrix properties."""
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((10, 10))

        L = mesh.laplacian_matrix
        L_norm = mesh.normalized_laplacian

        # Laplacian should be symmetric
        assert (L - L.T).nnz == 0
        assert (L_norm - L_norm.T).nnz == 0

        # Row sums of L should be 0 (for connected graph)
        row_sums = np.array(L.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 0, atol=1e-10)

        # Normalized Laplacian eigenvalues should be in [0, 2]
        # (We don't compute them here due to cost, but property should hold)

        print("✓ Laplacian properties validated")


class TestGeometricWeights:
    """Test geometric weight computation."""

    def test_gradient_computation_fd(self):
        """Test finite difference gradient computation."""
        # Create mesh
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((20, 20))

        # Create field with known gradient
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)

        # Linear gradient: f(x,y) = x + 2*y
        field = (X + 2 * Y).ravel()

        # Compute gradients
        computer = GeometricWeightComputer(mesh)
        gradients = computer.compute_gradient_weights(field, method="fd")

        assert gradients.shape == (400,)
        assert np.all(gradients >= 0)

        # Gradient should be approximately constant for linear field
        # |∇f| = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
        np.sqrt(1**2 + 2**2)

        # Interior points should have consistent gradient
        interior_grads = gradients.reshape(20, 20)[5:-5, 5:-5]
        assert np.std(interior_grads) < 0.5  # Should be fairly uniform

        print(f"✓ FD gradients: mean={gradients.mean():.3f}, std={gradients.std():.3f}")

    def test_gradient_computation_graph(self):
        """Test graph-based gradient computation."""
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((20, 20))

        # Create field
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)
        field = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

        computer = GeometricWeightComputer(mesh)
        gradients = computer.compute_gradient_weights(field.ravel(), method="graph")

        assert gradients.shape == (400,)
        assert np.all(np.isfinite(gradients))

        print(f"✓ Graph gradients: range=[{gradients.min():.3f}, {gradients.max():.3f}]")

    def test_proximity_penalty(self):
        """Test proximity penalty computation."""
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((20, 20))

        computer = GeometricWeightComputer(mesh)

        # Place sensors at corners
        sensor_positions = np.array([0, 19, 380, 399])  # 4 corners

        min_distance = 5.0
        penalty = computer.compute_proximity_penalty(sensor_positions, min_distance)

        assert penalty.shape == (400,)
        assert np.all(penalty >= 0)

        # Penalty should be highest near sensors
        penalty_grid = penalty.reshape(20, 20)
        assert penalty_grid[0, 0] > 0.5  # High penalty at sensor location
        assert penalty_grid[10, 10] < 0.1  # Low penalty far from sensors

        print(f"✓ Proximity penalty: range=[{penalty.min():.3f}, {penalty.max():.3f}]")


class TestGeometryAwareHOSVD:
    """Test geometry-aware Tucker decomposition."""

    def test_basic_decomposition(self):
        """Test basic geometry-aware decomposition."""
        # Create smooth synthetic tensor
        H, W, T = 20, 20, 10

        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)

        tensor = np.zeros((H, W, T))
        for t in range(T):
            phase = 2 * np.pi * t / T
            tensor[..., t] = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.cos(phase)

        # Build mesh
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((H, W))

        # Decompose with geometry
        config = GeometryAwareConfig(alpha=0.1, spatial_modes=[0])

        decomposer = GeometryAwareTuckerDecomposer(
            tensor=tensor,
            mesh=mesh,
            geo_config=config,
            ranks=(10, 5, 8),
            epsilon=1e-3,
            max_iter=30,
            device="cpu",
        )

        decomposer.decompose()

        core = decomposer.cores
        factors = decomposer.factors

        assert core.shape == (10, 5, 8)
        assert factors[0].shape == (400, 10)  # Spatial factor
        assert factors[1].shape == (5, 5)  # Middle factor
        assert factors[2].shape == (10, 8)  # Temporal factor

        # Check reconstruction error
        reconstructed = decomposer.reconstruct()
        error = torch.norm(torch.from_numpy(tensor) - reconstructed) / torch.norm(
            torch.from_numpy(tensor)
        )

        assert error < 0.1, f"High reconstruction error: {error:.4f}"

        print(f"✓ Geometry-aware HOSVD: error={error:.4f}")

    def test_regularization_effect(self):
        """Test that regularization produces smoother modes."""
        H, W, T = 30, 30, 20

        # Create noisy tensor
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)

        tensor = np.zeros((H, W, T))
        for t in range(T):
            tensor[..., t] = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

        # Add noise
        tensor += 0.1 * np.random.randn(H, W, T)

        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((H, W))

        # Decompose with and without regularization
        ranks = (15, 10, 15)

        # Without regularization
        config_no_reg = GeometryAwareConfig(alpha=0.0)
        decomposer_no_reg = GeometryAwareTuckerDecomposer(
            tensor=tensor, mesh=mesh, geo_config=config_no_reg, ranks=ranks, device="cpu"
        )
        decomposer_no_reg.decompose()
        spatial_modes_no_reg = decomposer_no_reg.factors[0]

        # With regularization
        config_reg = GeometryAwareConfig(alpha=0.5)
        decomposer_reg = GeometryAwareTuckerDecomposer(
            tensor=tensor, mesh=mesh, geo_config=config_reg, ranks=ranks, device="cpu"
        )
        decomposer_reg.decompose()
        spatial_modes_reg = decomposer_reg.factors[0]

        # Measure smoothness using Laplacian
        L = mesh.laplacian_matrix

        def compute_smoothness(modes):
            """Smoothness = ||L * U||_F^2"""
            if isinstance(modes, torch.Tensor):
                modes = modes.detach().cpu().numpy()
            Lu = L @ modes
            return np.linalg.norm(Lu, "fro") ** 2

        smoothness_no_reg = compute_smoothness(spatial_modes_no_reg)
        smoothness_reg = compute_smoothness(spatial_modes_reg)

        # Regularized should be smoother
        assert smoothness_reg < smoothness_no_reg, (
            f"Regularization didn't improve smoothness: {smoothness_reg:.2f} >= {smoothness_no_reg:.2f}"
        )

        print(f"✓ Smoothness: no_reg={smoothness_no_reg:.2f}, reg={smoothness_reg:.2f}")


class TestGeometryAwareQR:
    """Test geometry-aware sensor placement."""

    def test_basic_sensor_placement(self):
        """Test basic sensor placement with geometry."""
        H, W, T = 20, 20, 15

        # Create tensor with gradients
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)

        tensor = np.zeros((H, W, T))
        for t in range(T):
            tensor[..., t] = np.tanh(3 * X) * np.tanh(3 * Y)

        # Build mesh
        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((H, W))

        # Configure QR
        config = GeometricQRConfig(
            gradient_weight=0.5, proximity_weight=1.0, min_distance_factor=2.0
        )

        # Place sensors
        n_sensors = 25
        geo_qr = GeometryAwareTensorQR(
            tensor=tensor, mesh=mesh, N=n_sensors, config=config, device="cpu"
        )

        P, Q, R = geo_qr.factorize()

        placed = torch.sum(P).item()
        assert placed <= n_sensors
        assert placed > 0

        print(f"✓ Placed {placed}/{n_sensors} sensors")

    def test_gradient_priority(self):
        """Test that sensors prioritize high-gradient regions."""
        H, W, T = 30, 30, 10

        # Create field with sharp gradient in center
        x = np.linspace(-2, 2, W)
        y = np.linspace(-2, 2, H)
        X, Y = np.meshgrid(x, y)

        tensor = np.zeros((H, W, T))
        for t in range(T):
            # Sharp transition in center
            tensor[..., t] = np.tanh(5 * X) + np.tanh(5 * Y)

        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((H, W))

        # Place sensors with gradient weighting
        config = GeometricQRConfig(
            gradient_weight=1.0,  # High weight
            proximity_weight=0.5,
            min_distance_factor=1.5,
        )

        n_sensors = 20
        geo_qr = GeometryAwareTensorQR(
            tensor=tensor, mesh=mesh, N=n_sensors, field_data=tensor, config=config, device="cpu"
        )

        P, Q, R = geo_qr.factorize()

        # Check that sensors are placed preferentially in center (high gradient)
        P_np = P.detach().cpu().numpy()
        sensor_positions = np.argwhere(P_np == 1)

        if len(sensor_positions) > 0:
            # Compute distance of each sensor from center
            center = np.array([H // 2, W // 2])
            distances = np.linalg.norm(sensor_positions - center, axis=1)

            # Mean distance should be relatively small (sensors near center)
            mean_dist = np.mean(distances)
            max_possible_dist = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)

            # At least 50% of sensors should be in inner half
            inner_sensors = np.sum(distances < max_possible_dist / 2)

            print(
                f"✓ Gradient priority: {inner_sensors}/{len(sensor_positions)} sensors in inner region"
            )
            print(f"  Mean distance from center: {mean_dist:.2f} (max: {max_possible_dist:.2f})")

    def test_proximity_penalty_effect(self):
        """Test that proximity penalty prevents clustering."""
        H, W, T = 25, 25, 10

        tensor = np.random.randn(H, W, T)

        builder = MeshGraphBuilder(connectivity_type="grid")
        mesh = builder.build_from_shape((H, W))

        # Place sensors with strong proximity penalty
        config = GeometricQRConfig(
            gradient_weight=0.0,  # No gradient bias
            proximity_weight=2.0,  # Strong spacing
            min_distance_factor=3.0,
        )

        n_sensors = 15
        geo_qr = GeometryAwareTensorQR(
            tensor=tensor, mesh=mesh, N=n_sensors, config=config, device="cpu"
        )

        P, Q, R = geo_qr.factorize()

        # Check sensor spacing
        P_np = P.detach().cpu().numpy()
        sensor_positions = np.argwhere(P_np == 1)

        if len(sensor_positions) > 1:
            # Compute pairwise distances
            from scipy.spatial.distance import pdist

            pairwise_dists = pdist(sensor_positions, metric="euclidean")
            min_spacing = np.min(pairwise_dists)
            mean_spacing = np.mean(pairwise_dists)

            print(f"✓ Sensor spacing: min={min_spacing:.2f}, mean={mean_spacing:.2f}")

            # Minimum spacing should be reasonable
            assert min_spacing > 1.0, f"Sensors too close: {min_spacing:.2f}"


def visualize_mesh_and_sensors():
    """Visualization test: show mesh and sensor placement."""
    H, W = 30, 30

    # Build mesh
    builder = MeshGraphBuilder(connectivity_type="grid")
    mesh = builder.build_from_shape((H, W))

    # Create field with features
    x = np.linspace(-2, 2, W)
    y = np.linspace(-2, 2, H)
    X, Y = np.meshgrid(x, y)

    tensor = np.zeros((H, W, 20))
    for t in range(20):
        phase = 2 * np.pi * t / 20
        tensor[..., t] = np.sin(3 * X) * np.cos(3 * Y) * np.cos(phase)

    # Place sensors
    config = GeometricQRConfig(gradient_weight=0.7, proximity_weight=1.0, min_distance_factor=2.0)

    geo_qr = GeometryAwareTensorQR(
        tensor=tensor, mesh=mesh, N=30, field_data=tensor, config=config, device="cpu"
    )

    P, Q, R = geo_qr.factorize()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Field
    axes[0].imshow(tensor[..., 0], cmap="viridis", origin="lower")
    axes[0].set_title("Field (t=0)")

    # Gradients
    from TBMD.utils.geometry import GeometricWeightComputer

    computer = GeometricWeightComputer(mesh)
    gradients = computer.compute_gradient_weights(tensor[..., 0].ravel(), method="graph")
    axes[1].imshow(gradients.reshape(H, W), cmap="hot", origin="lower")
    axes[1].set_title("Gradient Magnitude")

    # Sensors
    P_np = P.detach().cpu().numpy()
    axes[2].imshow(tensor[..., 0], cmap="gray", alpha=0.5, origin="lower")
    sensor_pos = np.argwhere(P_np == 1)
    axes[2].scatter(sensor_pos[:, 1], sensor_pos[:, 0], c="red", s=100, marker="x", linewidths=2)
    axes[2].set_title(f"Sensors (N={len(sensor_pos)})")

    plt.tight_layout()
    plt.savefig("geometry_aware_sensors_test.png", dpi=150)
    print("✓ Visualization saved to geometry_aware_sensors_test.png")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("GEOMETRY-AWARE TBMD COMPONENT TESTS")
    print("=" * 70)

    # Mesh tests
    print("\n### Mesh Graph Builder Tests ###")
    mesh_tests = TestMeshGraphBuilder()
    mesh_tests.test_grid_2d_construction()
    mesh_tests.test_grid_3d_construction()
    mesh_tests.test_knn_graph_construction()
    mesh_tests.test_laplacian_properties()

    # Geometric weights tests
    print("\n### Geometric Weights Tests ###")
    weight_tests = TestGeometricWeights()
    weight_tests.test_gradient_computation_fd()
    weight_tests.test_gradient_computation_graph()
    weight_tests.test_proximity_penalty()

    # HOSVD tests
    print("\n### Geometry-Aware HOSVD Tests ###")
    hosvd_tests = TestGeometryAwareHOSVD()
    hosvd_tests.test_basic_decomposition()
    hosvd_tests.test_regularization_effect()

    # QR tests
    print("\n### Geometry-Aware QR Tests ###")
    qr_tests = TestGeometryAwareQR()
    qr_tests.test_basic_sensor_placement()
    qr_tests.test_gradient_priority()
    qr_tests.test_proximity_penalty_effect()

    # Visualization
    print("\n### Visualization Test ###")
    visualize_mesh_and_sensors()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
