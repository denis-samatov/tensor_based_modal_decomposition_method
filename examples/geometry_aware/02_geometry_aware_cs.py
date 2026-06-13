"""
Geometry-Aware Compressive Sensing Example
===========================================

This example demonstrates how to use GeometryAwareTensorCS for spatially smooth
reconstruction from incomplete measurements on an unstructured mesh.

Comparison:
- Standard CS: min ||Ax - y||² + ε||d||₁
- Geometry-aware CS: min ||Ax - y||² + ε||d||₁ + α||L·x||²

The geometry-aware version produces smoother, more physically realistic reconstructions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.core.geometry import MeshGraphBuilder
from TBMD.core.reconstruction import (
    CompressiveSensingConfig,
    GeometryAwareCSConfig,
    GeometryAwareTensorCS,
    TensorCompressiveSensing,
)


def generate_test_problem(
    n_spatial: int = 100, n_modes: int = 20, sensor_fraction: float = 0.3, noise_level: float = 0.01
):
    """
    Generate a synthetic compressive sensing problem with smooth spatial structure.

    Parameters
    ----------
    n_spatial : int
        Number of spatial cells (will create n_spatial x n_spatial grid).
    n_modes : int
        Number of basis modes.
    sensor_fraction : float
        Fraction of cells with sensors.
    noise_level : float
        Gaussian noise standard deviation.

    Returns
    -------
    A : np.ndarray (n_spatial², n_modes)
        Forward model (mode shapes).
    P : np.ndarray (n_spatial², ) bool
        Sensor mask.
    Y : np.ndarray (n_spatial², )
        Measurements.
    x_true : np.ndarray (n_modes, )
        True coefficients.
    spatial_shape : tuple
        Grid shape.
    """
    spatial_shape = (n_spatial, n_spatial)
    n_cells = n_spatial**2

    # Create smooth spatial modes (sum of sines)
    xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi, n_spatial), np.linspace(0, 2 * np.pi, n_spatial))

    A = np.zeros((n_cells, n_modes))
    for k in range(n_modes):
        # Create smooth mode shapes
        freq_x = 1 + k // 5
        freq_y = 1 + k % 5
        mode = np.sin(freq_x * xx) * np.cos(freq_y * yy)
        A[:, k] = mode.ravel()

    # Normalize modes
    A = A / np.linalg.norm(A, axis=0, keepdims=True)

    # Create sparse true coefficients
    x_true = np.zeros(n_modes)
    n_active = max(3, n_modes // 5)
    active_idx = np.random.choice(n_modes, n_active, replace=False)
    x_true[active_idx] = np.random.randn(n_active) * 5.0

    # Generate full field
    field_true = A @ x_true

    # Random sensor placement
    n_sensors = int(sensor_fraction * n_cells)
    sensor_idx = np.random.choice(n_cells, n_sensors, replace=False)
    P = np.zeros(n_cells, dtype=bool)
    P[sensor_idx] = True

    # Measurements with noise
    Y = field_true.copy()
    Y += np.random.randn(n_cells) * noise_level * np.std(field_true)

    return A, P.reshape(spatial_shape), Y.reshape(spatial_shape), x_true, spatial_shape


def compare_reconstructions(A, P, Y, x_true, spatial_shape, mesh, alpha_values=[0.0, 0.01, 0.1]):
    """
    Compare standard CS vs geometry-aware CS with different α values.

    Parameters
    ----------
    A, P, Y : np.ndarray
        Problem matrices.
    x_true : np.ndarray
        True coefficients.
    spatial_shape : tuple
        Grid shape.
    mesh : MeshGeometry
        Mesh object with Laplacian.
    alpha_values : list of float
        Regularization strengths to compare.

    Returns
    -------
    dict
        Results for each method.
    """
    results = {}

    # Standard CS (α = 0)
    print("\n" + "=" * 60)
    print("Standard Compressive Sensing (no geometry)")
    print("=" * 60)

    config_std = CompressiveSensingConfig(max_iter=500, epsilon_l1=1e-2, tol=1e-4, device="cpu")

    solver_std = TensorCompressiveSensing(A, P.flatten(), Y.flatten(), core_cfg=config_std)
    x_std, metrics_std = solver_std.solve()

    results["standard"] = {"x": x_std.numpy(), "metrics": metrics_std, "alpha": 0.0}

    print(f"  Iterations: {metrics_std.iterations}")
    print(f"  Converged: {metrics_std.converged}")
    print(f"  Time: {metrics_std.time_sec:.2f}s")
    print(f"  Final objective: {metrics_std.objective:.6e}")
    print(f"  ||x - x_true||: {np.linalg.norm(x_std.numpy() - x_true):.6f}")

    # Geometry-aware CS with different α
    for alpha in alpha_values:
        if alpha == 0.0:
            continue

        print("\n" + "=" * 60)
        print(f"Geometry-Aware CS (α = {alpha})")
        print("=" * 60)

        config_geo = GeometryAwareCSConfig(
            max_iter=500,
            epsilon_l1=1e-2,
            tol=1e-4,
            alpha=alpha,
            auto_alpha=False,  # Use fixed alpha for comparison
            device="cpu",
        )

        solver_geo = GeometryAwareTensorCS(A, P.flatten(), Y.flatten(), mesh, core_cfg=config_geo)
        x_geo, metrics_geo = solver_geo.solve()

        results[f"geometry_alpha_{alpha}"] = {
            "x": x_geo.numpy(),
            "metrics": metrics_geo,
            "alpha": alpha,
        }

        print(f"  Iterations: {metrics_geo.iterations}")
        print(f"  Converged: {metrics_geo.converged}")
        print(f"  Time: {metrics_geo.time_sec:.2f}s")
        print(f"  Final objective: {metrics_geo.objective:.6e}")
        print(f"  ||x - x_true||: {np.linalg.norm(x_geo.numpy() - x_true):.6f}")

    return results


def visualize_results(A, x_true, results, spatial_shape, save_path=None):
    """
    Visualize reconstruction comparison.

    Parameters
    ----------
    A : np.ndarray
        Forward model.
    x_true : np.ndarray
        True coefficients.
    results : dict
        Results from compare_reconstructions.
    spatial_shape : tuple
        Grid shape.
    save_path : str, optional
        Path to save figure.
    """
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))

    # Ground truth
    field_true = (A @ x_true).reshape(spatial_shape)

    im0 = axes[0, 0].imshow(field_true, cmap="RdBu_r", aspect="auto")
    axes[0, 0].set_title("Ground Truth\nField", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    axes[1, 0].stem(x_true, basefmt=" ", linefmt="k-", markerfmt="ko")
    axes[1, 0].set_title("True Coefficients", fontsize=10)
    axes[1, 0].set_xlabel("Mode index")
    axes[1, 0].grid(True, alpha=0.3)

    # Reconstructions
    for i, (method_name, result) in enumerate(results.items(), 1):
        x_rec = result["x"]
        field_rec = (A @ x_rec).reshape(spatial_shape)
        alpha = result["alpha"]

        # Field reconstruction
        im = axes[0, i].imshow(field_rec, cmap="RdBu_r", aspect="auto")
        title = f"α = {alpha}"
        if alpha == 0.0:
            title = "Standard CS\n" + title
        else:
            title = "Geometry-aware\n" + title
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis("off")
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

        # Coefficients
        axes[1, i].stem(x_rec, basefmt=" ", linefmt="b-", markerfmt="bo", label="Recovered")
        axes[1, i].stem(x_true, basefmt=" ", linefmt="k:", markerfmt="k^", label="True", alpha=0.5)
        axes[1, i].set_title(
            f"Recovered (error: {np.linalg.norm(x_rec - x_true):.3f})", fontsize=10
        )
        axes[1, i].set_xlabel("Mode index")
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def plot_convergence(results, save_path=None):
    """
    Plot convergence history comparison.

    Parameters
    ----------
    results : dict
        Results from compare_reconstructions.
    save_path : str, optional
        Path to save figure.
    """
    plt.figure(figsize=(10, 6))

    for method_name, result in results.items():
        history = result["metrics"].history
        alpha = result["alpha"]

        if alpha == 0.0:
            label = "Standard CS"
            linestyle = "-"
            linewidth = 2
        else:
            label = f"Geometry-aware (α={alpha})"
            linestyle = "--"
            linewidth = 1.5

        plt.semilogy(history, label=label, linestyle=linestyle, linewidth=linewidth)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("max(primal, dual) residual", fontsize=12)
    plt.title("ADMM Convergence Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved to: {save_path}")

    plt.show()


def main():
    """Run the complete comparison example."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Geometry-Aware Compressive Sensing Example")
    print("=" * 60)

    # Generate test problem
    print("\nGenerating synthetic problem...")
    n_spatial = 50
    A, P, Y, x_true, spatial_shape = generate_test_problem(
        n_spatial=n_spatial, n_modes=20, sensor_fraction=0.3, noise_level=0.01
    )

    print(f"  Spatial grid: {spatial_shape}")
    print(f"  Number of modes: {A.shape[1]}")
    print(f"  Sensors: {P.sum()} / {P.size} ({100 * P.sum() / P.size:.1f}%)")
    print(f"  Active coefficients: {np.sum(np.abs(x_true) > 1e-6)} / {len(x_true)}")

    # Build mesh
    print("\nBuilding mesh graph...")
    builder = MeshGraphBuilder(connectivity_type="grid")
    mesh = builder.build_from_shape(spatial_shape)
    print(f"  Mesh cells: {mesh.adjacency_matrix.shape[0]}")
    print(f"  Laplacian type: {type(mesh.laplacian_matrix)}")

    # Compare methods
    print("\nRunning compressive sensing reconstruction...")
    results = compare_reconstructions(
        A, P, Y, x_true, spatial_shape, mesh, alpha_values=[0.0, 0.01, 0.05, 0.1]
    )

    # Visualize
    print("\nGenerating visualizations...")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    visualize_results(
        A, x_true, results, spatial_shape, save_path=output_dir / "geometry_aware_cs_comparison.png"
    )

    plot_convergence(results, save_path=output_dir / "geometry_aware_cs_convergence.png")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
