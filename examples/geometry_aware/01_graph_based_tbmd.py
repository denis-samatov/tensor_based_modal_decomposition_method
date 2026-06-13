"""
Geometry-Aware Graph-Based TBMD Example

This script demonstrates a graph-based TBMD workflow that accounts for
the geometric structure of reservoir-style spatial data.

Key concepts:
1. Build a connectivity graph from a spatial grid.
2. Compute a Laplacian matrix for regularization.
3. Run geometry-aware TBMD with cell-neighborhood information.
4. Compare with the standard workflow.

The example compares:
- spatial-mode structure;
- sensor placement;
- reconstruction metrics under sparse sensing;
- the use of spatial structure in the model setup.

Author: TBMD Team
Date: 2025
"""

import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from TBMD.utils.tbmd_utils import compute_reconstruction_metrics, set_seed

from TBMD.config import SEED
from TBMD.core.decomposition import GeometryAwareDecompositionConfig as GeometryAwareConfig
from TBMD.core.decomposition import GeometryAwareTuckerDecomposer, TuckerDecomposer
from TBMD.core.geometry import MeshGeometry, MeshGraphBuilder
from TBMD.core.reconstruction import (
    CompressiveSensingConfig,
    GeometryAwareCSConfig,
    GeometryAwareTensorCS,
    TensorCompressiveSensing,
)
from TBMD.core.sensor_placement import (
    GeometricQRConfig,
    GeometryAwareTensorQR,
    TensorTubeQRDecomposition,
)

# Configuration.
set_seed(SEED)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def generate_synthetic_field_data(
    spatial_shape: Tuple[int, int] = (50, 50), n_timesteps: int = 100, noise_level: float = 0.05
) -> torch.Tensor:
    """Generate synthetic data with spatial structure.

    Creates fields with smooth spatial gradients that approximate simple
    physical processes.
    """
    nx, ny = spatial_shape

    # Create a spatial grid.
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Time series.
    t = np.linspace(0, 2 * np.pi, n_timesteps)

    # Create data with spatial-temporal structure.
    data = np.zeros((nx, ny, n_timesteps))

    # Base field.
    base_field = 100 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.2)

    for i, time in enumerate(t):
        # Several spatial modes.
        mode1 = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * time)
        mode2 = np.cos(3 * np.pi * X) * np.sin(np.pi * Y) * np.sin(time)
        mode3 = np.sin(np.pi * X) * np.cos(4 * np.pi * Y) * np.cos(3 * time)

        # Local hot spots, used as a simple well-like proxy.
        hotspot1 = 20 * np.exp(-50 * ((X - 0.3) ** 2 + (Y - 0.3) ** 2)) * np.sin(time)
        hotspot2 = -15 * np.exp(-50 * ((X - 0.7) ** 2 + (Y - 0.7) ** 2)) * np.cos(1.5 * time)

        # Combined field.
        field = base_field + 10 * mode1 + 5 * mode2 + 3 * mode3 + hotspot1 + hotspot2

        # Add noise.
        field += np.random.normal(0, noise_level * np.std(field), field.shape)

        data[:, :, i] = field

    return torch.from_numpy(data).float()


def build_mesh_graph(
    spatial_shape: Tuple[int, int], connectivity_type: str = "grid", k: int = 8
) -> MeshGeometry:
    """Build a grid connectivity graph.

    Parameters
    ----------
    spatial_shape : tuple
        Spatial grid dimensions.
    connectivity_type : str
        Connectivity type: 'grid', 'knn', 'radius', 'delaunay'.
    k : int
        Number of neighbors for knn.
    """
    print(f"\n{'=' * 70}")
    print(f"Building Grid Connectivity Graph ({connectivity_type})".center(70))
    print(f"{'=' * 70}")

    builder = MeshGraphBuilder(
        connectivity_type=connectivity_type, k=k if connectivity_type == "knn" else None
    )

    mesh = builder.build_from_shape(spatial_shape)

    # Get the number of cells from the adjacency matrix size.
    n_cells = mesh.adjacency_matrix.shape[0]
    n_edges = mesh.adjacency_matrix.nnz

    print("\nGraph built:")
    print(f"  Nodes (cells): {n_cells}")
    print(f"  Edges: {n_edges}")
    print(f"  Average node degree: {n_edges / n_cells:.2f}")
    print("  Laplacian type: normalized")

    return mesh


def visualize_mesh_graph(
    mesh: MeshGeometry,
    spatial_shape: Tuple[int, int],
    save_path: str = "mesh_graph_connectivity.png",
):
    """Visualize the connectivity graph."""
    nx, ny = spatial_shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Adjacency matrix.
    ax1 = axes[0]
    adj_dense = mesh.adjacency_matrix.toarray()
    im1 = ax1.imshow(adj_dense, cmap="Blues", aspect="auto")
    ax1.set_title("Adjacency Matrix", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Node j", fontsize=10)
    ax1.set_ylabel("Node i", fontsize=10)
    plt.colorbar(im1, ax=ax1)

    # 2. Laplacian matrix.
    ax2 = axes[1]
    lap_dense = mesh.laplacian_matrix.toarray()
    im2 = ax2.imshow(lap_dense, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax2.set_title("Laplacian Matrix", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Node j", fontsize=10)
    ax2.set_ylabel("Node i", fontsize=10)
    plt.colorbar(im2, ax=ax2)

    # 3. Node degree (connectivity).
    ax3 = axes[2]
    degrees = np.array(mesh.adjacency_matrix.sum(axis=1)).flatten()
    degree_field = degrees.reshape(spatial_shape)
    im3 = ax3.imshow(degree_field, cmap="viridis", aspect="auto", origin="lower")
    ax3.set_title("Node Degree (Connectivity)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Y", fontsize=10)
    ax3.set_ylabel("X", fontsize=10)
    plt.colorbar(im3, ax=ax3, label="Number of neighbors")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraph visualization saved: {save_path}")
    plt.close()


def compare_standard_vs_geometry_aware(
    data: torch.Tensor, mesh: MeshGeometry, n_modes: int = 30
) -> Dict:
    """Compare standard and geometry-aware TBMD."""
    print(f"\n{'=' * 70}")
    print("Comparison: Standard vs Geometry-Aware TBMD".center(70))
    print(f"{'=' * 70}")

    results = {}

    # 1. Standard TBMD.
    print("\n[1/2] Standard TBMD (without geometry)...")
    start_time = time.time()

    standard_decomposer = TuckerDecomposer(
        tensors=data,  # Use 'tensors' instead of 'tensor'.
        ranks=[
            n_modes,
            n_modes,
            n_modes // 2,
        ],  # 3 ranks for a 3D tensor (spatial_x, spatial_y, time).
        device="cpu",
    )
    standard_decomposer.decompose()

    standard_time = time.time() - start_time
    standard_decomposer.reconstruct()  # Method updates internal state.
    standard_recon = standard_decomposer.reconstructed_tensors
    standard_error = torch.norm(data - standard_recon) / torch.norm(data)

    print(f"  Time: {standard_time:.2f}s")
    print(f"  Reconstruction error: {standard_error:.6f}")

    results["standard"] = {
        "decomposer": standard_decomposer,
        "reconstruction": standard_recon,
        "error": float(standard_error.item()),
        "time": standard_time,
        "factors": standard_decomposer.factors,
    }

    # 2. Geometry-aware TBMD.
    print("\n[2/2] Geometry-Aware TBMD (with geometry)...")
    start_time = time.time()

    geo_config = GeometryAwareConfig(
        alpha=0.05,  # Laplacian regularization weight.
        spatial_modes=[0],  # Regularize the spatial mode.
        laplacian_type="normalized",
        connectivity_type="grid",
    )

    # Geometry-aware decomposition expects combined spatial dimensions.
    # Convert (50, 50, 100) -> (2500, 100).
    spatial_size = data.shape[0] * data.shape[1]
    data_2d = data.reshape(spatial_size, data.shape[2])
    print(f"  Tensor shape for Geometry-Aware: {data_2d.shape}")

    geo_decomposer = GeometryAwareTuckerDecomposer(
        tensor=data_2d,  # Use 2D shape (spatial_cells, time).
        mesh=mesh,
        geo_config=geo_config,
        ranks=[n_modes, n_modes // 2],  # 2 ranks for a 2D tensor (spatial, time).
        device="cpu",
    )
    geo_decomposer.decompose()

    geo_time = time.time() - start_time
    geo_recon_2d = geo_decomposer.reconstruct()
    # Convert back to 3D for comparison.
    geo_recon = geo_recon_2d.reshape(data.shape[0], data.shape[1], data.shape[2])
    geo_error = torch.norm(data - geo_recon) / torch.norm(data)

    print(f"  Time: {geo_time:.2f}s")
    print(f"  Reconstruction error: {geo_error:.6f}")

    # Convert geometry-aware factors to match the standard layout.
    # geo_factors[0] has shape (2500, n_modes); convert to (50, 50, n_modes).
    geo_factors_reshaped = [
        geo_decomposer.factors[0].reshape(data.shape[0], data.shape[1], -1),  # spatial factor
        geo_decomposer.factors[1],  # temporal factor
    ]

    results["geometry_aware"] = {
        "decomposer": geo_decomposer,
        "reconstruction": geo_recon,
        "error": float(geo_error.item()),
        "time": geo_time,
        "factors": geo_factors_reshaped,  # For visualization: (50, 50, n_modes).
        "factors_original": geo_decomposer.factors,  # Original factors for CS: (2500, n_modes).
    }

    # Comparison.
    print(f"\n{'=' * 70}")
    print("Comparison Results".center(70))
    print(f"{'=' * 70}")
    print(f"\n{'Method':<30} {'Error':<15} {'Time (s)':<15}")
    print("-" * 60)
    print(f"{'Standard':<30} {standard_error:.6f}      {standard_time:.2f}")
    print(f"{'Geometry-Aware':<30} {geo_error:.6f}      {geo_time:.2f}")
    print("-" * 60)

    improvement = (standard_error - geo_error) / standard_error * 100
    print(f"\nAccuracy improvement in this run: {improvement:.2f}%")

    return results


def visualize_spatial_modes_comparison(
    results: Dict,
    spatial_shape: Tuple[int, int],
    n_modes_to_show: int = 4,
    save_path: str = "modes_comparison.png",
):
    """Visualize and compare spatial modes."""
    print("\nVisualizing spatial modes...")

    # For standard decomposition, combine the first two factors with an outer product.
    # factors[0]: (50, 30), factors[1]: (50, 30) -> combined: (50, 50, 30).
    standard_factor_0 = results["standard"]["factors"][0].cpu().numpy()  # (50, n_modes)
    standard_factor_1 = results["standard"]["factors"][1].cpu().numpy()  # (50, n_modes)

    # Create combined spatial factor via outer product.
    n_modes = standard_factor_0.shape[1]
    standard_modes_spatial = np.zeros((spatial_shape[0], spatial_shape[1], n_modes))
    for k in range(n_modes):
        # For each mode: outer product of two factors.
        standard_modes_spatial[:, :, k] = np.outer(standard_factor_0[:, k], standard_factor_1[:, k])

    # Geometry-aware factors are already in the required shape (50, 50, 30).
    geo_modes_spatial = results["geometry_aware"]["factors"][0].cpu().numpy()

    fig, axes = plt.subplots(n_modes_to_show, 3, figsize=(15, 4 * n_modes_to_show))

    for i in range(n_modes_to_show):
        # Standard mode.
        ax1 = axes[i, 0] if n_modes_to_show > 1 else axes[0]
        im1 = ax1.imshow(
            standard_modes_spatial[:, :, i], cmap="RdBu_r", aspect="auto", origin="lower"
        )
        ax1.set_title(f"Standard Mode {i + 1}", fontsize=11, fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Geometry-aware mode.
        ax2 = axes[i, 1] if n_modes_to_show > 1 else axes[1]
        im2 = ax2.imshow(geo_modes_spatial[:, :, i], cmap="RdBu_r", aspect="auto", origin="lower")
        ax2.set_title(f"Geometry-Aware Mode {i + 1}", fontsize=11, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Difference.
        ax3 = axes[i, 2] if n_modes_to_show > 1 else axes[2]
        diff = np.abs(standard_modes_spatial[:, :, i] - geo_modes_spatial[:, :, i])
        im3 = ax3.imshow(diff, cmap="Reds", aspect="auto", origin="lower")
        ax3.set_title(f"|Difference| (Mode {i + 1})", fontsize=11, fontweight="bold")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046)

    plt.suptitle(
        "Spatial Mode Comparison: Standard vs Geometry-Aware",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Mode visualization saved: {save_path}")
    plt.close()


def test_sensor_placement_with_geometry(
    data: torch.Tensor, mesh: MeshGeometry, n_sensors: int = 25
) -> Dict:
    """Test geometry-aware sensor placement."""
    print(f"\n{'=' * 70}")
    print("Sensor Placement: Standard vs Geometry-Aware".center(70))
    print(f"{'=' * 70}")

    results = {}

    # 1. Standard QR.
    print("\n[1/2] Standard Tensor QR...")
    standard_qr = TensorTubeQRDecomposition(
        tensor=data, N=n_sensors, device="cpu", uniform_distribution=True
    )
    P_standard, Q_standard, R_standard = standard_qr.factorize()

    print(f"  Sensors placed: {torch.sum(P_standard).item()}")

    results["standard"] = {"sensor_locations": P_standard, "Q": Q_standard, "R": R_standard}

    # 2. Geometry-Aware QR
    print("\n[2/2] Geometry-Aware Tensor QR...")

    # Convert tensor to 2D form for geometry-aware QR.
    # (50, 50, 100) -> (2500, 100)
    spatial_size = data.shape[0] * data.shape[1]
    data_2d_qr = data.reshape(spatial_size, data.shape[2])

    # Create geometry-aware QR configuration with amplitude information.
    geo_qr_config = GeometricQRConfig(
        gradient_weight=0.3,  # Field-gradient weight.
        amplitude_weight=1.5,  # Field-amplitude weight.
        energy_weight=0.8,  # Local-energy weight.
        proximity_weight=1.0,  # Minimum sensor-distance weight.
        distribution_weight=0.5,  # Distribution-uniformity weight.
        min_distance_factor=2.0,  # Minimum distance between sensors.
    )

    geo_qr = GeometryAwareTensorQR(
        tensor=data_2d_qr,  # Use 2D shape.
        mesh=mesh,
        N=n_sensors,
        config=geo_qr_config,
        device="cpu",
    )
    P_geo, Q_geo, R_geo = geo_qr.factorize()

    print(f"  Sensors placed: {torch.sum(P_geo).item()}")

    results["geometry_aware"] = {"sensor_locations": P_geo, "Q": Q_geo, "R": R_geo}

    return results


def visualize_sensor_placement_comparison(
    sensor_results: Dict, data: torch.Tensor, save_path: str = "sensor_placement_comparison.png"
):
    """Visualize sensor placement comparison."""
    print("\nVisualizing sensor placement...")

    P_standard = sensor_results["standard"]["sensor_locations"].cpu().numpy()
    P_geo_raw = sensor_results["geometry_aware"]["sensor_locations"].cpu().numpy()

    # If P_geo is 1D from a 2D tensor, convert it to 2D.
    spatial_shape = (data.shape[0], data.shape[1])
    if P_geo_raw.ndim == 1:
        P_geo = P_geo_raw.reshape(spatial_shape)
    else:
        P_geo = P_geo_raw

    # Base field.
    base_field = data[:, :, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Standard placement.
    ax1 = axes[0]
    ax1.imshow(base_field, cmap="viridis", alpha=0.5, aspect="auto", origin="lower")
    sensor_pos_std = np.argwhere(P_standard == 1)
    if len(sensor_pos_std) > 0:
        ax1.scatter(
            sensor_pos_std[:, 1],
            sensor_pos_std[:, 0],
            c="red",
            marker="o",
            s=80,
            edgecolors="black",
            linewidths=1.5,
            label=f"Sensors ({len(sensor_pos_std)})",
        )
    ax1.set_title("Standard Placement", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.axis("off")

    # 2. Geometry-aware placement.
    ax2 = axes[1]
    ax2.imshow(base_field, cmap="viridis", alpha=0.5, aspect="auto", origin="lower")
    sensor_pos_geo = np.argwhere(P_geo == 1)
    if len(sensor_pos_geo) > 0:
        ax2.scatter(
            sensor_pos_geo[:, 1],
            sensor_pos_geo[:, 0],
            c="blue",
            marker="s",
            s=80,
            edgecolors="black",
            linewidths=1.5,
            label=f"Sensors ({len(sensor_pos_geo)})",
        )
    ax2.set_title("Geometry-Aware Placement", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.axis("off")

    # 3. Distribution comparison.
    ax3 = axes[2]

    # Compute row and column distributions.
    std_rows = P_standard.sum(axis=1)
    P_standard.sum(axis=0)
    geo_rows = P_geo.sum(axis=1)
    P_geo.sum(axis=0)

    # Plot histograms.
    x = np.arange(len(std_rows))
    width = 0.35

    ax3.bar(x - width / 2, std_rows, width, label="Standard", alpha=0.7, color="red")
    ax3.bar(x + width / 2, geo_rows, width, label="Geometry-Aware", alpha=0.7, color="blue")

    ax3.set_xlabel("Row (X)", fontsize=10)
    ax3.set_ylabel("Number of Sensors", fontsize=10)
    ax3.set_title("Sensor Distribution over X", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Sensor placement visualization saved: {save_path}")
    plt.close()


def test_reconstruction_with_geometry(
    data: torch.Tensor, mesh: MeshGeometry, sensor_results: Dict, decomposition_results: Dict
) -> Dict:
    """Test reconstruction with geometry-aware components."""
    print(f"{'=' * 70}")
    print("Field Reconstruction: Standard vs Geometry-Aware CS".center(70))
    print(f"{'=' * 70}")

    results = {}

    # Select a test time slice.
    test_time_idx = data.shape[-1] // 2
    test_field = data[:, :, test_time_idx]

    # Get spatial shape.
    spatial_shape = test_field.shape  # (spatial_x, spatial_y)
    spatial_size = int(np.prod(spatial_shape))

    # Standard basis.
    standard_basis = decomposition_results["standard"]["factors"][0]
    print(f"  Standard factor[0] shape: {standard_basis.shape}")
    print(f"  Standard factor[1] shape: {decomposition_results['standard']['factors'][1].shape}")

    # Geometry-aware basis: use the original factors for CS.
    geo_basis_original = decomposition_results["geometry_aware"]["factors_original"][
        0
    ]  # (2500, n_modes)
    print(f"  Geo-aware factor[0] (original) shape: {geo_basis_original.shape}")
    print(
        f"  Geo-aware factor[1] shape: {decomposition_results['geometry_aware']['factors_original'][1].shape}"
    )

    # Convert the original factor to 3D form for CS.
    geo_basis = geo_basis_original.reshape(*spatial_shape, -1)  # (50, 50, n_modes)
    print(f"  Geo-aware basis reshaped for CS: {geo_basis.shape}")

    # Sensors.
    P_standard = sensor_results["standard"]["sensor_locations"]
    P_geo_raw = sensor_results["geometry_aware"]["sensor_locations"]
    if P_standard.ndim == 3:
        P_standard_2d = P_standard[:, :, 0]
    else:
        P_standard_2d = P_standard
    if P_geo_raw.ndim == 1:
        P_geo_2d = P_geo_raw.reshape(spatial_shape)
    elif P_geo_raw.ndim == 3:
        P_geo_2d = P_geo_raw[:, :, 0]
    else:
        P_geo_2d = P_geo_raw

    # 1. Standard reconstruction.
    print("[1/4] Standard reconstruction (CS, standard sensors)...")

    # Build A_std dictionary from factors.
    standard_factor_0 = decomposition_results["standard"]["factors"][0].cpu().numpy()
    standard_factor_1 = decomposition_results["standard"]["factors"][1].cpu().numpy()
    n_modes = standard_factor_0.shape[1]

    A_std_np = np.zeros((spatial_size, n_modes))
    for k in range(n_modes):
        mode_2d = np.outer(standard_factor_0[:, k], standard_factor_1[:, k])
        A_std_np[:, k] = mode_2d.flatten()

    A_std = torch.from_numpy(A_std_np).float().reshape(*spatial_shape, -1)

    Y_std = torch.zeros_like(test_field)
    sensor_mask_std = P_standard_2d.bool()
    Y_std[sensor_mask_std] = test_field[sensor_mask_std]
    n_sensors_std = int(sensor_mask_std.sum().item())
    print(
        f"  Standard sensors: {n_sensors_std} ({n_sensors_std / test_field.numel() * 100:.2f}% sampling)"
    )
    print(
        f"  Measurement stats: mean={test_field[sensor_mask_std].mean():.2f}, min={test_field[sensor_mask_std].min():.2f}, max={test_field[sensor_mask_std].max():.2f}"
    )

    cs_config = CompressiveSensingConfig(max_iter=100, tol=1e-3, device="cpu")

    reconstructor_std = TensorCompressiveSensing(
        A=A_std, P=P_standard_2d, Y=Y_std, core_cfg=cs_config
    )

    x_std, _ = reconstructor_std.solve()
    print(
        f"  x_std shape: {x_std.shape}, norm: {torch.norm(x_std):.4f}, sparsity: {(x_std.abs() < 1e-3).sum().item()}/{x_std.numel()}"
    )

    recon_std_field = torch.einsum("ijk,k->ij", A_std, x_std).cpu()
    print(
        f"  Recon range: [{recon_std_field.min():.2f}, {recon_std_field.max():.2f}], true: [{test_field.min():.2f}, {test_field.max():.2f}]"
    )

    metrics_std = compute_reconstruction_metrics(test_field, recon_std_field)
    print(f"  RMSE: {metrics_std['rmse']:.6f}")
    print(f"  SSIM: {metrics_std['ssim']:.6f}")
    print(f"  Relative Error: {metrics_std['relative_error']:.6f}")

    results["standard"] = {"reconstruction": recon_std_field, "metrics": metrics_std}

    # Build A_geo dictionary from the geometry-aware basis.
    A_geo = geo_basis.cpu()  # geo_basis is already in the required shape (50, 50, n_modes).

    # 2. Geometry-aware reconstruction with geometry-aware CS.
    print("[2/4] Geometry-Aware reconstruction (Geometry-Aware CS, geometry sensors)...")

    Y_geo = torch.zeros_like(test_field)
    sensor_mask_geo = P_geo_2d.bool()
    Y_geo[sensor_mask_geo] = test_field[sensor_mask_geo]
    n_sensors_geo = int(sensor_mask_geo.sum().item())
    print(
        f"  Geometry sensors: {n_sensors_geo} ({n_sensors_geo / test_field.numel() * 100:.2f}% sampling)"
    )
    print(
        f"  Measurement stats: mean={test_field[sensor_mask_geo].mean():.2f}, min={test_field[sensor_mask_geo].min():.2f}, max={test_field[sensor_mask_geo].max():.2f}"
    )

    geo_cs_config = GeometryAwareCSConfig(max_iter=100, tol=1e-3, alpha=0.05, device="cpu")

    reconstructor_geo = GeometryAwareTensorCS(
        A=A_geo, P=P_geo_2d, Y=Y_geo, mesh=mesh, core_cfg=geo_cs_config
    )

    x_geo, _ = reconstructor_geo.solve()
    print(
        f"  x_geo shape: {x_geo.shape}, norm: {torch.norm(x_geo):.4f}, sparsity: {(x_geo.abs() < 1e-3).sum().item()}/{x_geo.numel()}"
    )

    recon_geo_field = torch.einsum("ijk,k->ij", A_geo, x_geo).cpu()
    print(
        f"  Recon range: [{recon_geo_field.min():.2f}, {recon_geo_field.max():.2f}], true: [{test_field.min():.2f}, {test_field.max():.2f}]"
    )

    metrics_geo = compute_reconstruction_metrics(test_field, recon_geo_field)
    print(f"  RMSE: {metrics_geo['rmse']:.6f}")
    print(f"  SSIM: {metrics_geo['ssim']:.6f}")
    print(f"  Relative Error: {metrics_geo['relative_error']:.6f}")

    results["geometry_aware"] = {"reconstruction": recon_geo_field, "metrics": metrics_geo}

    # 3. Cross combinations with the same solver (standard CS).
    print("[3/4] Cross combinations (standard CS solver for both pairs)...")
    cross_results = {}

    def run_cross_case(
        label: str,
        A_tensor: torch.Tensor,
        P_mask: torch.Tensor,
        Y_measure: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> None:
        print(f"{label}")
        cs_solver = TensorCompressiveSensing(A=A_tensor, P=P_mask, Y=Y_measure, core_cfg=cs_config)
        x_hat, _ = cs_solver.solve()
        print(
            f"  x shape: {x_hat.shape}, norm: {torch.norm(x_hat):.4f}, sparsity: {(x_hat.abs() < 1e-3).sum().item()}/{x_hat.numel()}"
        )
        recon_field = torch.einsum("ijk,k->ij", A_tensor, x_hat).cpu()
        print(f"  Recon range: [{recon_field.min():.2f}, {recon_field.max():.2f}]")
        metrics = compute_reconstruction_metrics(test_field, recon_field)
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  SSIM: {metrics['ssim']:.6f}")
        print(f"  Relative Error: {metrics['relative_error']:.6f}")
        cross_results[label] = {
            "reconstruction": recon_field,
            "metrics": metrics,
            "sensors": int(sensor_mask.sum().item()),
        }

    run_cross_case("Std basis + Geo sensors (standard CS)", A_std, P_geo_2d, Y_geo, sensor_mask_geo)

    run_cross_case(
        "Geo basis + Std sensors (standard CS)", A_geo, P_standard_2d, Y_std, sensor_mask_std
    )

    # 4. Summary table.
    print(f"{'=' * 70}")
    print("Reconstruction Results".center(70))
    print(f"{'=' * 70}")
    print(f"{'Method':<45} {'RMSE':<12} {'SSIM':<12} {'Rel.Error':<12}")
    print("-" * 85)
    print(
        f"{'Standard (standard CS)':<45} {metrics_std['rmse']:.6f}  {metrics_std['ssim']:.6f}  {metrics_std['relative_error']:.6f}"
    )
    print(
        f"{'Geometry-Aware (geo CS)':<45} {metrics_geo['rmse']:.6f}  {metrics_geo['ssim']:.6f}  {metrics_geo['relative_error']:.6f}"
    )
    for label, entry in cross_results.items():
        m = entry["metrics"]
        print(f"{label:<45} {m['rmse']:.6f}  {m['ssim']:.6f}  {m['relative_error']:.6f}")
    print("-" * 85)

    results["cross"] = cross_results
    results["true_field"] = test_field

    return results


def visualize_reconstruction_comparison(
    recon_results: Dict, save_path: str = "reconstruction_comparison.png"
):
    """Visualize reconstruction comparison."""
    print("\nVisualizing reconstruction...")

    true_field = recon_results["true_field"].cpu().numpy()
    recon_std = recon_results["standard"]["reconstruction"].cpu().numpy()
    recon_geo = recon_results["geometry_aware"]["reconstruction"].cpu().numpy()

    error_std = np.abs(true_field - recon_std)
    error_geo = np.abs(true_field - recon_geo)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin, vmax = true_field.min(), true_field.max()

    # Row 1: fields.
    # True field.
    ax1 = axes[0, 0]
    im1 = ax1.imshow(
        true_field, cmap="viridis", aspect="auto", origin="lower", vmin=vmin, vmax=vmax
    )
    ax1.set_title("True Field", fontsize=12, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Standard reconstruction.
    ax2 = axes[0, 1]
    im2 = ax2.imshow(recon_std, cmap="viridis", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    metrics_std = recon_results["standard"]["metrics"]
    ax2.set_title(f"Standard (SSIM={metrics_std['ssim']:.4f})", fontsize=12, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Geometry-aware reconstruction.
    ax3 = axes[0, 2]
    im3 = ax3.imshow(recon_geo, cmap="viridis", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    metrics_geo = recon_results["geometry_aware"]["metrics"]
    ax3.set_title(
        f"Geometry-Aware (SSIM={metrics_geo['ssim']:.4f})", fontsize=12, fontweight="bold"
    )
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: errors.
    error_max = max(error_std.max(), error_geo.max())

    # Empty cell.
    axes[1, 0].axis("off")

    # Standard error.
    ax5 = axes[1, 1]
    im5 = ax5.imshow(error_std, cmap="Reds", aspect="auto", origin="lower", vmin=0, vmax=error_max)
    ax5.set_title(f"Error (RMSE={metrics_std['rmse']:.4f})", fontsize=12, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Geometry-aware error.
    ax6 = axes[1, 2]
    im6 = ax6.imshow(error_geo, cmap="Reds", aspect="auto", origin="lower", vmin=0, vmax=error_max)
    ax6.set_title(f"Error (RMSE={metrics_geo['rmse']:.4f})", fontsize=12, fontweight="bold")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    plt.suptitle(
        "Reconstruction Comparison: Standard vs Geometry-Aware",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Reconstruction visualization saved: {save_path}")
    plt.close()


def main():
    """Run the example."""
    print("=" * 70)
    print(" Geometry-Aware Graph-Based TBMD Demo ".center(70, "="))
    print("=" * 70)
    print()

    # Parameters.
    spatial_shape = (50, 50)
    n_timesteps = 100
    n_modes = 30
    n_sensors = 25

    print("Experiment parameters:")
    print(f"  Spatial shape: {spatial_shape}")
    print(f"  Time steps: {n_timesteps}")
    print(f"  Number of modes: {n_modes}")
    print(f"  Number of sensors: {n_sensors}")

    # 1. Data generation.
    print(f"\n{'=' * 70}")
    print("Generating Synthetic Data".center(70))
    print(f"{'=' * 70}")

    data = generate_synthetic_field_data(spatial_shape, n_timesteps)
    print(f"\nData generated: {data.shape}")
    print(f"  Value range: [{data.min():.2f}, {data.max():.2f}]")

    # 2. Graph construction.
    mesh = build_mesh_graph(spatial_shape, connectivity_type="grid")
    visualize_mesh_graph(mesh, spatial_shape)

    # 3. Decomposition comparison.
    decomp_results = compare_standard_vs_geometry_aware(data, mesh, n_modes)
    visualize_spatial_modes_comparison(decomp_results, spatial_shape)

    # 4. Sensor placement comparison.
    sensor_results = test_sensor_placement_with_geometry(data, mesh, n_sensors)
    visualize_sensor_placement_comparison(sensor_results, data)

    # 5. Reconstruction comparison.
    recon_results = test_reconstruction_with_geometry(data, mesh, sensor_results, decomp_results)
    visualize_reconstruction_comparison(recon_results)

    # 6. Adaptive alpha.
    test_adaptive_alpha(data, mesh, sensor_results, decomp_results)

    # Final summary.
    print(f"\n{'=' * 70}")
    print(" Final Summary ".center(70, "="))
    print(f"{'=' * 70}")

    print("\n1. Decomposition:")
    print(f"   Standard error: {decomp_results['standard']['error']:.6f}")
    print(f"   Geometry-Aware error: {decomp_results['geometry_aware']['error']:.6f}")
    improvement_decomp = (
        (decomp_results["standard"]["error"] - decomp_results["geometry_aware"]["error"])
        / decomp_results["standard"]["error"]
        * 100
    )
    print(f"   Improvement in this run: {improvement_decomp:.2f}%")

    print("\n2. Sensor placement:")
    n_std = torch.sum(sensor_results["standard"]["sensor_locations"]).item()
    n_geo = torch.sum(sensor_results["geometry_aware"]["sensor_locations"]).item()
    print(f"   Standard: {n_std} sensors")
    print(f"   Geometry-Aware: {n_geo} sensors")

    print("\n3. Reconstruction:")
    metrics_std = recon_results["standard"]["metrics"]
    metrics_geo = recon_results["geometry_aware"]["metrics"]
    print(f"   Standard SSIM: {metrics_std['ssim']:.6f}")
    print(f"   Geometry-Aware SSIM: {metrics_geo['ssim']:.6f}")
    ssim_improvement = (metrics_geo["ssim"] - metrics_std["ssim"]) / metrics_std["ssim"] * 100
    print(f"   Improvement in this run: {ssim_improvement:.2f}%")

    print(f"\n{'=' * 70}")
    print(" Demonstration Complete ".center(70, "="))
    print(f"{'=' * 70}")

    print("\nGenerated files:")
    print("  - mesh_graph_connectivity.png")
    print("  - modes_comparison.png")
    print("  - sensor_placement_comparison.png")
    print("  - reconstruction_comparison.png")

    print("\nGeometry-aware outputs generated for comparison:")
    print("  - spatial mode comparison")
    print("  - sensor placement comparison")
    print("  - reconstruction comparison")
    print("  - adaptive alpha comparison")


def test_adaptive_alpha(
    data: torch.Tensor, mesh: MeshGeometry, sensor_results: Dict, decomposition_results: Dict
) -> Dict:
    """Test adaptive alpha for Geometry-Aware CS.

    Evaluates how alpha adaptation changes metrics for the sampled measurements.
    """
    print(f"\n{'=' * 70}")
    print("Adaptive Alpha for Geometry-Aware CS".center(70))
    print(f"{'=' * 70}")

    print("\nConcept:")
    print("  Higher measurements (mean=88) -> less smoothing (lower alpha)")
    print("  Lower measurements (mean=44) -> more smoothing (higher alpha)")
    print("  Formula: alpha_adaptive = alpha_base * (reference / actual)")

    # Prepare data.
    test_time_idx = data.shape[-1] // 2
    test_field = data[:, :, test_time_idx]
    spatial_shape = test_field.shape

    geo_basis_original = decomposition_results["geometry_aware"]["factors_original"][0]
    geo_basis = geo_basis_original.reshape(*spatial_shape, -1)

    P_geo_raw = sensor_results["geometry_aware"]["sensor_locations"]
    if P_geo_raw.ndim == 1:
        P_geo_2d = P_geo_raw.reshape(spatial_shape)
    elif P_geo_raw.ndim == 3:
        P_geo_2d = P_geo_raw[:, :, 0]
    else:
        P_geo_2d = P_geo_raw

    Y_geo = torch.zeros_like(test_field)
    sensor_mask_geo = P_geo_2d.bool()
    Y_geo[sensor_mask_geo] = test_field[sensor_mask_geo]

    measurement_mean = test_field[sensor_mask_geo].mean().item()
    print(f"\nMeasurement statistics: mean={measurement_mean:.2f}")

    # Test 1: Fixed alpha
    print("\n[1/2] Fixed alpha=0.05 (baseline)...")
    geo_cs_config_fixed = GeometryAwareCSConfig(
        max_iter=100, tol=1e-3, alpha=0.05, adaptive_alpha=False, device="cpu"
    )

    reconstructor_fixed = GeometryAwareTensorCS(
        A=geo_basis.cpu(), P=P_geo_2d, Y=Y_geo, mesh=mesh, core_cfg=geo_cs_config_fixed
    )

    x_fixed, _ = reconstructor_fixed.solve()
    recon_fixed = torch.einsum("ijk,k->ij", geo_basis.cpu(), x_fixed)
    metrics_fixed = compute_reconstruction_metrics(test_field, recon_fixed)

    print(f"  RMSE: {metrics_fixed['rmse']:.6f}, SSIM: {metrics_fixed['ssim']:.6f}")
    print(f"  Alpha: {reconstructor_fixed.alpha:.6f}")

    # Test 2: Adaptive alpha
    print("\n[2/2] Adaptive alpha (reference=70.0)...")
    geo_cs_config_adapt = GeometryAwareCSConfig(
        max_iter=100,
        tol=1e-3,
        alpha=0.05,
        adaptive_alpha=True,
        alpha_reference_amplitude=70.0,
        device="cpu",
    )

    reconstructor_adapt = GeometryAwareTensorCS(
        A=geo_basis.cpu(), P=P_geo_2d, Y=Y_geo, mesh=mesh, core_cfg=geo_cs_config_adapt
    )

    x_adapt, _ = reconstructor_adapt.solve()
    recon_adapt = torch.einsum("ijk,k->ij", geo_basis.cpu(), x_adapt)
    metrics_adapt = compute_reconstruction_metrics(test_field, recon_adapt)

    print(f"  RMSE: {metrics_adapt['rmse']:.6f}, SSIM: {metrics_adapt['ssim']:.6f}")
    print(f"  Alpha: {reconstructor_adapt.alpha:.6f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("Results".center(70))
    print(f"{'=' * 70}")
    print(f"{'Method':<30} {'RMSE':<12} {'SSIM':<12} {'Alpha':<12}")
    print("-" * 68)
    print(
        f"{'Fixed alpha=0.05':<30} {metrics_fixed['rmse']:.6f}  {metrics_fixed['ssim']:.6f}  {reconstructor_fixed.alpha:.6f}"
    )
    print(
        f"{'Adaptive alpha':<30} {metrics_adapt['rmse']:.6f}  {metrics_adapt['ssim']:.6f}  {reconstructor_adapt.alpha:.6f}"
    )
    print("-" * 68)
    print(f"{'Best baseline (Std+Geo)':<30} {'20.376736':<12} {'0.136554':<12}")
    print(f"{'=' * 70}")

    if metrics_adapt["rmse"] < 20.376736:
        improvement = (20.376736 - metrics_adapt["rmse"]) / 20.376736 * 100
        print(f"\nAdaptive alpha improved on the baseline by {improvement:.1f}% in this run.")
    elif metrics_adapt["rmse"] < metrics_fixed["rmse"]:
        improvement = (metrics_fixed["rmse"] - metrics_adapt["rmse"]) / metrics_fixed["rmse"] * 100
        print(f"\nAdaptive alpha improved on fixed alpha by {improvement:.1f}% in this run.")

    return {
        "fixed": {"metrics": metrics_fixed, "alpha": reconstructor_fixed.alpha},
        "adaptive": {"metrics": metrics_adapt, "alpha": reconstructor_adapt.alpha},
    }


if __name__ == "__main__":
    main()
