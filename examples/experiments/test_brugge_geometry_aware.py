#!/usr/bin/env python
"""Geometry-aware TBMD experiment for the Brugge dataset.

Tests the geometry-aware approach on local Brugge reservoir data.

Dataset: Brugge (139 x 48 x 2 x 133)
- Spatial grid: 139 x 48 = 6,672 cells
- 2 variables: pressure and saturation
- 133 time steps

Tested components:
1. Standard TBMD (baseline)
2. Improved Geometry-aware QR (amplitude + energy weights)
3. Geometry-aware Tucker decomposition
4. Geometry-aware CS with alpha=0.15
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

warnings.filterwarnings("ignore")

# Style configuration.
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# === TBMD imports ===
try:
    # Standard modules
    from TBMD.utils.tbmd_utils import compute_reconstruction_metrics, set_seed

    from TBMD.config import SEED
    from TBMD.modules.GeometryAwareTensorCS import GeometryAwareCSConfig, GeometryAwareTensorCS

    # Geometry-aware modules
    from TBMD.modules.GeometryAwareTensorHOSVD import (
        GeometryAwareConfig,
        GeometryAwareTuckerDecomposer,
    )
    from TBMD.modules.GeometryAwareTensorQR import GeometricQRConfig, GeometryAwareTensorQR
    from TBMD.modules.TensorBasedCompressiveSensing import (
        CompressiveSensingConfig,
        TensorCompressiveSensing,
    )
    from TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import (
        TensorQRConfig,
        TensorTubeQRDecomposition,
    )
    from TBMD.modules.TensorHOSVD import TuckerDecomposer
    from TBMD.utils.DataLoader import DataLoader

    # Utils
    from TBMD.utils.geometry import MeshGeometry, MeshGraphBuilder

    set_seed(SEED)
    print(f"TBMD modules loaded. Seed: {SEED}\n")

except Exception as e:
    print(f"Import error: {e}")
    raise


def load_brugge_data():
    """Load the Brugge dataset."""
    print("=" * 70)
    print("Loading Brugge Dataset".center(70))
    print("=" * 70)

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "Brugge data"
    h5_path = data_dir / "data_exp_4_.h5"
    wells_path = data_dir / "all_wells_exp_4.json"

    tensors = DataLoader.load_h5_tensors(str(h5_path))
    wells = DataLoader.load_wells_from_json(str(wells_path))

    # Swap coordinates
    for case_id in wells:
        wells[case_id] = [[y, x] for x, y in wells[case_id]]

    # Select case1.
    case_name = "case1"
    full_tensor = tensors["all"][case_name]  # (139, 48, 2, 133)

    # Extract pressure (variable index = 0).
    pressure_tensor = torch.from_numpy(full_tensor[:, :, 0, :]).float()  # (139, 48, 133)

    print("\nData loaded:")
    print(f"  Shape: {pressure_tensor.shape} (x, y, time)")
    print(
        f"  Grid: {pressure_tensor.shape[0]} x {pressure_tensor.shape[1]} = {pressure_tensor.shape[0] * pressure_tensor.shape[1]} cells"
    )
    print(f"  Time steps: {pressure_tensor.shape[2]}")
    print(f"  Range: [{pressure_tensor.min():.2f}, {pressure_tensor.max():.2f}]")
    print(f"  Wells: {len(wells[case_name])}")

    return pressure_tensor, wells[case_name]


def build_mesh_graph(spatial_shape: Tuple[int, int]):
    """Build a connectivity graph."""
    print("\n" + "=" * 70)
    print("Building Connectivity Graph".center(70))
    print("=" * 70)

    builder = MeshGraphBuilder(connectivity_type="grid")
    mesh = builder.build_from_shape(spatial_shape)

    n_cells = mesh.adjacency_matrix.shape[0]
    n_edges = mesh.adjacency_matrix.nnz

    print("\nGraph built:")
    print(f"  Nodes: {n_cells}")
    print(f"  Edges: {n_edges}")
    print(f"  Average degree: {n_edges / n_cells:.2f}")

    return mesh


def test_decomposition(data: torch.Tensor, mesh: MeshGeometry, n_modes: int):
    """Compare standard and geometry-aware decomposition."""
    print("\n" + "=" * 70)
    print("Decomposition: Standard vs Geometry-Aware".center(70))
    print("=" * 70)

    spatial_shape = data.shape[:2]
    n_timesteps = data.shape[2]

    # Standard
    print("\n[1/2] Standard Tucker...")
    standard_decomposer = TuckerDecomposer(
        tensors=data, ranks=[n_modes, n_modes, n_modes // 2], device="cpu"
    )
    standard_decomposer.decompose()
    standard_decomposer.reconstruct()
    standard_recon = standard_decomposer.reconstructed_tensors
    standard_error = torch.norm(data - standard_recon) / torch.norm(data)
    print(f"  Error: {standard_error:.6f}")

    # Geometry-aware
    print("\n[2/2] Geometry-Aware Tucker...")
    data_2d = data.reshape(-1, n_timesteps)

    geo_config = GeometryAwareConfig(alpha=0.05, spatial_modes=[0], laplacian_type="normalized")

    geo_decomposer = GeometryAwareTuckerDecomposer(
        tensor=data_2d,
        mesh=mesh,
        geo_config=geo_config,
        ranks=[n_modes, n_modes // 2],
        device="cpu",
    )
    geo_decomposer.decompose()
    geo_recon_2d = geo_decomposer.reconstruct()
    geo_recon = geo_recon_2d.reshape(spatial_shape[0], spatial_shape[1], n_timesteps)
    geo_error = torch.norm(data - geo_recon) / torch.norm(data)
    print(f"  Error: {geo_error:.6f}")

    # Summary
    print(f"\n{'Method':<25} {'Error':<12} {'Improvement'}")
    print("-" * 50)
    print(f"{'Standard':<25} {standard_error:.6f}  {'-'}")
    print(
        f"{'Geometry-Aware':<25} {geo_error:.6f}  {(standard_error - geo_error) / standard_error * 100:+.2f}%"
    )

    return {
        "standard": {
            "decomposer": standard_decomposer,
            "factors": standard_decomposer.factors,
            "error": standard_error.item(),
        },
        "geometry_aware": {
            "decomposer": geo_decomposer,
            "factors": geo_decomposer.factors,
            "factors_original": geo_decomposer.factors,
            "error": geo_error.item(),
        },
    }


def test_sensor_placement(data: torch.Tensor, mesh: MeshGeometry, n_sensors: int):
    """Compare standard and geometry-aware QR sensor placement."""
    print("\n" + "=" * 70)
    print("Sensor Placement: Standard vs Geometry-Aware".center(70))
    print("=" * 70)

    spatial_shape = data.shape[:2]
    n_timesteps = data.shape[2]

    # Standard QR
    print("\n[1/2] Standard QR...")
    std_qr = TensorTubeQRDecomposition(
        tensor=data, N=n_sensors, config=TensorQRConfig(), device="cpu"
    )
    P_standard, _, _ = std_qr.factorize()
    n_std = torch.sum(P_standard).item()
    print(f"  Sensors: {n_std}")

    # Geometry-aware QR with amplitude and energy weights.
    print("\n[2/2] Geometry-Aware QR (amplitude + energy)...")
    data_2d = data.reshape(-1, n_timesteps)

    geo_qr_config = GeometricQRConfig(
        gradient_weight=0.3,
        amplitude_weight=1.5,
        energy_weight=0.8,
        proximity_weight=1.0,
        distribution_weight=0.5,
        min_distance_factor=2.0,
    )

    geo_qr = GeometryAwareTensorQR(
        tensor=data_2d, mesh=mesh, N=n_sensors, config=geo_qr_config, device="cpu"
    )
    P_geo, _, _ = geo_qr.factorize()
    n_geo = torch.sum(P_geo).item()
    print(f"  Sensors: {n_geo}")

    # Reshape P_geo
    P_geo_2d = P_geo.reshape(spatial_shape) if P_geo.ndim == 1 else P_geo

    return {
        "standard": {"sensor_locations": P_standard},
        "geometry_aware": {"sensor_locations": P_geo_2d},
    }


def test_reconstruction(
    data: torch.Tensor, mesh: MeshGeometry, sensor_results: Dict, decomp_results: Dict, n_modes: int
):
    """Test all CS combinations."""
    print("\n" + "=" * 70)
    print("Compressive Sensing: All Combinations".center(70))
    print("=" * 70)

    spatial_shape = data.shape[:2]
    n_timesteps = data.shape[2]
    test_time_idx = n_timesteps // 2
    test_field = data[:, :, test_time_idx]

    results = {}

    # Build bases
    std_factor_0 = decomp_results["standard"]["factors"][0].cpu().numpy()
    std_factor_1 = decomp_results["standard"]["factors"][1].cpu().numpy()
    A_std_np = np.zeros((spatial_shape[0] * spatial_shape[1], n_modes))
    for k in range(n_modes):
        mode_2d = np.outer(std_factor_0[:, k], std_factor_1[:, k])
        A_std_np[:, k] = mode_2d.flatten()
    A_std = torch.from_numpy(A_std_np).float().reshape(*spatial_shape, -1)

    geo_basis_original = decomp_results["geometry_aware"]["factors_original"][0]
    A_geo = geo_basis_original.reshape(*spatial_shape, -1).cpu()

    # Sensor masks
    P_standard = sensor_results["standard"]["sensor_locations"]
    P_geo = sensor_results["geometry_aware"]["sensor_locations"]
    P_std_2d = P_standard[:, :, 0] if P_standard.ndim == 3 else P_standard

    # 1. Std + Std
    print("\n[1/4] Std + Std...")
    Y_std = torch.zeros_like(test_field)
    mask_std = P_std_2d.bool()
    Y_std[mask_std] = test_field[mask_std]

    cs_std = TensorCompressiveSensing(
        A=A_std,
        P=P_std_2d,
        Y=Y_std,
        core_cfg=CompressiveSensingConfig(max_iter=100, tol=1e-3, device="cpu"),
    )
    x_std, _ = cs_std.solve()
    recon_std = torch.einsum("ijk,k->ij", A_std, x_std).cpu()
    metrics_std = compute_reconstruction_metrics(test_field, recon_std)
    print(f"  RMSE: {metrics_std['rmse']:.6f}, SSIM: {metrics_std['ssim']:.6f}")
    results["std_std"] = metrics_std

    # 2. Std + Geo
    print("\n[2/4] Std + Geo...")
    Y_std_geo = torch.zeros_like(test_field)
    mask_geo = P_geo.bool()
    Y_std_geo[mask_geo] = test_field[mask_geo]

    cs_std_geo = TensorCompressiveSensing(
        A=A_std,
        P=P_geo,
        Y=Y_std_geo,
        core_cfg=CompressiveSensingConfig(max_iter=100, tol=1e-3, device="cpu"),
    )
    x_std_geo, _ = cs_std_geo.solve()
    recon_std_geo = torch.einsum("ijk,k->ij", A_std, x_std_geo).cpu()
    metrics_std_geo = compute_reconstruction_metrics(test_field, recon_std_geo)
    print(f"  RMSE: {metrics_std_geo['rmse']:.6f}, SSIM: {metrics_std_geo['ssim']:.6f}")
    results["std_geo"] = metrics_std_geo

    # 3. Geo + Std
    print("\n[3/4] Geo + Std...")
    Y_geo_std = torch.zeros_like(test_field)
    Y_geo_std[mask_std] = test_field[mask_std]

    cs_geo_std = TensorCompressiveSensing(
        A=A_geo,
        P=P_std_2d,
        Y=Y_geo_std,
        core_cfg=CompressiveSensingConfig(max_iter=100, tol=1e-3, device="cpu"),
    )
    x_geo_std, _ = cs_geo_std.solve()
    recon_geo_std = torch.einsum("ijk,k->ij", A_geo, x_geo_std).cpu()
    metrics_geo_std = compute_reconstruction_metrics(test_field, recon_geo_std)
    print(f"  RMSE: {metrics_geo_std['rmse']:.6f}, SSIM: {metrics_geo_std['ssim']:.6f}")
    results["geo_std"] = metrics_geo_std

    # 4. Geo + Geo (optimal alpha=0.15)
    print("\n[4/4] Geo + Geo (alpha=0.15)...")
    Y_geo_geo = torch.zeros_like(test_field)
    Y_geo_geo[mask_geo] = test_field[mask_geo]

    geo_cs_config = GeometryAwareCSConfig(
        max_iter=100, tol=1e-3, alpha=0.15, auto_alpha=False, adaptive_alpha=False, device="cpu"
    )

    cs_geo_geo = GeometryAwareTensorCS(
        A=A_geo, P=P_geo, Y=Y_geo_geo, mesh=mesh, core_cfg=geo_cs_config
    )
    x_geo_geo, _ = cs_geo_geo.solve()
    recon_geo_geo = torch.einsum("ijk,k->ij", A_geo, x_geo_geo).cpu()
    metrics_geo_geo = compute_reconstruction_metrics(test_field, recon_geo_geo)
    print(f"  RMSE: {metrics_geo_geo['rmse']:.6f}, SSIM: {metrics_geo_geo['ssim']:.6f}")
    print(f"  Alpha used: {cs_geo_geo.alpha:.6f}")
    results["geo_geo"] = metrics_geo_geo

    return results


def print_summary(results: Dict, baseline_rmse: float):
    """Print the final summary."""
    print("\n" + "=" * 70)
    print("Summary Table".center(70))
    print("=" * 70)
    print(f"{'Method':<40} {'RMSE':<12} {'SSIM':<12} {'Improv.'}")
    print("-" * 72)

    methods = [
        ("Std + Std", results["std_std"]),
        ("Std + Geo", results["std_geo"]),
        ("Geo + Std", results["geo_std"]),
        ("Geo + Geo (alpha=0.15)", results["geo_geo"]),
    ]

    for name, metrics in methods:
        improvement = (baseline_rmse - metrics["rmse"]) / baseline_rmse * 100
        print(f"{name:<40} {metrics['rmse']:.6f}  {metrics['ssim']:.6f}  {improvement:+7.2f}%")

    print("-" * 72)

    # Find best
    best_method = min(methods, key=lambda x: x[1]["rmse"])
    best_rmse = best_method[1]["rmse"]
    improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100

    print(f"\nBest method: {best_method[0]}")
    print(f"   RMSE: {best_rmse:.6f}")
    print(f"   Improvement: {improvement:.1f}%")
    print("=" * 70)


def main():
    """Run the experiment."""
    print("\n" + "=" * 70)
    print("GEOMETRY-AWARE TBMD FOR THE BRUGGE DATASET".center(70))
    print("=" * 70)

    # Parameters.
    n_modes = 30
    n_sensors = 25

    print("\nExperiment parameters:")
    print(f"  Tucker rank: {n_modes}")
    print(f"  Sensors: {n_sensors}")

    # 1. Load data.
    data, wells = load_brugge_data()
    spatial_shape = data.shape[:2]

    # 2. Build graph.
    mesh = build_mesh_graph(spatial_shape)

    # 3. Decomposition.
    decomp_results = test_decomposition(data, mesh, n_modes)

    # 4. Sensor placement.
    sensor_results = test_sensor_placement(data, mesh, n_sensors)

    # 5. Reconstruction.
    cs_results = test_reconstruction(data, mesh, sensor_results, decomp_results, n_modes)

    # 6. Summary.
    baseline_rmse = cs_results["std_std"]["rmse"]
    print_summary(cs_results, baseline_rmse)

    print("\nExperiment completed successfully.")
    print("Geometry-aware QR ran on the local Brugge data.")
    print("Alpha=0.15 was evaluated for this experiment.")
    print("Results generated for local review.")


if __name__ == "__main__":
    main()
