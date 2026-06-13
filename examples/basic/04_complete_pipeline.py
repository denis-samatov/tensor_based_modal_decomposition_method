#!/usr/bin/env python3
"""
Complete TBMD Pipeline Example

Complete pipeline: decomposition -> placement -> reconstruction.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.config import DecompositionConfig, ReconstructionConfig, SensorPlacementConfig
from TBMD.core import TensorCompressiveSensing, TensorTubeQRDecomposition, TuckerDecomposer


def parse_args():
    parser = argparse.ArgumentParser(description="TBMD Complete Pipeline")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of spatial modes")
    parser.add_argument("--n-sensors", type=int, default=40, help="Number of sensors")
    parser.add_argument(
        "--solver",
        type=str,
        default="admm",
        choices=["least_squares", "admm", "ista"],
        help="Reconstruction solver",
    )
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    return parser.parse_args()


def create_synthetic_reservoir_data(I=200, J=3, T=50, seed=42):
    """Create synthetic reservoir data.

    Variables:
    - J=0: pressure
    - J=1: oil saturation
    - J=2: temperature
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Spatial grid.
    x = torch.linspace(0, 1, I)

    # Time grid.
    time = torch.linspace(0, 1, T)

    data = torch.zeros(I, J, T)

    # Pressure: exponential decay with spatial heterogeneity.
    pressure_base = 1.0 - 0.5 * x  # Pressure gradient.
    for ti, t in enumerate(time):
        decay = torch.exp(-0.5 * t)
        data[:, 0, ti] = pressure_base * decay + 0.1 * torch.sin(5 * x) * decay

    # Saturation: wavefront.
    for ti, t in enumerate(time):
        wavefront = torch.sigmoid(10 * (x - 0.5 * t - 0.2))
        data[:, 1, ti] = wavefront + 0.05 * torch.sin(3 * x)

    # Temperature: diffusion.
    temp_center = I // 2
    for ti, t in enumerate(time):
        spread = 0.1 + 0.3 * t
        data[:, 2, ti] = torch.exp(-((x - x[temp_center]) ** 2) / spread)

    # Add realistic noise.
    data += 0.02 * torch.randn_like(data)

    # Normalize each variable.
    for j in range(J):
        data[:, j, :] = (data[:, j, :] - data[:, j, :].mean()) / (data[:, j, :].std() + 1e-8)

    return data


def run_tbmd_pipeline(data, n_modes, n_sensors, solver="admm"):
    """Run the complete TBMD pipeline."""

    results = {}

    # Step 1: Decomposition
    print("\n" + "=" * 60)
    print("Step 1: Tucker Decomposition")
    print("=" * 60)

    decomp_config = DecompositionConfig(
        ranks=[n_modes, n_modes // 2], backend="torch", verbose=True
    )

    decomposer = TuckerDecomposer(decomp_config)
    decomp_result = decomposer.decompose(data)

    results["decomposition"] = decomp_result

    print("Decomposition complete:")
    print(f"   - Spatial modes: {decomp_result.spatial_modes.shape}")
    print(f"   - Temporal modes: {decomp_result.temporal_modes.shape}")
    print(f"   - Core tensor: {decomp_result.core.shape}")
    print(f"   - Energy retained: {decomp_result.energy_retained:.2%}")
    print(f"   - Reconstruction error: {decomp_result.reconstruction_error:.4f}")

    # Step 2: Sensor placement
    print("\n" + "=" * 60)
    print("Step 2: Sensor Placement")
    print("=" * 60)

    sensor_config = SensorPlacementConfig(n_sensors=n_sensors, backend="torch", verbose=True)

    sensor_placer = TensorTubeQRDecomposition(sensor_config)
    sensor_result = sensor_placer.place_sensors(decomp_result.spatial_modes)

    results["sensors"] = sensor_result

    print("Sensor placement complete:")
    print(f"   - Number of sensors: {len(sensor_result.sensor_indices)}")
    print(f"   - Coverage score: {sensor_result.coverage_score:.4f}")
    print(f"   - Measurement matrix: {sensor_result.measurement_matrix.shape}")

    # Step 3: Reconstruction
    print("\n" + "=" * 60)
    print("Step 3: Field Reconstruction")
    print("=" * 60)

    recon_config = ReconstructionConfig(
        solver=solver, max_iterations=100, lambda_reg=0.01, backend="torch", verbose=True
    )

    reconstructor = TensorCompressiveSensing(recon_config)

    # Reconstruct every time step.
    I, J, T = data.shape
    reconstructed_data = torch.zeros_like(data)
    reconstruction_errors = []

    for ti in range(T):
        field = data[:, :, ti]
        measurements = sensor_result.measurement_matrix @ field.reshape(-1)

        recon_result = reconstructor.reconstruct(
            dictionary=decomp_result.spatial_modes,
            measurements=measurements.unsqueeze(1),
            measurement_matrix=sensor_result.measurement_matrix,
        )

        reconstructed = recon_result.reconstructed_field.reshape(I, J)
        reconstructed_data[:, :, ti] = reconstructed

        error = torch.norm(field - reconstructed) / torch.norm(field)
        reconstruction_errors.append(error.item())

    results["reconstruction"] = {
        "data": reconstructed_data,
        "errors": reconstruction_errors,
        "mean_error": np.mean(reconstruction_errors),
        "std_error": np.std(reconstruction_errors),
    }

    print("Reconstruction complete:")
    print(f"   - Mean error: {results['reconstruction']['mean_error']:.4f}")
    print(f"   - Std error: {results['reconstruction']['std_error']:.4f}")
    print(f"   - Min error: {min(reconstruction_errors):.4f}")
    print(f"   - Max error: {max(reconstruction_errors):.4f}")

    # Step 4: Analysis
    print("\n" + "=" * 60)
    print("Step 4: Analysis")
    print("=" * 60)

    compression_ratio = (I * J * T) / (
        decomp_result.spatial_modes.numel()
        + decomp_result.temporal_modes.numel()
        + decomp_result.core.numel()
    )

    sensing_ratio = n_sensors / (I * J)

    print("Compression analysis:")
    print(f"   - Original size: {I * J * T} elements")
    print(
        f"   - Compressed size: {decomp_result.spatial_modes.numel() + decomp_result.temporal_modes.numel() + decomp_result.core.numel()} elements"
    )
    print(f"   - Compression ratio: {compression_ratio:.2f}x")
    print("\nSensing analysis:")
    print(f"   - Sensors needed: {n_sensors} / {I * J} = {sensing_ratio:.2%}")
    print(f"   - Data reduction: {(1 - sensing_ratio) * 100:.1f}%")

    return results


def visualize_results(data, results, args):
    """Create result visualizations."""

    I, J, T = data.shape
    decomp = results["decomposition"]
    sensors = results["sensors"]
    recon = results["reconstruction"]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

    # Row 1: first 3 spatial modes.
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        mode = decomp.spatial_modes[:, i].reshape(I, J).mean(dim=1)
        ax.plot(mode.numpy(), linewidth=2)
        ax.set_title(f"Spatial Mode {i + 1}")
        ax.set_xlabel("Spatial Points")
        ax.grid(True, alpha=0.3)

    # Row 1, Col 4: mode energy.
    ax = fig.add_subplot(gs[0, 3])
    mode_energies = torch.norm(decomp.spatial_modes, dim=0)[:15]
    ax.bar(range(len(mode_energies)), mode_energies.numpy())
    ax.set_title("Mode Energies")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3, axis="y")

    # Row 2: sensor placement by variable.
    sensor_indices = sensors.sensor_indices
    sensor_i = sensor_indices // J
    sensor_j = sensor_indices % J

    for j in range(J):
        ax = fig.add_subplot(gs[1, j])
        var_sensors = sensor_i[sensor_j == j]
        ax.hist(var_sensors.numpy(), bins=20, alpha=0.7, edgecolor="black")
        ax.set_title(f"Sensor Distribution - Variable {j}")
        ax.set_xlabel("Spatial Position")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Row 2, Col 4: coverage score.
    ax = fig.add_subplot(gs[1, 3])
    ax.text(
        0.5,
        0.5,
        f"Coverage Score\n{sensors.coverage_score:.4f}\n\n"
        f"Sensors: {len(sensor_indices)}\n"
        f"Ratio: {len(sensor_indices) / (I * J):.2%}",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.axis("off")

    # Row 3: reconstruction vs original.
    t_sample = T // 2
    for j in range(3):
        ax = fig.add_subplot(gs[2, j])
        ax.plot(data[:, j, t_sample].numpy(), label="Original", linewidth=2)
        ax.plot(
            recon["data"][:, j, t_sample].numpy(),
            label="Reconstructed",
            linestyle="--",
            linewidth=2,
        )
        # Mark sensors.
        var_sensors = sensor_i[sensor_j == j]
        if len(var_sensors) > 0:
            ax.scatter(
                var_sensors.numpy(), data[var_sensors, j, t_sample].numpy(), c="red", s=50, zorder=5
            )
        ax.set_title(f"Variable {j} (T={t_sample})")
        ax.set_xlabel("Spatial Points")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 3, Col 4: reconstruction error over time.
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(recon["errors"], linewidth=2)
    ax.axhline(
        y=recon["mean_error"], color="red", linestyle="--", label=f"Mean: {recon['mean_error']:.4f}"
    )
    ax.fill_between(
        range(T),
        recon["mean_error"] - recon["std_error"],
        recon["mean_error"] + recon["std_error"],
        alpha=0.3,
        color="red",
    )
    ax.set_title("Reconstruction Error Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Relative Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 4: heatmaps for one variable.
    var_idx = 0
    for idx, (data_slice, title) in enumerate(
        [
            (data[:, var_idx, :], "Original (Variable 0)"),
            (recon["data"][:, var_idx, :], "Reconstructed (Variable 0)"),
            (torch.abs(data[:, var_idx, :] - recon["data"][:, var_idx, :]), "Absolute Error"),
            (
                torch.abs(
                    (data[:, var_idx, :] - recon["data"][:, var_idx, :])
                    / (data[:, var_idx, :] + 1e-8)
                ),
                "Relative Error",
            ),
        ]
    ):
        ax = fig.add_subplot(gs[3, idx])
        im = ax.imshow(
            data_slice.numpy(),
            aspect="auto",
            cmap="viridis" if idx < 2 else "Reds",
            interpolation="bilinear",
        )
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Spatial Points")
        plt.colorbar(im, ax=ax)

    plt.suptitle(
        f"TBMD Complete Pipeline Results\n"
        f"Modes={args.n_modes}, Sensors={args.n_sensors}, Solver={args.solver}",
        fontsize=16,
        y=0.995,
    )

    filename = f"tbmd_complete_pipeline_{args.n_modes}modes_{args.n_sensors}sensors.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved: {filename}")


def main():
    args = parse_args()

    print("=" * 60)
    print("TBMD - Complete Pipeline Example")
    print("=" * 60)
    print("Parameters:")
    print(f"  - Spatial modes: {args.n_modes}")
    print(f"  - Sensors: {args.n_sensors}")
    print(f"  - Solver: {args.solver}")

    # Create data.
    print("\nGenerating synthetic reservoir data...")
    data = create_synthetic_reservoir_data()
    I, J, T = data.shape
    print(f"Data generated: {data.shape}")
    print(f"   - Spatial points: {I}")
    print(f"   - Variables: {J} (pressure, oil saturation, temperature)")
    print(f"   - Time steps: {T}")

    # Run pipeline.
    results = run_tbmd_pipeline(data, args.n_modes, args.n_sensors, args.solver)

    # Visualization.
    if args.visualize:
        print("\nCreating visualization...")
        visualize_results(data, results, args)

    print("\n" + "=" * 60)
    print("TBMD Complete Pipeline completed successfully.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
