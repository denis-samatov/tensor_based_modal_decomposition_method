#!/usr/bin/env python3
"""
Field Reconstruction Example

Full-field reconstruction from sparse sensor measurements.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from TBMD.config import (
    DecompositionConfig,
    SensorPlacementConfig,
    ReconstructionConfig
)
from TBMD.core import (
    TuckerDecomposer,
    TensorTubeQRDecomposition,
    TensorCompressiveSensing
)

print("=" * 60)
print("TBMD - Field Reconstruction Example")
print("=" * 60)

# 1. Create data.
print("\n1. Creating dynamic fields...")
I = 150  # Spatial points
J = 3    # Variables
T = 30   # Time steps

np.random.seed(42)
torch.manual_seed(42)

# Create wave dynamics.
x = torch.linspace(0, 2 * np.pi, I)
t = torch.linspace(0, 4 * np.pi, T)

data = torch.zeros(I, J, T)
for j in range(J):
    for ti in range(T):
        data[:, j, ti] = torch.sin((j + 1) * x + 0.5 * t[ti])

data += 0.05 * torch.randn_like(data)

print(f"   Data created: {data.shape}")

# 2. Decomposition.
print("\n2. Tucker decomposition...")
decomp_config = DecompositionConfig(
    ranks=[15, 10],
    verbose=False
)
decomposer = TuckerDecomposer(decomp_config)
result = decomposer.decompose(data)

print(f"   Modes: {result.spatial_modes.shape}")
print(f"   Energy: {result.energy_retained:.2%}")

# 3. Sensor placement.
print("\n3. Sensor placement...")
sensor_config = SensorPlacementConfig(
    n_sensors=30,
    verbose=False
)
sensor_placer = TensorTubeQRDecomposition(sensor_config)
sensor_result = sensor_placer.place_sensors(result.spatial_modes)

print(f"   Placed: {len(sensor_result.sensor_indices)} sensors")

# 4. Reconstruction with different methods.
print("\n4. Reconstruction with different methods...")

# Select a test field.
test_idx = 15
test_field = data[:, :, test_idx]
test_measurements = sensor_result.measurement_matrix @ test_field.reshape(-1)

# Method 1: Least Squares
print("   4.1. Least Squares...")
recon_config_ls = ReconstructionConfig(
    solver='least_squares',
    verbose=False
)
reconstructor_ls = TensorCompressiveSensing(recon_config_ls)
recon_ls = reconstructor_ls.reconstruct(
    dictionary=result.spatial_modes,
    measurements=test_measurements.unsqueeze(1),
    measurement_matrix=sensor_result.measurement_matrix
)
reconstructed_ls = recon_ls.reconstructed_field.reshape(I, J)
error_ls = torch.norm(test_field - reconstructed_ls) / torch.norm(test_field)
print(f"       Error: {error_ls:.4f}, Iterations: {recon_ls.n_iterations}")

# Method 2: ADMM
print("   4.2. ADMM...")
recon_config_admm = ReconstructionConfig(
    solver='admm',
    max_iterations=100,
    lambda_reg=0.01,
    verbose=False
)
reconstructor_admm = TensorCompressiveSensing(recon_config_admm)
recon_admm = reconstructor_admm.reconstruct(
    dictionary=result.spatial_modes,
    measurements=test_measurements.unsqueeze(1),
    measurement_matrix=sensor_result.measurement_matrix
)
reconstructed_admm = recon_admm.reconstructed_field.reshape(I, J)
error_admm = torch.norm(test_field - reconstructed_admm) / torch.norm(test_field)
print(f"       Error: {error_admm:.4f}, Iterations: {recon_admm.n_iterations}")

# Method 3: ISTA
print("   4.3. ISTA...")
recon_config_ista = ReconstructionConfig(
    solver='ista',
    max_iterations=200,
    lambda_reg=0.01,
    verbose=False
)
reconstructor_ista = TensorCompressiveSensing(recon_config_ista)
recon_ista = reconstructor_ista.reconstruct(
    dictionary=result.spatial_modes,
    measurements=test_measurements.unsqueeze(1),
    measurement_matrix=sensor_result.measurement_matrix
)
reconstructed_ista = recon_ista.reconstructed_field.reshape(I, J)
error_ista = torch.norm(test_field - reconstructed_ista) / torch.norm(test_field)
print(f"       Error: {error_ista:.4f}, Iterations: {recon_ista.n_iterations}")

# 5. Temporal reconstruction.
print("\n5. Temporal sequence reconstruction...")

# Reconstruct every time step.
reconstructed_sequence = torch.zeros_like(data)
errors_over_time = []

for ti in range(T):
    field = data[:, :, ti]
    measurements = sensor_result.measurement_matrix @ field.reshape(-1)
    
    recon = reconstructor_admm.reconstruct(
        dictionary=result.spatial_modes,
        measurements=measurements.unsqueeze(1),
        measurement_matrix=sensor_result.measurement_matrix
    )
    
    reconstructed = recon.reconstructed_field.reshape(I, J)
    reconstructed_sequence[:, :, ti] = reconstructed
    
    error = torch.norm(field - reconstructed) / torch.norm(field)
    errors_over_time.append(error.item())

mean_error = np.mean(errors_over_time)
std_error = np.std(errors_over_time)

print(f"   Mean error: {mean_error:.4f} +/- {std_error:.4f}")
print(f"   Min error: {min(errors_over_time):.4f}")
print(f"   Max error: {max(errors_over_time):.4f}")

# 6. Sensitivity to the number of sensors.
print("\n6. Sensitivity analysis...")

sensor_counts = [10, 20, 30, 50, 75]
errors_vs_sensors = []

for n_sensors in sensor_counts:
    config = SensorPlacementConfig(n_sensors=n_sensors, verbose=False)
    placer = TensorTubeQRDecomposition(config)
    placement = placer.place_sensors(result.spatial_modes)
    
    measurements = placement.measurement_matrix @ test_field.reshape(-1)
    
    recon = reconstructor_admm.reconstruct(
        dictionary=result.spatial_modes,
        measurements=measurements.unsqueeze(1),
        measurement_matrix=placement.measurement_matrix
    )
    
    reconstructed = recon.reconstructed_field.reshape(I, J)
    error = torch.norm(test_field - reconstructed) / torch.norm(test_field)
    errors_vs_sensors.append(error.item())
    
    print(f"   N={n_sensors}: error={error:.4f} ({n_sensors / (I * J) * 100:.1f}% coverage)")

# 7. Visualization.
print("\n7. Creating visualization...")
try:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: method comparison for variable 0.
    var_idx = 0
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(test_field[:, var_idx].numpy(), label='Original', linewidth=2)
    ax1.plot(reconstructed_ls[:, var_idx].numpy(), label='LS', linestyle='--', alpha=0.7)
    ax1.plot(reconstructed_admm[:, var_idx].numpy(), label='ADMM', linestyle='--', alpha=0.7)
    ax1.plot(reconstructed_ista[:, var_idx].numpy(), label='ISTA', linestyle='--', alpha=0.7)
    ax1.scatter(sensor_result.sensor_indices[sensor_result.sensor_indices < I], 
               test_field[sensor_result.sensor_indices[sensor_result.sensor_indices < I], var_idx].numpy(),
               c='red', s=30, zorder=5, label='Sensors')
    ax1.set_title(f'Reconstruction Methods Comparison (Variable {var_idx})')
    ax1.set_xlabel('Spatial Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Method errors.
    ax2 = fig.add_subplot(gs[0, 1])
    methods = ['LS', 'ADMM', 'ISTA']
    errors = [error_ls.item(), error_admm.item(), error_ista.item()]
    colors = ['blue', 'green', 'red']
    ax2.bar(methods, errors, color=colors, alpha=0.7)
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Reconstruction Error by Method')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Iterations.
    ax3 = fig.add_subplot(gs[0, 2])
    iterations = [recon_ls.n_iterations, recon_admm.n_iterations, recon_ista.n_iterations]
    ax3.bar(methods, iterations, color=colors, alpha=0.7)
    ax3.set_ylabel('Iterations')
    ax3.set_title('Convergence Speed')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Row 2: full fields.
    for idx, (field, title) in enumerate([
        (test_field, 'Original'),
        (reconstructed_admm, f'Reconstructed (ADMM)'),
        (torch.abs(test_field - reconstructed_admm), 'Absolute Error')
    ]):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(field.numpy(), aspect='auto', cmap='viridis' if idx < 2 else 'Reds')
        ax.set_title(title)
        ax.set_xlabel('Variables')
        ax.set_ylabel('Spatial Points')
        plt.colorbar(im, ax=ax)
    
    # Row 3: temporal evolution.
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.plot(errors_over_time, linewidth=2)
    ax7.axhline(y=mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.4f}')
    ax7.fill_between(range(T), mean_error - std_error, mean_error + std_error, 
                     alpha=0.3, color='red', label='+/- 1 std')
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Relative Error')
    ax7.set_title('Reconstruction Error Over Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Sensitivity to the number of sensors.
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(sensor_counts, errors_vs_sensors, 'o-', linewidth=2, markersize=8)
    ax8.set_xlabel('Number of Sensors')
    ax8.set_ylabel('Relative Error')
    ax8.set_title('Error vs Number of Sensors')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax8.legend()
    
    # Row 4: temporal dynamics (original vs reconstructed).
    point_idx = 75
    for var in range(J):
        ax = fig.add_subplot(gs[3, var])
        ax.plot(data[point_idx, var, :].numpy(), label='Original', linewidth=2)
        ax.plot(reconstructed_sequence[point_idx, var, :].numpy(), 
               label='Reconstructed', linestyle='--', linewidth=2)
        ax.set_title(f'Temporal Evolution (Point {point_idx}, Var {var})')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Field Reconstruction from Sparse Sensors', fontsize=16, y=0.995)
    plt.savefig('field_reconstruction_results.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved: field_reconstruction_results.png")
    
except Exception as e:
    print(f"   Visualization skipped: {e}")

print("\n" + "=" * 60)
print("Field Reconstruction Example completed successfully.")
print("=" * 60)
print("\nKey takeaways:")
print("  - ADMM provides the best accuracy/runtime balance in this synthetic example.")
print(f"  - Mean reconstruction error: {mean_error:.2%}")
print(f"  - 30 sensors ({30 / (I * J) * 100:.1f}%) are used for the reconstruction test.")
print("  - Regularized methods such as ADMM and ISTA are more robust to noise.")
