#!/usr/bin/env python3
"""
Sensor Placement Example

Sensor placement with Tensor Tube QR.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from TBMD.config import SensorPlacementConfig, DecompositionConfig
from TBMD.core.decomposition import TuckerDecomposer
from TBMD.core.sensor_placement import TensorTubeQRDecomposition

print("=" * 60)
print("TBMD - Sensor Placement Example")
print("=" * 60)

# 1. Create synthetic data.
print("\n1. Creating synthetic data...")
I = 200  # Spatial points
J = 2    # Variables, for example pressure and saturation
T = 40   # Time steps

np.random.seed(42)
torch.manual_seed(42)

# Create a spatial grid.
x = torch.linspace(0, 10, I)

# Create several spatial modes as waves.
modes = []
for k in [1, 2, 3]:
    mode = torch.sin(k * x) * torch.exp(-0.1 * k * x)
    modes.append(mode)

# Combine spatial modes.
n_modes = len(modes)
spatial_modes = torch.stack(modes, dim=1)  # (I, n_modes)

# Expand to J variables.
spatial_modes_full = spatial_modes.unsqueeze(1).repeat(1, J, 1)  # (I, J, n_modes)
spatial_modes_flat = spatial_modes_full.reshape(I * J, n_modes)

# Create temporal dynamics.
temporal_dynamics = torch.randn(n_modes, T)

# Full data.
data_flat = spatial_modes_flat @ temporal_dynamics
data = data_flat.reshape(I, J, T)
data += 0.1 * torch.randn_like(data)

print(f"   Data created: {data.shape}")
print(f"   Range: [{data.min():.2f}, {data.max():.2f}]")

# 2. Tucker decomposition to obtain modes.
print("\n2. Tucker decomposition...")
decomp_config = DecompositionConfig(
    ranks=[10, 8],
    verbose=False
)

decomposer = TuckerDecomposer(decomp_config)
result = decomposer.decompose(data)

print(f"   Spatial modes: {result.spatial_modes.shape}")
print(f"   Energy retained: {result.energy_retained:.2%}")

# 3. Sensor placement for several sensor counts.
print("\n3. Sensor placement...")

sensor_counts = [10, 20, 50, 100]
placements = {}

for n_sensors in sensor_counts:
    config = SensorPlacementConfig(
        n_sensors=n_sensors,
        verbose=False
    )
    
    placer = TensorTubeQRDecomposition(config)
    placement = placer.place_sensors(result.spatial_modes)
    
    placements[n_sensors] = placement
    
    print(f"   N={n_sensors}: placed {len(placement.sensor_indices)} sensors, "
          f"coverage={placement.coverage_score:.4f}")

# 4. Coverage analysis.
print("\n4. Spatial coverage analysis...")

# Use the 50-sensor case for detailed analysis.
n_selected = 50
selected_placement = placements[n_selected]
sensor_indices = selected_placement.sensor_indices

# Convert flat indices to (i, j).
sensor_i = sensor_indices // J
sensor_j = sensor_indices % J

print(f"   Selected {n_selected} sensors")
print(f"   Variable coverage: {dict(zip(*np.unique(sensor_j, return_counts=True)))}")
print(f"   Spatial coverage: {len(np.unique(sensor_i))} of {I} points")

# 5. Placement quality assessment.
print("\n5. Placement quality assessment...")

# Estimate condition number for each sensor count.
for n_sensors in sensor_counts:
    placement = placements[n_sensors]
    
    # Measurement matrix condition number
    M = placement.measurement_matrix
    try:
        cond = torch.linalg.cond(M @ result.spatial_modes).item()
    except:
        cond = float('inf')
    
    print(f"   N={n_sensors}: "
          f"coverage={placement.coverage_score:.4f}, "
          f"condition={cond:.2e}")

# 6. Reconstruction test.
print("\n6. Reconstruction test with sensors...")

# Take one field.
test_field = data[:, :, 25]  # (I, J)
test_field_flat = test_field.reshape(-1)

# Sensor measurements.
measurements = selected_placement.measurement_matrix @ test_field_flat

# Simple reconstruction through least squares over modes.
# measurements = M @ field = M @ (Phi @ coeffs)
# => coeffs = (M @ Phi)^+ @ measurements
# => field = Phi @ coeffs

Phi = result.spatial_modes
M = selected_placement.measurement_matrix

# Solve.
A = M @ Phi
coeffs, _, _, _ = torch.linalg.lstsq(A, measurements.unsqueeze(1))
reconstructed_flat = Phi @ coeffs
reconstructed = reconstructed_flat.reshape(I, J)

# Error.
error = torch.norm(test_field - reconstructed) / torch.norm(test_field)
print(f"   Reconstruction error: {error:.4f}")
print(f"   Measurements used: {len(sensor_indices)} of {I * J} = {len(sensor_indices) / (I * J):.1%}")

# 7. Visualization.
print("\n7. Creating visualization...")
try:
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 3x3
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: spatial modes.
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        mode = result.spatial_modes[:, i].reshape(I, J).mean(dim=1)
        ax.plot(x.numpy(), mode.numpy(), linewidth=2)
        ax.set_title(f'Spatial Mode {i+1}')
        ax.set_xlabel('X')
        ax.grid(True, alpha=0.3)
    
    # Row 2: sensor placement.
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(x[sensor_i].numpy(), sensor_j.numpy(), c='red', s=50, alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Variable Index')
    ax1.set_title(f'Sensor Placement (N={n_selected})')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, J - 0.5)
    
    # Coverage vs N sensors
    ax2 = fig.add_subplot(gs[1, 1])
    n_list = sorted(sensor_counts)
    coverage_list = [placements[n].coverage_score for n in n_list]
    ax2.plot(n_list, coverage_list, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Sensors')
    ax2.set_ylabel('Coverage Score')
    ax2.set_title('Coverage vs Number of Sensors')
    ax2.grid(True, alpha=0.3)
    
    # Spatial distribution histogram
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(x[sensor_i].numpy(), bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Number of Sensors')
    ax3.set_title('Spatial Distribution of Sensors')
    ax3.grid(True, alpha=0.3)
    
    # Row 3: reconstruction.
    ax4 = fig.add_subplot(gs[2, 0])
    im1 = ax4.imshow(test_field.numpy(), aspect='auto', cmap='viridis')
    ax4.set_title('Original Field')
    ax4.set_xlabel('Variable')
    ax4.set_ylabel('Spatial Points')
    plt.colorbar(im1, ax=ax4)
    
    ax5 = fig.add_subplot(gs[2, 1])
    im2 = ax5.imshow(reconstructed.numpy(), aspect='auto', cmap='viridis')
    ax5.set_title(f'Reconstructed (N={n_selected} sensors)')
    ax5.set_xlabel('Variable')
    ax5.set_ylabel('Spatial Points')
    plt.colorbar(im2, ax=ax5)
    
    ax6 = fig.add_subplot(gs[2, 2])
    error_field = torch.abs(test_field - reconstructed)
    im3 = ax6.imshow(error_field.numpy(), aspect='auto', cmap='Reds')
    ax6.set_title(f'Absolute Error (Rel: {error:.2%})')
    ax6.set_xlabel('Variable')
    ax6.set_ylabel('Spatial Points')
    plt.colorbar(im3, ax=ax6)
    
    plt.suptitle('Tensor-Based Sensor Placement Results', fontsize=16, y=0.995)
    plt.savefig('sensor_placement_results.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved: sensor_placement_results.png")
    
except Exception as e:
    print(f"   Visualization skipped: {e}")

print("\n" + "=" * 60)
print("Sensor Placement Example completed successfully.")
print("=" * 60)
print("\nKey takeaways:")
print("  - QR factorization provides an information-driven placement strategy.")
print(f"  - {n_selected} sensors ({n_selected / (I * J) * 100:.1f}%) are used in this reconstruction test.")
print(f"  - Reconstruction error: {error:.2%}")
print("  - Sensors are placed in high-information regions for this synthetic example.")
