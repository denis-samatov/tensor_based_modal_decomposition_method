#!/usr/bin/env python3
"""
Tucker Decomposition Example

Basic example of Tucker decomposition for tensor data compression.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from TBMD.config import DecompositionConfig
from TBMD.core.decomposition import TuckerDecomposer

print("=" * 60)
print("TBMD - Tucker Decomposition Example")
print("=" * 60)

# 1. Create synthetic data.
print("\n1. Creating synthetic data...")
I = 100  # Spatial points
J = 3    # Variables, for example pressure, temperature, saturation
T = 50   # Time steps

# Create low-rank data.
np.random.seed(42)
torch.manual_seed(42)

# Low-rank spatial modes.
n_true_modes = 5
spatial_basis = torch.randn(I * J, n_true_modes)
temporal_basis = torch.randn(n_true_modes, T)

# Full data.
data_flat = spatial_basis @ temporal_basis
data = data_flat.reshape(I, J, T)

# Add small noise.
data += 0.1 * torch.randn_like(data)

print(f"   Data created: {data.shape}")
print(f"   True rank: {n_true_modes}")
print(f"   Value range: [{data.min():.2f}, {data.max():.2f}]")

# 2. Tucker decomposition.
print("\n2. Tucker decomposition...")

# Test several ranks.
ranks_to_test = [3, 5, 10, 20]

# Flatten data for decomposition: (Spatial, Temporal)
data_reshaped = data.reshape(-1, T)

results = {}
for rank in ranks_to_test:
    config = DecompositionConfig(
        ranks=[rank, int(rank / 2)],  # [spatial_rank, temporal_rank]
        backend='torch',
        verbose=False
    )
    
    # Initialize implementation with tensor and config
    decomposer = TuckerDecomposer(tensors=data_reshaped, config=config)
    decomposer.decompose()
    # Compute reconstruction error
    decomposer.reconstruct()
    
    # Store decomposer object itself as result
    results[rank] = decomposer
    
    print(f"   Rank={rank}: "
          f"error={decomposer.reconstruction_errors:.4f}")

# 3. Select a representative rank.
print("\n3. Result analysis...")
best_rank = 10  # Use a middle value for the demonstration.
best_result = results[best_rank]

print(f"   Selected rank: {best_rank}")
# Factors is list of tensors. [0] is spatial (N_s x R_s), [1] is temporal (T x R_t) or (R_t x T)? 
# HOSVD typically returns U_mode.
# Check shapes
print(f"   Spatial modes: {best_result.factors[0].shape}")
print(f"   Temporal modes: {best_result.factors[1].shape}")
print(f"   Core tensor: {best_result.cores.shape}")
print(f"   Reconstruction error: {best_result.reconstruction_errors:.4f}")

# 4. Reconstruction.
print("\n4. Reconstructing data...")
reconstructed_flat = best_result.reconstructed_tensors
# Reshape back to 3D
reconstructed = reconstructed_flat.reshape(I, J, T)

# Compute metrics.
relative_error = torch.norm(data - reconstructed) / torch.norm(data)
compression_ratio = (I * J * T) / (
    best_result.factors[0].numel() +
    best_result.factors[1].numel() +
    best_result.cores.numel()
)

print(f"   Relative error: {relative_error:.4f}")
print(f"   Compression ratio: {compression_ratio:.2f}x")
print(f"   Original size: {I * J * T} elements")
print(f"   Compressed size: {best_result.factors[0].numel() + best_result.factors[1].numel() + best_result.cores.numel()} elements")

# 5. Visualization.
print("\n5. Creating visualization...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Tucker Decomposition Results', fontsize=16)
    
    # Temporal signal at point 50.
    axes[0, 0].plot(data[50, 0, :].numpy(), label='Original', linewidth=2)
    axes[0, 0].plot(reconstructed[50, 0, :].numpy(), label='Reconstructed', 
                   linestyle='--', linewidth=2)
    axes[0, 0].set_title('Temporal Signal (Point 50, Variable 0)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spatial field at T=25.
    t_idx = 25
    im1 = axes[0, 1].imshow(data[:, :, t_idx].numpy(), aspect='auto', cmap='viridis')
    axes[0, 1].set_title(f'Original Field (T={t_idx})')
    axes[0, 1].set_xlabel('Variables')
    axes[0, 1].set_ylabel('Spatial Points')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(reconstructed[:, :, t_idx].numpy(), aspect='auto', cmap='viridis')
    axes[0, 2].set_title(f'Reconstructed Field (T={t_idx})')
    axes[0, 2].set_xlabel('Variables')
    axes[0, 2].set_ylabel('Spatial Points')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # First spatial modes.
    # factors[0] is (N_space, R_space)
    n_modes_to_show = min(5, best_result.factors[0].shape[1])
    for i in range(n_modes_to_show):
        mode = best_result.factors[0][:, i].reshape(I, J).mean(dim=1)
        axes[1, 0].plot(mode.numpy(), label=f'Mode {i+1}', alpha=0.7)
    axes[1, 0].set_title('Spatial Modes (averaged over variables)')
    axes[1, 0].set_xlabel('Spatial Points')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Temporal modes.
    # factors[1] is (T, R_temp) usually for Tucker in tensorly? Or (R_temp, T)?
    # Tensorly tucker returns factors as (dim_size, rank)
    for i in range(n_modes_to_show):
        # Taking column if (T, R)
        if best_result.factors[1].shape[0] == T:
             axes[1, 1].plot(best_result.factors[1][:, i].numpy(), 
                        label=f'Mode {i+1}', alpha=0.7)
        else:
             axes[1, 1].plot(best_result.factors[1][i, :].numpy(), 
                        label=f'Mode {i+1}', alpha=0.7)
        
    axes[1, 1].set_title('Temporal Modes')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error vs rank.
    ranks_list = sorted(results.keys())
    errors = [results[r].reconstruction_errors for r in ranks_list]
    # Energy retained not directly available in new API, skipping
    # energies = [results[r].energy_retained for r in ranks_list]
    
    ax1 = axes[1, 2]
    ax1.plot(ranks_list, errors, 'o-', color='red', linewidth=2, label='Error')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Reconstruction Error', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)
    
    # ax2 = ax1.twinx()
    # ax2.plot(ranks_list, energies, 's-', color='blue', linewidth=2, label='Energy')
    # ax2.set_ylabel('Energy Retained', color='blue')
    # ax2.tick_params(axis='y', labelcolor='blue')
    
    axes[1, 2].set_title('Error vs Rank')
    
    plt.tight_layout()
    plt.savefig('tucker_decomposition_results.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved: tucker_decomposition_results.png")
    
except Exception as e:
    print(f"   Visualization skipped: {e}")

# 6. Additional analysis.
print("\n6. Additional analysis...")

# Singular values as mode energy proxy.
mode_energies = torch.norm(best_result.factors[0], dim=0)
print(f"   First 5 mode energies: {mode_energies[:5].tolist()}")

# Cumulative energy.
cumulative_energy = torch.cumsum(mode_energies ** 2, dim=0)
# Create new tensor to avoid in-place modification error
cumulative_energy = cumulative_energy / cumulative_energy[-1]
print(f"   First 5 modes contain {cumulative_energy[4]:.2%} of energy")

print("\n" + "=" * 60)
print("Tucker Decomposition Example completed successfully.")
print("=" * 60)
print("\nKey takeaways:")
print("  - Tucker decomposition compresses this synthetic low-rank dataset.")
print(f"  - Compression ratio: {compression_ratio:.1f}x")
print(f"  - Relative error: {relative_error:.2%}")
print("  - Rank selection trades off reconstruction accuracy and compression.")
