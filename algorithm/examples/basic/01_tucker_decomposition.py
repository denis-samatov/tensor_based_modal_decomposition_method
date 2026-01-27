#!/usr/bin/env python3
"""
Tucker Decomposition Example

Базовый пример использования Tucker декомпозиции для сжатия тензорных данных
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure algorithm is in path (2 levels up: basic -> examples -> algorithm)
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
if algorithm_path not in sys.path:
    sys.path.append(algorithm_path)

from TBMD.config import DecompositionConfig
from TBMD.core.decomposition import TuckerDecomposer

print("=" * 60)
print("TBMD - Tucker Decomposition Example")
print("=" * 60)

# 1. Создать синтетические данные
print("\n1. Создание синтетических данных...")
I = 100  # Пространственные точки
J = 3    # Переменные (например, давление, температура, насыщенность)
T = 50   # Временные шаги

# Создать низкоранговые данные
np.random.seed(42)
torch.manual_seed(42)

# Пространственные моды (низкоранговые)
n_true_modes = 5
spatial_basis = torch.randn(I * J, n_true_modes)
temporal_basis = torch.randn(n_true_modes, T)

# Полные данные
data_flat = spatial_basis @ temporal_basis
data = data_flat.reshape(I, J, T)

# Добавить небольшой шум
data += 0.1 * torch.randn_like(data)

print(f"   Данные созданы: {data.shape}")
print(f"   Истинный ранг: {n_true_modes}")
print(f"   Диапазон значений: [{data.min():.2f}, {data.max():.2f}]")

# 2. Tucker декомпозиция
print("\n2. Tucker декомпозиция...")

# Попробуем разные ранги
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

# 3. Выбрать лучший ранг
print("\n3. Анализ результатов...")
best_rank = 10  # Выберем средний для демонстрации
best_result = results[best_rank]

print(f"   Выбранный ранг: {best_rank}")
# Factors is list of tensors. [0] is spatial (N_s x R_s), [1] is temporal (T x R_t) or (R_t x T)? 
# HOSVD typically returns U_mode.
# Check shapes
print(f"   Spatial modes: {best_result.factors[0].shape}")
print(f"   Temporal modes: {best_result.factors[1].shape}")
print(f"   Core tensor: {best_result.cores.shape}")
print(f"   Ошибка реконструкции: {best_result.reconstruction_errors:.4f}")

# 4. Реконструкция
print("\n4. Реконструкция данных...")
reconstructed_flat = best_result.reconstructed_tensors
# Reshape back to 3D
reconstructed = reconstructed_flat.reshape(I, J, T)

# Вычислить метрики
relative_error = torch.norm(data - reconstructed) / torch.norm(data)
compression_ratio = (I * J * T) / (
    best_result.factors[0].numel() +
    best_result.factors[1].numel() +
    best_result.cores.numel()
)

print(f"   Относительная ошибка: {relative_error:.4f}")
print(f"   Коэффициент сжатия: {compression_ratio:.2f}x")
print(f"   Исходный размер: {I * J * T} элементов")
print(f"   Размер после сжатия: {best_result.factors[0].numel() + best_result.factors[1].numel() + best_result.cores.numel()} элементов")

# 5. Визуализация
print("\n5. Создание визуализации...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Tucker Decomposition Results', fontsize=16)
    
    # Временной сигнал в точке 50
    axes[0, 0].plot(data[50, 0, :].numpy(), label='Original', linewidth=2)
    axes[0, 0].plot(reconstructed[50, 0, :].numpy(), label='Reconstructed', 
                   linestyle='--', linewidth=2)
    axes[0, 0].set_title('Temporal Signal (Point 50, Variable 0)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Пространственное поле в момент T=25
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
    
    # Первые пространственные моды
    # factors[0] is (N_space, R_space)
    n_modes_to_show = min(5, best_result.factors[0].shape[1])
    for i in range(n_modes_to_show):
        mode = best_result.factors[0][:, i].reshape(I, J).mean(dim=1)
        axes[1, 0].plot(mode.numpy(), label=f'Mode {i+1}', alpha=0.7)
    axes[1, 0].set_title('Spatial Modes (averaged over variables)')
    axes[1, 0].set_xlabel('Spatial Points')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Временные моды
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
    
    # Ошибка vs Ранг
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
    print("   ✅ Визуализация сохранена: tucker_decomposition_results.png")
    
except Exception as e:
    print(f"   ⚠️  Визуализация пропущена: {e}")

# 6. Дополнительный анализ
print("\n6. Дополнительный анализ...")

# Сингулярные значения (энергия мод)
mode_energies = torch.norm(best_result.factors[0], dim=0)
print(f"   Энергия первых 5 мод: {mode_energies[:5].tolist()}")

# Кумулятивная энергия
cumulative_energy = torch.cumsum(mode_energies ** 2, dim=0)
# Create new tensor to avoid in-place modification error
cumulative_energy = cumulative_energy / cumulative_energy[-1]
print(f"   Первые 5 мод содержат {cumulative_energy[4]:.2%} энергии")

print("\n" + "=" * 60)
print("✅ Tucker Decomposition Example завершен успешно!")
print("=" * 60)
print("\nКлючевые выводы:")
print(f"  • Tucker декомпозиция эффективно сжимает данные")
print(f"  • Коэффициент сжатия: {compression_ratio:.1f}x")
print(f"  • Относительная ошибка: {relative_error:.2%}")
print(f"  • Выбор ранга - компромисс между точностью и сжатием")

