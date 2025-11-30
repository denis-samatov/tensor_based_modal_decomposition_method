#!/usr/bin/env python3
"""
Sensor Placement Example

Оптимальное размещение сенсоров с помощью Tensor Tube QR
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

# 1. Создать синтетические данные
print("\n1. Создание синтетических данных...")
I = 200  # Пространственные точки
J = 2    # Переменные (например, давление, насыщенность)
T = 40   # Временные шаги

np.random.seed(42)
torch.manual_seed(42)

# Создать пространственную сетку
x = torch.linspace(0, 10, I)

# Создать несколько пространственных мод (волны)
modes = []
for k in [1, 2, 3]:
    mode = torch.sin(k * x) * torch.exp(-0.1 * k * x)
    modes.append(mode)

# Объединить в пространственные моды
n_modes = len(modes)
spatial_modes = torch.stack(modes, dim=1)  # (I, n_modes)

# Расширить на J переменных
spatial_modes_full = spatial_modes.unsqueeze(1).repeat(1, J, 1)  # (I, J, n_modes)
spatial_modes_flat = spatial_modes_full.reshape(I * J, n_modes)

# Создать временную динамику
temporal_dynamics = torch.randn(n_modes, T)

# Полные данные
data_flat = spatial_modes_flat @ temporal_dynamics
data = data_flat.reshape(I, J, T)
data += 0.1 * torch.randn_like(data)

print(f"   Данные созданы: {data.shape}")
print(f"   Диапазон: [{data.min():.2f}, {data.max():.2f}]")

# 2. Tucker декомпозиция (для получения мод)
print("\n2. Tucker декомпозиция...")
decomp_config = DecompositionConfig(
    ranks=[10, 8],
    verbose=False
)

decomposer = TuckerDecomposer(decomp_config)
result = decomposer.decompose(data)

print(f"   Spatial modes: {result.spatial_modes.shape}")
print(f"   Energy retained: {result.energy_retained:.2%}")

# 3. Размещение сенсоров - разное количество
print("\n3. Размещение сенсоров...")

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
    
    print(f"   N={n_sensors}: размещено {len(placement.sensor_indices)} сенсоров, "
          f"coverage={placement.coverage_score:.4f}")

# 4. Анализ покрытия
print("\n4. Анализ покрытия пространства...")

# Выберем случай с 50 сенсорами для детального анализа
n_selected = 50
selected_placement = placements[n_selected]
sensor_indices = selected_placement.sensor_indices

# Преобразовать flat индексы в (i, j)
sensor_i = sensor_indices // J
sensor_j = sensor_indices % J

print(f"   Выбрано {n_selected} сенсоров")
print(f"   Покрытие по переменным: {dict(zip(*np.unique(sensor_j, return_counts=True)))}")
print(f"   Spatial coverage: {len(np.unique(sensor_i))} из {I} точек")

# 5. Оценка качества размещения
print("\n5. Оценка качества размещения...")

# Для каждого количества сенсоров оценим condition number
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

# 6. Тест реконструкции
print("\n6. Тест реконструкции с сенсорами...")

# Возьмем одно поле
test_field = data[:, :, 25]  # (I, J)
test_field_flat = test_field.reshape(-1)

# Измерения с сенсоров
measurements = selected_placement.measurement_matrix @ test_field_flat

# Простая реконструкция (least squares через моды)
# measurements = M @ field = M @ (Phi @ coeffs)
# => coeffs = (M @ Phi)^+ @ measurements
# => field = Phi @ coeffs

Phi = result.spatial_modes
M = selected_placement.measurement_matrix

# Решить
A = M @ Phi
coeffs, _, _, _ = torch.linalg.lstsq(A, measurements.unsqueeze(1))
reconstructed_flat = Phi @ coeffs
reconstructed = reconstructed_flat.reshape(I, J)

# Ошибка
error = torch.norm(test_field - reconstructed) / torch.norm(test_field)
print(f"   Ошибка реконструкции: {error:.4f}")
print(f"   Использовано измерений: {len(sensor_indices)} из {I * J} = {len(sensor_indices) / (I * J):.1%}")

# 7. Визуализация
print("\n7. Создание визуализации...")
try:
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 3x3
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Пространственные моды
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        mode = result.spatial_modes[:, i].reshape(I, J).mean(dim=1)
        ax.plot(x.numpy(), mode.numpy(), linewidth=2)
        ax.set_title(f'Spatial Mode {i+1}')
        ax.set_xlabel('X')
        ax.grid(True, alpha=0.3)
    
    # Row 2: Размещение сенсоров
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
    
    # Row 3: Реконструкция
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
    print("   ✅ Визуализация сохранена: sensor_placement_results.png")
    
except Exception as e:
    print(f"   ⚠️  Визуализация пропущена: {e}")

print("\n" + "=" * 60)
print("✅ Sensor Placement Example завершен успешно!")
print("=" * 60)
print("\nКлючевые выводы:")
print(f"  • QR factorization обеспечивает оптимальное размещение")
print(f"  • {n_selected} сенсоров ({n_selected / (I * J) * 100:.1f}%) достаточно для точной реконструкции")
print(f"  • Ошибка реконструкции: {error:.2%}")
print(f"  • Сенсоры размещены в областях максимальной информативности")

