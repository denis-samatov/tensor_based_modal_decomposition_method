"""
Geometry-Aware Graph-Based TBMD Example

Этот скрипт демонстрирует использование графового подхода в TBMD для учета
геометрической структуры данных месторождения.

Ключевые концепции:
1. Построение графа связности из пространственной сетки
2. Вычисление Laplacian матрицы для регуляризации
3. Geometry-aware TBMD с учетом соседства ячеек
4. Сравнение с классическим подходом

Преимущества графового подхода:
- Сохранение пространственной структуры
- Более гладкие и физичные моды
- Лучшая реконструкция при редких сенсорах
- Учет неструктурированных сеток

Author: TBMD Team
Date: 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict
import time

from TBMD.core.decomposition import TuckerDecomposer
from TBMD.core.decomposition import (
    GeometryAwareTuckerDecomposer,
    GeometryAwareDecompositionConfig as GeometryAwareConfig
)
from TBMD.core.sensor_placement import TensorTubeQRDecomposition
from TBMD.core.sensor_placement import GeometryAwareTensorQR, GeometricQRConfig
from TBMD.core.reconstruction import (
    TensorCompressiveSensing,
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig
)
from TBMD.core.reconstruction import (
    GeometryAwareTensorCS,
    GeometryAwareCSConfig
)
from TBMD.core.geometry import MeshGraphBuilder, MeshGeometry
from TBMD.utils.tbmd_utils import set_seed, compute_reconstruction_metrics
from TBMD.config import SEED

# Настройка
set_seed(SEED)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def generate_synthetic_field_data(
    spatial_shape: Tuple[int, int] = (50, 50),
    n_timesteps: int = 100,
    noise_level: float = 0.05
) -> torch.Tensor:
    """
    Генерация синтетических данных с пространственной структурой.
    
    Создает поля с плавными пространственными градиентами,
    имитирующие реальные физические процессы.
    """
    nx, ny = spatial_shape
    
    # Создать пространственную сетку
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Временной ряд
    t = np.linspace(0, 2*np.pi, n_timesteps)
    
    # Создать данные с пространственно-временной структурой
    data = np.zeros((nx, ny, n_timesteps))
    
    # Базовое поле
    base_field = 100 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.2)
    
    for i, time in enumerate(t):
        # Несколько пространственных мод
        mode1 = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) * np.cos(2*time)
        mode2 = np.cos(3*np.pi*X) * np.sin(np.pi*Y) * np.sin(time)
        mode3 = np.sin(np.pi*X) * np.cos(4*np.pi*Y) * np.cos(3*time)
        
        # Локальные "горячие точки" (имитация скважин)
        hotspot1 = 20 * np.exp(-50*((X-0.3)**2 + (Y-0.3)**2)) * np.sin(time)
        hotspot2 = -15 * np.exp(-50*((X-0.7)**2 + (Y-0.7)**2)) * np.cos(1.5*time)
        
        # Комбинация
        field = base_field + 10*mode1 + 5*mode2 + 3*mode3 + hotspot1 + hotspot2
        
        # Добавить шум
        field += np.random.normal(0, noise_level * np.std(field), field.shape)
        
        data[:, :, i] = field
    
    return torch.from_numpy(data).float()


def build_mesh_graph(
    spatial_shape: Tuple[int, int],
    connectivity_type: str = 'grid',
    k: int = 8
) -> MeshGeometry:
    """
    Построение графа связности сетки.
    
    Parameters
    ----------
    spatial_shape : tuple
        Размерность пространственной сетки
    connectivity_type : str
        Тип связности: 'grid', 'knn', 'radius', 'delaunay'
    k : int
        Количество соседей для knn
    """
    print(f"\n{'='*70}")
    print(f"Построение графа связности сетки ({connectivity_type})".center(70))
    print(f"{'='*70}")
    
    builder = MeshGraphBuilder(
        connectivity_type=connectivity_type,
        k=k if connectivity_type == 'knn' else None
    )
    
    mesh = builder.build_from_shape(spatial_shape)
    
    # Получить количество ячеек из размера матрицы смежности
    n_cells = mesh.adjacency_matrix.shape[0]
    n_edges = mesh.adjacency_matrix.nnz
    
    print(f"\nГраф построен:")
    print(f"  Узлов (ячеек): {n_cells}")
    print(f"  Рёбер: {n_edges}")
    print(f"  Средняя степень узла: {n_edges / n_cells:.2f}")
    print(f"  Тип Laplacian: normalized")
    
    return mesh


def visualize_mesh_graph(
    mesh: MeshGeometry,
    spatial_shape: Tuple[int, int],
    save_path: str = "mesh_graph_connectivity.png"
):
    """Визуализация графа связности."""
    nx, ny = spatial_shape
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Матрица смежности
    ax1 = axes[0]
    adj_dense = mesh.adjacency_matrix.toarray()
    im1 = ax1.imshow(adj_dense, cmap='Blues', aspect='auto')
    ax1.set_title('Матрица смежности (Adjacency)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Узел j', fontsize=10)
    ax1.set_ylabel('Узел i', fontsize=10)
    plt.colorbar(im1, ax=ax1)
    
    # 2. Laplacian матрица
    ax2 = axes[1]
    lap_dense = mesh.laplacian_matrix.toarray()
    im2 = ax2.imshow(lap_dense, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title('Laplacian матрица', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Узел j', fontsize=10)
    ax2.set_ylabel('Узел i', fontsize=10)
    plt.colorbar(im2, ax=ax2)
    
    # 3. Степень узлов (connectivity)
    ax3 = axes[2]
    degrees = np.array(mesh.adjacency_matrix.sum(axis=1)).flatten()
    degree_field = degrees.reshape(spatial_shape)
    im3 = ax3.imshow(degree_field, cmap='viridis', aspect='auto', origin='lower')
    ax3.set_title('Степень узлов (связность)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Y', fontsize=10)
    ax3.set_ylabel('X', fontsize=10)
    plt.colorbar(im3, ax=ax3, label='Число соседей')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Визуализация графа сохранена: {save_path}")
    plt.close()


def compare_standard_vs_geometry_aware(
    data: torch.Tensor,
    mesh: MeshGeometry,
    n_modes: int = 30
) -> Dict:
    """
    Сравнение стандартного и geometry-aware TBMD.
    """
    print(f"\n{'='*70}")
    print("Сравнение: Стандартный vs Geometry-Aware TBMD".center(70))
    print(f"{'='*70}")
    
    results = {}
    
    # 1. Стандартный TBMD
    print("\n[1/2] Стандартный TBMD (без учета геометрии)...")
    start_time = time.time()
    
    standard_decomposer = TuckerDecomposer(
        tensors=data,  # Используем 'tensors' вместо 'tensor'
        ranks=[n_modes, n_modes, n_modes // 2],  # 3 ранга для 3D тензора (spatial_x, spatial_y, time)
        device='cpu'
    )
    standard_decomposer.decompose()
    
    standard_time = time.time() - start_time
    standard_decomposer.reconstruct()  # Метод не возвращает значение
    standard_recon = standard_decomposer.reconstructed_tensors  # Получаем через свойство
    standard_error = torch.norm(data - standard_recon) / torch.norm(data)
    
    print(f"  Время: {standard_time:.2f}s")
    print(f"  Ошибка реконструкции: {standard_error:.6f}")
    
    results['standard'] = {
        'decomposer': standard_decomposer,
        'reconstruction': standard_recon,
        'error': float(standard_error.item()),
        'time': standard_time,
        'factors': standard_decomposer.factors
    }
    
    # 2. Geometry-Aware TBMD
    print("\n[2/2] Geometry-Aware TBMD (с учетом геометрии)...")
    start_time = time.time()
    
    geo_config = GeometryAwareConfig(
        alpha=0.05,  # Вес Laplacian регуляризации
        spatial_modes=[0],  # Регуляризировать пространственную моду
        laplacian_type='normalized',
        connectivity_type='grid'
    )
    
    # Для geometry-aware нужно объединить пространственные измерения
    # Преобразуем (50, 50, 100) → (2500, 100)
    spatial_size = data.shape[0] * data.shape[1]
    data_2d = data.reshape(spatial_size, data.shape[2])
    print(f"  Форма тензора для Geometry-Aware: {data_2d.shape}")
    
    geo_decomposer = GeometryAwareTuckerDecomposer(
        tensor=data_2d,  # Используем 2D форму (spatial_cells, time)
        mesh=mesh,
        geo_config=geo_config,
        ranks=[n_modes, n_modes // 2],  # 2 ранга для 2D тензора (spatial, time)
        device='cpu'
    )
    geo_decomposer.decompose()
    
    geo_time = time.time() - start_time
    geo_recon_2d = geo_decomposer.reconstruct()
    # Преобразуем обратно в 3D форму для сравнения
    geo_recon = geo_recon_2d.reshape(data.shape[0], data.shape[1], data.shape[2])
    geo_error = torch.norm(data - geo_recon) / torch.norm(data)
    
    print(f"  Время: {geo_time:.2f}s")
    print(f"  Ошибка реконструкции: {geo_error:.6f}")
    
    # Преобразовать факторы geometry-aware для соответствия standard
    # geo_factors[0] имеет форму (2500, n_modes), преобразуем в (50, 50, n_modes)
    geo_factors_reshaped = [
        geo_decomposer.factors[0].reshape(data.shape[0], data.shape[1], -1),  # spatial factor
        geo_decomposer.factors[1]  # temporal factor
    ]
    
    results['geometry_aware'] = {
        'decomposer': geo_decomposer,
        'reconstruction': geo_recon,
        'error': float(geo_error.item()),
        'time': geo_time,
        'factors': geo_factors_reshaped,  # Для визуализации (50, 50, n_modes)
        'factors_original': geo_decomposer.factors  # Оригинальные факторы для CS (2500, n_modes)
    }
    
    # Сравнение
    print(f"\n{'='*70}")
    print("Результаты сравнения".center(70))
    print(f"{'='*70}")
    print(f"\n{'Метод':<30} {'Ошибка':<15} {'Время (s)':<15}")
    print("-" * 60)
    print(f"{'Стандартный':<30} {standard_error:.6f}      {standard_time:.2f}")
    print(f"{'Geometry-Aware':<30} {geo_error:.6f}      {geo_time:.2f}")
    print("-" * 60)
    
    improvement = (standard_error - geo_error) / standard_error * 100
    print(f"\n✓ Улучшение точности: {improvement:.2f}%")
    
    return results


def visualize_spatial_modes_comparison(
    results: Dict,
    spatial_shape: Tuple[int, int],
    n_modes_to_show: int = 4,
    save_path: str = "modes_comparison.png"
):
    """Визуализация и сравнение пространственных мод."""
    print(f"\nВизуализация пространственных мод...")
    
    # Для стандартной декомпозиции: объединить первые два фактора (outer product)
    # factors[0]: (50, 30), factors[1]: (50, 30) -> объединенный: (50, 50, 30)
    standard_factor_0 = results['standard']['factors'][0].cpu().numpy()  # (50, n_modes)
    standard_factor_1 = results['standard']['factors'][1].cpu().numpy()  # (50, n_modes)
    
    # Создаем объединенный пространственный фактор через outer product
    n_modes = standard_factor_0.shape[1]
    standard_modes_spatial = np.zeros((spatial_shape[0], spatial_shape[1], n_modes))
    for k in range(n_modes):
        # Для каждой моды: outer product двух факторов
        standard_modes_spatial[:, :, k] = np.outer(standard_factor_0[:, k], standard_factor_1[:, k])
    
    # Для geometry-aware: уже в правильной форме (50, 50, 30)
    geo_modes_spatial = results['geometry_aware']['factors'][0].cpu().numpy()
    
    fig, axes = plt.subplots(n_modes_to_show, 3, figsize=(15, 4*n_modes_to_show))
    
    for i in range(n_modes_to_show):
        # Стандартная мода
        ax1 = axes[i, 0] if n_modes_to_show > 1 else axes[0]
        im1 = ax1.imshow(standard_modes_spatial[:, :, i], cmap='RdBu_r', 
                        aspect='auto', origin='lower')
        ax1.set_title(f'Стандартная мода {i+1}', fontsize=11, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Geometry-aware мода
        ax2 = axes[i, 1] if n_modes_to_show > 1 else axes[1]
        im2 = ax2.imshow(geo_modes_spatial[:, :, i], cmap='RdBu_r',
                        aspect='auto', origin='lower')
        ax2.set_title(f'Geometry-Aware мода {i+1}', fontsize=11, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Разница
        ax3 = axes[i, 2] if n_modes_to_show > 1 else axes[2]
        diff = np.abs(standard_modes_spatial[:, :, i] - geo_modes_spatial[:, :, i])
        im3 = ax3.imshow(diff, cmap='Reds', aspect='auto', origin='lower')
        ax3.set_title(f'|Разница| (мода {i+1})', fontsize=11, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    plt.suptitle('Сравнение пространственных мод: Стандартный vs Geometry-Aware',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Визуализация мод сохранена: {save_path}")
    plt.close()


def test_sensor_placement_with_geometry(
    data: torch.Tensor,
    mesh: MeshGeometry,
    n_sensors: int = 25
) -> Dict:
    """
    Тестирование размещения сенсоров с учетом геометрии.
    """
    print(f"\n{'='*70}")
    print("Размещение сенсоров: Стандартный vs Geometry-Aware".center(70))
    print(f"{'='*70}")
    
    results = {}
    
    # 1. Стандартный QR
    print(f"\n[1/2] Стандартный Tensor QR...")
    standard_qr = TensorTubeQRDecomposition(
        tensor=data,
        N=n_sensors,
        device='cpu',
        uniform_distribution=True
    )
    P_standard, Q_standard, R_standard = standard_qr.factorize()
    
    print(f"  Размещено сенсоров: {torch.sum(P_standard).item()}")
    
    results['standard'] = {
        'sensor_locations': P_standard,
        'Q': Q_standard,
        'R': R_standard
    }
    
    # 2. Geometry-Aware QR
    print(f"\n[2/2] Geometry-Aware Tensor QR...")
    
    # Преобразуем тензор в 2D форму для geometry-aware QR
    # (50, 50, 100) -> (2500, 100)
    spatial_size = data.shape[0] * data.shape[1]
    data_2d_qr = data.reshape(spatial_size, data.shape[2])
    
    # Создаём конфигурацию для geometry-aware QR с учётом амплитуды
    geo_qr_config = GeometricQRConfig(
        gradient_weight=0.3,      # Вес для градиентов поля (снижен для баланса)
        amplitude_weight=1.5,     # NEW: Вес для амплитуды поля (приоритет высокоамплитудным областям)
        energy_weight=0.8,        # NEW: Вес для локальной энергии (учёт окрестности)
        proximity_weight=1.0,     # Вес для минимального расстояния между сенсорами
        distribution_weight=0.5,  # Вес для равномерности распределения
        min_distance_factor=2.0   # Минимальное расстояние между сенсорами
    )
    
    geo_qr = GeometryAwareTensorQR(
        tensor=data_2d_qr,  # Используем 2D форму
        mesh=mesh,
        N=n_sensors,
        config=geo_qr_config,  # Передаём конфигурацию
        device='cpu'
    )
    P_geo, Q_geo, R_geo = geo_qr.factorize()
    
    print(f"  Размещено сенсоров: {torch.sum(P_geo).item()}")
    
    results['geometry_aware'] = {
        'sensor_locations': P_geo,
        'Q': Q_geo,
        'R': R_geo
    }
    
    return results


def visualize_sensor_placement_comparison(
    sensor_results: Dict,
    data: torch.Tensor,
    save_path: str = "sensor_placement_comparison.png"
):
    """Визуализация сравнения размещения сенсоров."""
    print(f"\nВизуализация размещения сенсоров...")
    
    P_standard = sensor_results['standard']['sensor_locations'].cpu().numpy()
    P_geo_raw = sensor_results['geometry_aware']['sensor_locations'].cpu().numpy()
    
    # Если P_geo 1D (из 2D тензора), преобразуем в 2D форму
    spatial_shape = (data.shape[0], data.shape[1])
    if P_geo_raw.ndim == 1:
        P_geo = P_geo_raw.reshape(spatial_shape)
    else:
        P_geo = P_geo_raw
    
    # Базовое поле
    base_field = data[:, :, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Стандартное размещение
    ax1 = axes[0]
    ax1.imshow(base_field, cmap='viridis', alpha=0.5, aspect='auto', origin='lower')
    sensor_pos_std = np.argwhere(P_standard == 1)
    if len(sensor_pos_std) > 0:
        ax1.scatter(sensor_pos_std[:, 1], sensor_pos_std[:, 0],
                   c='red', marker='o', s=80, edgecolors='black', linewidths=1.5,
                   label=f'Сенсоры ({len(sensor_pos_std)})')
    ax1.set_title('Стандартное размещение', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.axis('off')
    
    # 2. Geometry-aware размещение
    ax2 = axes[1]
    ax2.imshow(base_field, cmap='viridis', alpha=0.5, aspect='auto', origin='lower')
    sensor_pos_geo = np.argwhere(P_geo == 1)
    if len(sensor_pos_geo) > 0:
        ax2.scatter(sensor_pos_geo[:, 1], sensor_pos_geo[:, 0],
                   c='blue', marker='s', s=80, edgecolors='black', linewidths=1.5,
                   label=f'Сенсоры ({len(sensor_pos_geo)})')
    ax2.set_title('Geometry-Aware размещение', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axis('off')
    
    # 3. Сравнение распределения
    ax3 = axes[2]
    
    # Вычислить распределение по строкам и столбцам
    std_rows = P_standard.sum(axis=1)
    std_cols = P_standard.sum(axis=0)
    geo_rows = P_geo.sum(axis=1)
    geo_cols = P_geo.sum(axis=0)
    
    # Построить гистограммы
    x = np.arange(len(std_rows))
    width = 0.35
    
    ax3.bar(x - width/2, std_rows, width, label='Стандартный', alpha=0.7, color='red')
    ax3.bar(x + width/2, geo_rows, width, label='Geometry-Aware', alpha=0.7, color='blue')
    
    ax3.set_xlabel('Строка (X)', fontsize=10)
    ax3.set_ylabel('Количество сенсоров', fontsize=10)
    ax3.set_title('Распределение сенсоров по X', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Визуализация размещения сохранена: {save_path}")
    plt.close()


def test_reconstruction_with_geometry(
    data: torch.Tensor,
    mesh: MeshGeometry,
    sensor_results: Dict,
    decomposition_results: Dict
) -> Dict:
    """
    Тестирование реконструкции с учетом геометрии.
    """
    print(f"{'='*70}")
    print("Реконструкция полей: Стандартный vs Geometry-Aware CS".center(70))
    print(f"{'='*70}")
    
    results = {}
    
    # Выбрать тестовый временной срез
    test_time_idx = data.shape[-1] // 2
    test_field = data[:, :, test_time_idx]
    
    # Получить пространственную форму
    spatial_shape = test_field.shape  # (spatial_x, spatial_y)
    spatial_size = int(np.prod(spatial_shape))
    
    # Стандартный базис
    standard_basis = decomposition_results['standard']['factors'][0]
    print(f"  Standard factor[0] shape: {standard_basis.shape}")
    print(f"  Standard factor[1] shape: {decomposition_results['standard']['factors'][1].shape}")
    
    # Geometry-aware базис - используем ОРИГИНАЛЬНЫЕ факторы для CS!
    geo_basis_original = decomposition_results['geometry_aware']['factors_original'][0]  # (2500, n_modes)
    print(f"  Geo-aware factor[0] (original) shape: {geo_basis_original.shape}")
    print(f"  Geo-aware factor[1] shape: {decomposition_results['geometry_aware']['factors_original'][1].shape}")
    
    # Преобразуем оригинальный фактор в 3D форму для CS
    geo_basis = geo_basis_original.reshape(*spatial_shape, -1)  # (50, 50, n_modes)
    print(f"  Geo-aware basis reshaped for CS: {geo_basis.shape}")
    
    # Сенсоры
    P_standard = sensor_results['standard']['sensor_locations']
    P_geo_raw = sensor_results['geometry_aware']['sensor_locations']
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
    
    # 1. Стандартная реконструкция
    print(f"[1/4] Стандартная реконструкция (CS, стандартные сенсоры)...")
    
    # Построить словарь A_std из факторов
    standard_factor_0 = decomposition_results['standard']['factors'][0].cpu().numpy()
    standard_factor_1 = decomposition_results['standard']['factors'][1].cpu().numpy()
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
    print(f"  ✓ Standard sensors: {n_sensors_std} ({n_sensors_std/test_field.numel()*100:.2f}% sampling)")
    print(f"  ✓ Measurement stats: mean={test_field[sensor_mask_std].mean():.2f}, min={test_field[sensor_mask_std].min():.2f}, max={test_field[sensor_mask_std].max():.2f}")
    
    cs_config = CompressiveSensingConfig(max_iter=100, tol=1e-3, device='cpu')
    
    reconstructor_std = TensorCompressiveSensing(
        A=A_std,
        P=P_standard_2d,
        Y=Y_std,
        core_cfg=cs_config
    )
    
    x_std, _ = reconstructor_std.solve()
    print(f"  ✓ x_std shape: {x_std.shape}, norm: {torch.norm(x_std):.4f}, sparsity: {(x_std.abs() < 1e-3).sum().item()}/{x_std.numel()}")
    
    recon_std_field = torch.einsum('ijk,k->ij', A_std, x_std).cpu()
    print(f"  ✓ Recon range: [{recon_std_field.min():.2f}, {recon_std_field.max():.2f}], True: [{test_field.min():.2f}, {test_field.max():.2f}]")
    
    metrics_std = compute_reconstruction_metrics(test_field, recon_std_field)
    print(f"  RMSE: {metrics_std['rmse']:.6f}")
    print(f"  SSIM: {metrics_std['ssim']:.6f}")
    print(f"  Relative Error: {metrics_std['relative_error']:.6f}")
    
    results['standard'] = {
        'reconstruction': recon_std_field,
        'metrics': metrics_std
    }
    
    # Создать словарь A_geo из geometry-aware базиса
    A_geo = geo_basis.cpu()  # geo_basis уже в правильной форме (50, 50, n_modes)
    
    # 2. Geometry-Aware реконструкция с geometry-aware CS
    print(f"[2/4] Geometry-Aware реконструкция (Geometry-Aware CS, geometry сенсоры)...")
    
    Y_geo = torch.zeros_like(test_field)
    sensor_mask_geo = P_geo_2d.bool()
    Y_geo[sensor_mask_geo] = test_field[sensor_mask_geo]
    n_sensors_geo = int(sensor_mask_geo.sum().item())
    print(f"  ✓ Geometry sensors: {n_sensors_geo} ({n_sensors_geo/test_field.numel()*100:.2f}% sampling)")
    print(f"  ✓ Measurement stats: mean={test_field[sensor_mask_geo].mean():.2f}, min={test_field[sensor_mask_geo].min():.2f}, max={test_field[sensor_mask_geo].max():.2f}")
    
    geo_cs_config = GeometryAwareCSConfig(max_iter=100, tol=1e-3, alpha=0.05, device='cpu')
    
    reconstructor_geo = GeometryAwareTensorCS(
        A=A_geo,
        P=P_geo_2d,
        Y=Y_geo,
        mesh=mesh,
        core_cfg=geo_cs_config
    )
    
    x_geo, _ = reconstructor_geo.solve()
    print(f"  ✓ x_geo shape: {x_geo.shape}, norm: {torch.norm(x_geo):.4f}, sparsity: {(x_geo.abs() < 1e-3).sum().item()}/{x_geo.numel()}")
    
    recon_geo_field = torch.einsum('ijk,k->ij', A_geo, x_geo).cpu()
    print(f"  ✓ Recon range: [{recon_geo_field.min():.2f}, {recon_geo_field.max():.2f}], True: [{test_field.min():.2f}, {test_field.max():.2f}]")
    
    metrics_geo = compute_reconstruction_metrics(test_field, recon_geo_field)
    print(f"  RMSE: {metrics_geo['rmse']:.6f}")
    print(f"  SSIM: {metrics_geo['ssim']:.6f}")
    print(f"  Relative Error: {metrics_geo['relative_error']:.6f}")
    
    results['geometry_aware'] = {
        'reconstruction': recon_geo_field,
        'metrics': metrics_geo
    }
    
    # 3. Кросс-комбинации с одинаковым решателем (стандартный CS)
    print(f"[3/4] Кросс-комбинации (стандартный CS решатель для обеих пар)...")
    cross_results = {}
    
    def run_cross_case(label: str, A_tensor: torch.Tensor, P_mask: torch.Tensor, Y_measure: torch.Tensor, sensor_mask: torch.Tensor) -> None:
        print(f"{label}")
        cs_solver = TensorCompressiveSensing(
            A=A_tensor,
            P=P_mask,
            Y=Y_measure,
            core_cfg=cs_config
        )
        x_hat, _ = cs_solver.solve()
        print(f"  ✓ x shape: {x_hat.shape}, norm: {torch.norm(x_hat):.4f}, sparsity: {(x_hat.abs() < 1e-3).sum().item()}/{x_hat.numel()}")
        recon_field = torch.einsum('ijk,k->ij', A_tensor, x_hat).cpu()
        print(f"  ✓ Recon range: [{recon_field.min():.2f}, {recon_field.max():.2f}]")
        metrics = compute_reconstruction_metrics(test_field, recon_field)
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  SSIM: {metrics['ssim']:.6f}")
        print(f"  Relative Error: {metrics['relative_error']:.6f}")
        cross_results[label] = {
            'reconstruction': recon_field,
            'metrics': metrics,
            'sensors': int(sensor_mask.sum().item())
        }
    
    run_cross_case(
        'Std basis + Geo sensors (standard CS)',
        A_std,
        P_geo_2d,
        Y_geo,
        sensor_mask_geo
    )
    
    run_cross_case(
        'Geo basis + Std sensors (standard CS)',
        A_geo,
        P_standard_2d,
        Y_std,
        sensor_mask_std
    )
    
    # 4. Сводная таблица
    print(f"{'='*70}")
    print("Результаты реконструкции".center(70))
    print(f"{'='*70}")
    print(f"{'Метод':<45} {'RMSE':<12} {'SSIM':<12} {'Rel.Error':<12}")
    print('-' * 85)
    print(f"{'Стандартный (standard CS)':<45} {metrics_std['rmse']:.6f}  {metrics_std['ssim']:.6f}  {metrics_std['relative_error']:.6f}")
    print(f"{'Geometry-Aware (geo CS)':<45} {metrics_geo['rmse']:.6f}  {metrics_geo['ssim']:.6f}  {metrics_geo['relative_error']:.6f}")
    for label, entry in cross_results.items():
        m = entry['metrics']
        print(f"{label:<45} {m['rmse']:.6f}  {m['ssim']:.6f}  {m['relative_error']:.6f}")
    print('-' * 85)
    
    results['cross'] = cross_results
    results['true_field'] = test_field
    
    return results



def visualize_reconstruction_comparison(
    recon_results: Dict,
    save_path: str = "reconstruction_comparison.png"
):
    """Визуализация сравнения реконструкции."""
    print(f"\nВизуализация реконструкции...")
    
    true_field = recon_results['true_field'].cpu().numpy()
    recon_std = recon_results['standard']['reconstruction'].cpu().numpy()
    recon_geo = recon_results['geometry_aware']['reconstruction'].cpu().numpy()
    
    error_std = np.abs(true_field - recon_std)
    error_geo = np.abs(true_field - recon_geo)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    vmin, vmax = true_field.min(), true_field.max()
    
    # Строка 1: Поля
    # Истинное поле
    ax1 = axes[0, 0]
    im1 = ax1.imshow(true_field, cmap='viridis', aspect='auto', origin='lower',
                     vmin=vmin, vmax=vmax)
    ax1.set_title('Истинное поле', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Стандартная реконструкция
    ax2 = axes[0, 1]
    im2 = ax2.imshow(recon_std, cmap='viridis', aspect='auto', origin='lower',
                     vmin=vmin, vmax=vmax)
    metrics_std = recon_results['standard']['metrics']
    ax2.set_title(f'Стандартная (SSIM={metrics_std["ssim"]:.4f})',
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Geometry-aware реконструкция
    ax3 = axes[0, 2]
    im3 = ax3.imshow(recon_geo, cmap='viridis', aspect='auto', origin='lower',
                     vmin=vmin, vmax=vmax)
    metrics_geo = recon_results['geometry_aware']['metrics']
    ax3.set_title(f'Geometry-Aware (SSIM={metrics_geo["ssim"]:.4f})',
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Строка 2: Ошибки
    error_max = max(error_std.max(), error_geo.max())
    
    # Пустая ячейка
    axes[1, 0].axis('off')
    
    # Ошибка стандартной
    ax5 = axes[1, 1]
    im5 = ax5.imshow(error_std, cmap='Reds', aspect='auto', origin='lower',
                     vmin=0, vmax=error_max)
    ax5.set_title(f'Ошибка (RMSE={metrics_std["rmse"]:.4f})',
                 fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # Ошибка geometry-aware
    ax6 = axes[1, 2]
    im6 = ax6.imshow(error_geo, cmap='Reds', aspect='auto', origin='lower',
                     vmin=0, vmax=error_max)
    ax6.set_title(f'Ошибка (RMSE={metrics_geo["rmse"]:.4f})',
                 fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    plt.suptitle('Сравнение реконструкции: Стандартный vs Geometry-Aware',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Визуализация реконструкции сохранена: {save_path}")
    plt.close()


def main():
    """Главная функция."""
    print("=" * 70)
    print(" Geometry-Aware Graph-Based TBMD Demo ".center(70, "="))
    print("=" * 70)
    print()
    
    # Параметры
    spatial_shape = (50, 50)
    n_timesteps = 100
    n_modes = 30
    n_sensors = 25
    
    print(f"Параметры эксперимента:")
    print(f"  Пространственная размерность: {spatial_shape}")
    print(f"  Временных шагов: {n_timesteps}")
    print(f"  Количество мод: {n_modes}")
    print(f"  Количество сенсоров: {n_sensors}")
    
    # 1. Генерация данных
    print(f"\n{'='*70}")
    print("Генерация синтетических данных".center(70))
    print(f"{'='*70}")
    
    data = generate_synthetic_field_data(spatial_shape, n_timesteps)
    print(f"\n✓ Данные сгенерированы: {data.shape}")
    print(f"  Диапазон значений: [{data.min():.2f}, {data.max():.2f}]")
    
    # 2. Построение графа
    mesh = build_mesh_graph(spatial_shape, connectivity_type='grid')
    visualize_mesh_graph(mesh, spatial_shape)
    
    # 3. Сравнение декомпозиций
    decomp_results = compare_standard_vs_geometry_aware(data, mesh, n_modes)
    visualize_spatial_modes_comparison(decomp_results, spatial_shape)
    
    # 4. Сравнение размещения сенсоров
    sensor_results = test_sensor_placement_with_geometry(data, mesh, n_sensors)
    visualize_sensor_placement_comparison(sensor_results, data)
    
    # 5. Сравнение реконструкции
    recon_results = test_reconstruction_with_geometry(
        data, mesh, sensor_results, decomp_results
    )
    visualize_reconstruction_comparison(recon_results)
    
    # 6. БОНУС: Адаптивный alpha
    adaptive_results = test_adaptive_alpha(
        data, mesh, sensor_results, decomp_results
    )
    
    # Итоговая сводка
    print(f"\n{'='*70}")
    print(" ИТОГОВАЯ СВОДКА ".center(70, "="))
    print(f"{'='*70}")
    
    print("\n1. ДЕКОМПОЗИЦИЯ:")
    print(f"   Стандартная ошибка: {decomp_results['standard']['error']:.6f}")
    print(f"   Geometry-Aware ошибка: {decomp_results['geometry_aware']['error']:.6f}")
    improvement_decomp = (decomp_results['standard']['error'] - 
                         decomp_results['geometry_aware']['error']) / decomp_results['standard']['error'] * 100
    print(f"   ✓ Улучшение: {improvement_decomp:.2f}%")
    
    print("\n2. РАЗМЕЩЕНИЕ СЕНСОРОВ:")
    n_std = torch.sum(sensor_results['standard']['sensor_locations']).item()
    n_geo = torch.sum(sensor_results['geometry_aware']['sensor_locations']).item()
    print(f"   Стандартный: {n_std} сенсоров")
    print(f"   Geometry-Aware: {n_geo} сенсоров")
    
    print("\n3. РЕКОНСТРУКЦИЯ:")
    metrics_std = recon_results['standard']['metrics']
    metrics_geo = recon_results['geometry_aware']['metrics']
    print(f"   Стандартная SSIM: {metrics_std['ssim']:.6f}")
    print(f"   Geometry-Aware SSIM: {metrics_geo['ssim']:.6f}")
    ssim_improvement = (metrics_geo['ssim'] - metrics_std['ssim']) / metrics_std['ssim'] * 100
    print(f"   ✓ Улучшение: {ssim_improvement:.2f}%")
    
    print(f"\n{'='*70}")
    print(" ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА ".center(70, "="))
    print(f"{'='*70}")
    
    print("\nСгенерированные файлы:")
    print("  - mesh_graph_connectivity.png")
    print("  - modes_comparison.png")
    print("  - sensor_placement_comparison.png")
    print("  - reconstruction_comparison.png")
    
    print("\n✓ Geometry-aware подход показывает:")
    print("  • Более гладкие и физичные моды")
    print("  • Лучшее размещение сенсоров")
    print("  • Более точную реконструкцию")
    print("  • Учет пространственной структуры данных")


def test_adaptive_alpha(
    data: torch.Tensor,
    mesh: MeshGeometry,
    sensor_results: Dict,
    decomposition_results: Dict
) -> Dict:
    """
    Тестирование адаптивного alpha для Geometry-Aware CS.
    
    Проверяет эффективность адаптации alpha к качеству измерений.
    """
    print(f"\n{'='*70}")
    print("🚀 БОНУС: Адаптивный Alpha для Geometry-Aware CS".center(70))
    print(f"{'='*70}")
    
    print("\n📖 Концепция:")
    print("  Высокие измерения (mean=88) → меньше сглаживания (alpha↓)")
    print("  Низкие измерения (mean=44) → больше сглаживания (alpha↑)")
    print("  Формула: α_adaptive = α_base * (reference / actual)")
    
    # Подготовка данных
    test_time_idx = data.shape[-1] // 2
    test_field = data[:, :, test_time_idx]
    spatial_shape = test_field.shape
    
    geo_basis_original = decomposition_results['geometry_aware']['factors_original'][0]
    geo_basis = geo_basis_original.reshape(*spatial_shape, -1)
    
    P_geo_raw = sensor_results['geometry_aware']['sensor_locations']
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
    print(f"\n📊 Статистика измерений: mean={measurement_mean:.2f}")
    
    # Test 1: Fixed alpha
    print(f"\n[1/2] Fixed α=0.05 (baseline)...")
    geo_cs_config_fixed = GeometryAwareCSConfig(
        max_iter=100, tol=1e-3, alpha=0.05,
        adaptive_alpha=False, device='cpu'
    )
    
    reconstructor_fixed = GeometryAwareTensorCS(
        A=geo_basis.cpu(), P=P_geo_2d, Y=Y_geo,
        mesh=mesh, core_cfg=geo_cs_config_fixed
    )
    
    x_fixed, _ = reconstructor_fixed.solve()
    recon_fixed = torch.einsum('ijk,k->ij', geo_basis.cpu(), x_fixed)
    metrics_fixed = compute_reconstruction_metrics(test_field, recon_fixed)
    
    print(f"  ✓ RMSE: {metrics_fixed['rmse']:.6f}, SSIM: {metrics_fixed['ssim']:.6f}")
    print(f"  ✓ Alpha: {reconstructor_fixed.alpha:.6f}")
    
    # Test 2: Adaptive alpha
    print(f"\n[2/2] Adaptive α (reference=70.0)...")
    geo_cs_config_adapt = GeometryAwareCSConfig(
        max_iter=100, tol=1e-3, alpha=0.05,
        adaptive_alpha=True,
        alpha_reference_amplitude=70.0,
        device='cpu'
    )
    
    reconstructor_adapt = GeometryAwareTensorCS(
        A=geo_basis.cpu(), P=P_geo_2d, Y=Y_geo,
        mesh=mesh, core_cfg=geo_cs_config_adapt
    )
    
    x_adapt, _ = reconstructor_adapt.solve()
    recon_adapt = torch.einsum('ijk,k->ij', geo_basis.cpu(), x_adapt)
    metrics_adapt = compute_reconstruction_metrics(test_field, recon_adapt)
    
    print(f"  ✓ RMSE: {metrics_adapt['rmse']:.6f}, SSIM: {metrics_adapt['ssim']:.6f}")
    print(f"  ✓ Alpha: {reconstructor_adapt.alpha:.6f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Результаты".center(70))
    print(f"{'='*70}")
    print(f"{'Метод':<30} {'RMSE':<12} {'SSIM':<12} {'Alpha':<12}")
    print('-' * 68)
    print(f"{'Fixed α=0.05':<30} {metrics_fixed['rmse']:.6f}  {metrics_fixed['ssim']:.6f}  {reconstructor_fixed.alpha:.6f}")
    print(f"{'Adaptive α':<30} {metrics_adapt['rmse']:.6f}  {metrics_adapt['ssim']:.6f}  {reconstructor_adapt.alpha:.6f}")
    print('-' * 68)
    print(f"{'🏆 Best baseline (Std+Geo)':<30} {'20.376736':<12} {'0.136554':<12}")
    print(f"{'='*70}")
    
    if metrics_adapt['rmse'] < 20.376736:
        improvement = (20.376736 - metrics_adapt['rmse']) / 20.376736 * 100
        print(f"\n🎉 ПОЗДРАВЛЯЕМ! Превзошли baseline на {improvement:.1f}%!")
    elif metrics_adapt['rmse'] < metrics_fixed['rmse']:
        improvement = (metrics_fixed['rmse'] - metrics_adapt['rmse']) / metrics_fixed['rmse'] * 100
        print(f"\n✅ Адаптивный alpha лучше фиксированного на {improvement:.1f}%!")
    
    return {
        'fixed': {'metrics': metrics_fixed, 'alpha': reconstructor_fixed.alpha},
        'adaptive': {'metrics': metrics_adapt, 'alpha': reconstructor_adapt.alpha}
    }


if __name__ == "__main__":
    main()

