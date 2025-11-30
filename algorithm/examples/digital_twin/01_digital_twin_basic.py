#!/usr/bin/env python3
"""
Digital Twin Basic Example

Демонстрация основных возможностей цифрового двойника с TBMD
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from TBMD.config import DigitalTwinConfig
from TBMD.core.digital_twin import DigitalTwin


print("=" * 60)
print("TBMD Digital Twin - Basic Example")
print("=" * 60)

# 1. Создать синтетические данные месторождения
print("\n1. Создание синтетических данных...")
I = 100  # Пространственные точки
J = 2    # Переменные (давление, насыщенность)
T = 50   # Временные шаги

# Симулируем простую динамику
np.random.seed(42)
torch.manual_seed(42)

# Базовое поле
x = torch.linspace(0, 1, I)
base_pressure = torch.sin(2 * np.pi * x).unsqueeze(0).unsqueeze(-1)  # (1, I, 1)
base_saturation = torch.cos(2 * np.pi * x).unsqueeze(0).unsqueeze(-1)  # (1, I, 1)

# Динамика во времени
time = torch.linspace(0, 1, T)
temporal_evolution = torch.exp(-0.1 * time).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

# Исторические данные (I × J × T)
historical_data = torch.zeros(I, J, T)
historical_data[:, 0, :] = (base_pressure.squeeze() * temporal_evolution.squeeze()).T
historical_data[:, 1, :] = (base_saturation.squeeze() * (1 - temporal_evolution.squeeze())).T

# Добавить шум
historical_data += 0.1 * torch.randn_like(historical_data)

print(f"   Данные созданы: {historical_data.shape}")
print(f"   Диапазон давления: [{historical_data[:, 0, :].min():.2f}, {historical_data[:, 0, :].max():.2f}]")
print(f"   Диапазон насыщенности: [{historical_data[:, 1, :].min():.2f}, {historical_data[:, 1, :].max():.2f}]")

# 2. Создать и обучить Digital Twin
print("\n2. Обучение Digital Twin...")
config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=15,
    verbose=True
)

twin = DigitalTwin(config)
twin.train(historical_data, normalize=True)

print(f"   ✅ Twin обучен!")
print(f"   Размещено сенсоров: {len(twin.get_sensor_locations())}")

# 3. Прогнозирование
print("\n3. Прогнозирование будущих состояний...")
current_state = historical_data[:, :, -1]  # Последнее состояние
forecast = twin.predict(current_state, n_steps=10, return_full_field=True)

print(f"   Прогноз создан: {forecast.shape}")
print(f"   Диапазон прогноза: [{forecast.min():.2f}, {forecast.max():.2f}]")

# 4. Симуляция измерений и реконструкция
print("\n4. Реконструкция из измерений сенсоров...")
sensor_indices = twin.get_sensor_locations()

# Симулировать измерения
true_field = current_state.clone()
sensor_measurements = true_field[sensor_indices, :]

print(f"   Измерения с {len(sensor_indices)} сенсоров")

# Реконструировать
reconstructed = twin.update_from_sensors(sensor_measurements)

# Вычислить ошибку
error = torch.norm(reconstructed - true_field) / torch.norm(true_field)
print(f"   Ошибка реконструкции: {error:.4f}")

# 5. Сценарный анализ
print("\n5. Сценарный анализ...")
scenarios = [
    {'name': 'baseline', 'description': 'Текущий режим'},
    {'name': 'optimistic', 'description': 'Оптимистичный сценарий'},
    {'name': 'pessimistic', 'description': 'Пессимистичный сценарий'}
]

results = twin.evaluate_scenarios(scenarios, n_steps=10)

print(f"   Оценено {len(results)} сценариев:")
for name, metrics in results.items():
    print(f"   - {name}: mean={metrics['mean_value']:.3f}, "
          f"std={metrics['std_value']:.3f}")

# 6. Детекция аномалий
print("\n6. Детекция аномалий...")

# Создать данные с аномалией
sensor_data_normal = torch.randn(len(sensor_indices), 10) * 0.1
sensor_data_anomaly = sensor_data_normal.clone()
sensor_data_anomaly[:, 5] += 5.0  # Добавить аномалию на шаге 5

anomalies = twin.detect_anomalies(sensor_data_anomaly, threshold=2.0)

print(f"   Обнаружено аномалий: {len(anomalies)}")
for anomaly in anomalies:
    print(f"   - Timestamp: {anomaly['timestamp']}, "
          f"Residual: {anomaly['residual']:.3f}, "
          f"Severity: {anomaly['severity']}")

# 7. Статистика
print("\n7. Статистика Digital Twin...")
stats = twin.get_statistics()
print(f"   Откалиброван: {stats['is_calibrated']}")
print(f"   Пространственных мод: {stats['n_spatial_modes']}")
print(f"   Сенсоров: {stats['n_sensors']}")
print(f"   Статус: {stats['alert_status']}")

# 8. Визуализация (опционально)
print("\n8. Создание визуализации...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Digital Twin Results', fontsize=16)
    
    # Исторические данные
    axes[0, 0].plot(historical_data[50, 0, :].numpy(), label='Давление')
    axes[0, 0].set_title('Историческое давление (точка 50)')
    axes[0, 0].set_xlabel('Время')
    axes[0, 0].legend()
    
    # Текущее состояние
    axes[0, 1].plot(current_state[:, 0].numpy(), label='Давление')
    axes[0, 1].scatter(sensor_indices, current_state[sensor_indices, 0].numpy(), 
                      c='red', label='Сенсоры', zorder=5)
    axes[0, 1].set_title('Текущее поле + сенсоры')
    axes[0, 1].legend()
    
    # Прогноз
    axes[0, 2].plot(forecast[50, 0, :].numpy())
    axes[0, 2].set_title('Прогноз (точка 50)')
    axes[0, 2].set_xlabel('Шаги прогноза')
    
    # Реконструкция
    axes[1, 0].plot(true_field[:, 0].numpy(), label='True')
    axes[1, 0].plot(reconstructed[:, 0].numpy(), label='Reconstructed', alpha=0.7)
    axes[1, 0].set_title('Реконструкция vs True')
    axes[1, 0].legend()
    
    # Сценарии
    scenario_names = list(results.keys())
    scenario_means = [results[name]['mean_value'] for name in scenario_names]
    axes[1, 1].bar(range(len(scenario_names)), scenario_means)
    axes[1, 1].set_xticks(range(len(scenario_names)))
    axes[1, 1].set_xticklabels(scenario_names, rotation=45)
    axes[1, 1].set_title('Сценарии - Средние значения')
    
    # Аномалии
    residuals = [a['residual'] for a in anomalies]
    timestamps = [a['timestamp'] for a in anomalies]
    if anomalies:
        axes[1, 2].scatter(timestamps, residuals, c='red', s=100)
        axes[1, 2].axhline(y=2.0, color='orange', linestyle='--', label='Threshold')
        axes[1, 2].set_title('Обнаруженные аномалии')
        axes[1, 2].set_xlabel('Timestamp')
        axes[1, 2].set_ylabel('Residual')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('digital_twin_results.png', dpi=150, bbox_inches='tight')
    print("   ✅ Визуализация сохранена: digital_twin_results.png")
    
except Exception as e:
    print(f"   ⚠️  Визуализация пропущена: {e}")

print("\n" + "=" * 60)
print("✅ Digital Twin Example завершен успешно!")
print("=" * 60)

