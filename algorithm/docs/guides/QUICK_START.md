# TBMD Quick Start Guide

**Быстрый старт для работы с Tensor-Based Modal Decomposition**

## 📦 Установка

```bash
# Клонировать репозиторий
git clone <repository-url>
cd tensor-based-modal-decomposition-method

# Установить зависимости
pip install -r requirements.txt

# Или с conda
conda env create -f environment.yml
conda activate tbmd
```

## 🚀 Первые шаги

### 1. Базовая декомпозиция

```python
import torch
from TBMD.config import DecompositionConfig
from TBMD.core.decomposition import TuckerDecomposer

# Ваши данные (I × J × T)
data = torch.randn(100, 3, 50)

# Конфигурация
config = DecompositionConfig(
    ranks=[20, 10],  # [spatial_rank, temporal_rank]
    verbose=True
)

# Декомпозиция
decomposer = TuckerDecomposer(config)
result = decomposer.decompose(data)

print(f"Energy retained: {result.energy_retained:.2%}")
print(f"Reconstruction error: {result.reconstruction_error:.4f}")

# Реконструкция
reconstructed = result.reconstruct()
```

### 2. Размещение сенсоров

```python
from TBMD.config import SensorPlacementConfig
from TBMD.core.sensor_placement import TensorTubeQRDecomposition

# Конфигурация
config = SensorPlacementConfig(
    n_sensors=30,
    verbose=True
)

# Размещение
placer = TensorTubeQRDecomposition(config)
sensor_result = placer.place_sensors(result.spatial_modes)

# Индексы сенсоров
sensor_indices = sensor_result.sensor_indices
print(f"Placed {len(sensor_indices)} sensors")
```

### 3. Реконструкция из измерений

```python
from TBMD.config import ReconstructionConfig
from TBMD.core.reconstruction import TensorCompressiveSensing

# Измерения с сенсоров
measurements = sensor_result.measurement_matrix @ your_field.reshape(-1)

# Конфигурация
config = ReconstructionConfig(
    solver='admm',
    max_iterations=100,
    verbose=True
)

# Реконструкция
reconstructor = TensorCompressiveSensing(config)
recon_result = reconstructor.reconstruct(
    dictionary=result.spatial_modes,
    measurements=measurements.unsqueeze(1),
    measurement_matrix=sensor_result.measurement_matrix
)

reconstructed_field = recon_result.reconstructed_field
```

### 4. Полный пайплайн

```python
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

# Исторические данные
historical_data = load_your_data()  # (I, J, T)

# 1. Декомпозиция
decomposer = TuckerDecomposer(
    DecompositionConfig(ranks=[40, 20])
)
decomp = decomposer.decompose(historical_data)

# 2. Размещение сенсоров
placer = TensorTubeQRDecomposition(
    SensorPlacementConfig(n_sensors=50)
)
sensors = placer.place_sensors(decomp.spatial_modes)

# 3. В реальном времени: измерения -> реконструкция
current_measurements = get_sensor_readings()

reconstructor = TensorCompressiveSensing(
    ReconstructionConfig(solver='admm')
)
recon = reconstructor.reconstruct(
    dictionary=decomp.spatial_modes,
    measurements=current_measurements,
    measurement_matrix=sensors.measurement_matrix
)

current_field = recon.reconstructed_field
```

### 5. Digital Twin

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin import DigitalTwin

# Конфигурация
config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_sensors=30,
    verbose=True
)

# Создать twin
twin = DigitalTwin(config)

# Обучить на исторических данных
twin.train(historical_data, normalize=True)

# Прогноз
forecast = twin.predict(current_state, n_steps=10)

# Обновление из сенсоров
sensor_readings = get_current_readings()
reconstructed = twin.update_from_sensors(sensor_readings)

# Сценарный анализ
scenarios = [
    {'name': 'baseline'},
    {'name': 'optimistic'},
    {'name': 'pessimistic'}
]
results = twin.evaluate_scenarios(scenarios, n_steps=10)

# Детекция аномалий
anomalies = twin.detect_anomalies(sensor_data, threshold=3.0)
```

## 📂 Примеры

### Запуск базовых примеров

```bash
cd algorithm

# Tucker декомпозиция
python examples/basic/01_tucker_decomposition.py

# Размещение сенсоров
python examples/basic/02_sensor_placement.py

# Реконструкция полей
python examples/basic/03_field_reconstruction.py

# Полный пайплайн
python examples/basic/04_complete_pipeline.py --n-modes 30 --n-sensors 40 --visualize

# Digital Twin
python examples/digital_twin/01_digital_twin_basic.py
```

## ⚙️ Конфигурация

### Основные параметры

```python
from TBMD.config import DecompositionConfig

config = DecompositionConfig(
    ranks=[40, 20],          # Ранги декомпозиции
    backend='torch',         # 'torch' или 'numpy'
    device='cuda',           # 'cpu', 'cuda', 'mps'
    dtype='float32',         # 'float32' или 'float64'
    normalize=True,          # Нормализовать данные
    verbose=True,            # Вывод информации
    eps=1e-8                 # Точность
)
```

### Настройка сенсоров

```python
from TBMD.config import SensorPlacementConfig

config = SensorPlacementConfig(
    n_sensors=50,            # Количество сенсоров
    backend='torch',
    device='cpu',
    verbose=True
)
```

### Настройка реконструкции

```python
from TBMD.config import ReconstructionConfig

config = ReconstructionConfig(
    solver='admm',           # 'least_squares', 'admm', 'ista'
    max_iterations=100,      # Максимум итераций
    tolerance=1e-4,          # Точность сходимости
    lambda_reg=0.01,         # Регуляризация
    rho=1.0,                 # ADMM параметр
    backend='torch',
    verbose=True
)
```

## 🎯 Типичные use cases

### 1. Сжатие больших данных

```python
# Данные: 1000 × 10 × 500 = 5M элементов
large_data = torch.randn(1000, 10, 500)

# Декомпозиция с рангом 50
decomposer = TuckerDecomposer(
    DecompositionConfig(ranks=[50, 25])
)
result = decomposer.decompose(large_data)

# Размер после сжатия: ~76K элементов
# Коэффициент сжатия: ~65x
compression_ratio = large_data.numel() / (
    result.spatial_modes.numel() +
    result.temporal_modes.numel() +
    result.core.numel()
)
```

### 2. Оптимальное размещение сенсоров

```python
# Исторические данные
data = load_field_data()  # (200, 5, 100)

# Найти моды
decomposer = TuckerDecomposer(
    DecompositionConfig(ranks=[40, 20])
)
result = decomposer.decompose(data)

# Разместить 30 сенсоров
placer = TensorTubeQRDecomposition(
    SensorPlacementConfig(n_sensors=30)
)
sensors = placer.place_sensors(result.spatial_modes)

# Вместо 1000 точек измерений нужно только 30!
print(f"Reduction: {(1 - 30 / 1000) * 100:.1f}%")
```

### 3. Реконструкция в реальном времени

```python
# Подготовка (один раз)
decomposer = TuckerDecomposer(...)
placer = TensorTubeQRDecomposition(...)
reconstructor = TensorCompressiveSensing(...)

# В реальном времени (много раз)
while monitoring:
    # Получить измерения
    measurements = read_sensors()
    
    # Реконструировать полное поле
    field = reconstructor.reconstruct(
        dictionary=spatial_modes,
        measurements=measurements,
        measurement_matrix=measurement_matrix
    )
    
    # Анализ и принятие решений
    analyze_field(field.reconstructed_field)
```

### 4. Прогнозирование с Digital Twin

```python
# Обучение (один раз)
twin = DigitalTwin(config)
twin.train(historical_data)

# Эксплуатация
current_state = get_current_state()

# Прогноз на 10 шагов вперед
forecast = twin.predict(current_state, n_steps=10)

# Оценка разных сценариев
scenarios = [...]
results = twin.evaluate_scenarios(scenarios)
```

## 📊 Типичная производительность

| Задача | Данные | Сжатие | Точность | Время |
|--------|--------|--------|----------|-------|
| Декомпозиция | 100×3×50 | 20x | 95% энергии | <1s |
| Размещение | 30 сенсоров | 5% от полных | - | <0.1s |
| Реконструкция | ADMM, 100 iter | - | <5% ошибка | ~1s |

## 🔧 Устранение неполадок

### Ошибка: "Out of memory"
```python
# Решение 1: Уменьшить ранг
config = DecompositionConfig(ranks=[20, 10])  # вместо [50, 25]

# Решение 2: Использовать CPU
config = DecompositionConfig(device='cpu')

# Решение 3: Обрабатывать по частям
for chunk in data_chunks:
    result = decomposer.decompose(chunk)
```

### Ошибка: "Reconstruction not converging"
```python
# Решение 1: Увеличить итерации
config = ReconstructionConfig(max_iterations=200)

# Решение 2: Изменить solver
config = ReconstructionConfig(solver='least_squares')

# Решение 3: Настроить регуляризацию
config = ReconstructionConfig(lambda_reg=0.1)
```

### Ошибка: "Poor energy retention"
```python
# Решение: Увеличить ранг
config = DecompositionConfig(ranks=[60, 30])  # вместо [20, 10]
```

## 📚 Дополнительная документация

- [Core Modules](TBMD_CORE_MODULES.md) - Детали реализации
- [Configuration Guide](TBMD_CONFIGURATION.md) - Все параметры конфигурации
- [Examples](../../examples/README.md) - Больше примеров
- [API Reference](API_REFERENCE.md) - Полный API

## 🤝 Поддержка

- 📧 Email: [your-email]
- 🐛 Issues: [GitHub Issues]
- 📖 Docs: [Full Documentation]

## 📝 Цитирование

Если вы используете TBMD в своих исследованиях:

```bibtex
@article{tbmd2024,
  title={Tensor-Based Modal Decomposition for Reservoir Monitoring},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

