# TBMD Examples

Примеры использования Tensor-Based Modal Decomposition

## Структура

### 📁 `basic/` - Базовые примеры
Знакомство с основными компонентами TBMD:
- `01_tucker_decomposition.py` - Tucker разложение
- `02_sensor_placement.py` - Размещение сенсоров
- `03_field_reconstruction.py` - Реконструкция полей
- `04_complete_pipeline.py` - Полный пайплайн TBMD

### 📁 `digital_twin/` - Цифровой двойник
Примеры работы с цифровым двойником:
- `01_digital_twin_basic.py` - Базовый digital twin
- `02_digital_twin_monitoring.py` - Мониторинг и алерты
- `03_digital_twin_scenarios.py` - Сценарный анализ

### 📁 `geometry_aware/` - Geometry-Aware TBMD
Примеры с учетом геометрии:
- `01_geometry_aware_basic.py` - Базовый пример с геометрией
- `02_graph_based_tbmd.py` - TBMD на графах
- `03_anisotropic_fields.py` - Анизотропные поля

### 📁 `advanced/` - Продвинутые примеры
Сложные применения:
- `01_multiphysics.py` - Мультифизика
- `02_uncertainty_quantification.py` - Квантификация неопределенности
- `03_online_learning.py` - Онлайн-обучение

### 📁 `applications/` - Прикладные примеры
Реальные приложения:
- `brugge_field/` - Месторождение Brugge
- `fluid_dynamics/` - Динамика жидкости
- `climate_data/` - Климатические данные

## Быстрый старт

### 1. Базовый пример - Tucker декомпозиция

```python
import torch
from TBMD.config import DecompositionConfig
from TBMD.core.decomposition import TuckerDecomposer

# Данные
data = torch.randn(100, 3, 50)  # (I, J, T)

# Конфигурация
config = DecompositionConfig(
    ranks=[20, 10],
    verbose=True
)

# Декомпозиция
decomposer = TuckerDecomposer(config)
result = decomposer.decompose(data)

print(f"Spatial modes: {result.spatial_modes.shape}")
print(f"Reconstruction error: {result.reconstruction_error:.4f}")
```

### 2. Полный пайплайн

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

# 1. Декомпозиция
decomposer = TuckerDecomposer(
    DecompositionConfig(ranks=[20, 10])
)
result = decomposer.decompose(historical_data)

# 2. Размещение сенсоров
sensor_placer = TensorTubeQRDecomposition(
    SensorPlacementConfig(n_sensors=30)
)
sensors = sensor_placer.place_sensors(result.spatial_modes)

# 3. Реконструкция
reconstructor = TensorCompressiveSensing(
    ReconstructionConfig(solver='admm')
)
recon = reconstructor.reconstruct(
    dictionary=result.spatial_modes,
    measurements=measurements,
    measurement_matrix=sensors.measurement_matrix
)
```

### 3. Digital Twin

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin import DigitalTwin

# Конфигурация
config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_sensors=30
)

# Создать и обучить
twin = DigitalTwin(config)
twin.train(historical_data)

# Прогноз
forecast = twin.predict(current_state, n_steps=10)

# Реконструкция из сенсоров
reconstructed = twin.update_from_sensors(sensor_readings)
```

## Запуск примеров

### Из корня проекта
```bash
cd algorithm
python examples/basic/01_tucker_decomposition.py
```

### С параметрами
```bash
python examples/basic/04_complete_pipeline.py --n-modes 30 --n-sensors 20
```

## Зависимости

```bash
pip install torch numpy tensorly matplotlib
```

Для geometry-aware примеров:
```bash
pip install torch-geometric scipy
```

## Датасеты

### Синтетические
Все базовые примеры генерируют синтетические данные

### Реальные датасеты
- **Brugge Field**: `data/Brugge data/`
- **Navier-Stokes**: `data/navier_stokes/`
- **HW Dynamic**: `data/HW dynamic data/`

## Дополнительная информация

- 📖 [TBMD Overview](../TBMD/docs/TBMD_OVERVIEW.md)
- 📖 [Configuration Guide](../TBMD/docs/TBMD_CONFIGURATION.md)
- 📖 [Core Modules](../TBMD/docs/TBMD_CORE_MODULES.md)
- 📖 [Quick Start](../../QUICK_START_REFACTORING.md)

## Поддержка

При возникновении проблем:
1. Проверьте версии зависимостей
2. Убедитесь, что `PYTHONPATH` настроен правильно
3. Обратитесь к документации модулей
4. Создайте issue в репозитории

