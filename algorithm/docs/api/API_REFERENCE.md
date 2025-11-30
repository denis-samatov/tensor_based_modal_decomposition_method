# TBMD API Reference

Полная справка по API библиотеки TBMD.

## 📦 Содержание

- [Configuration](#configuration)
- [Core Modules](#core-modules)
  - [Decomposition](#decomposition)
  - [Sensor Placement](#sensor-placement)
  - [Reconstruction](#reconstruction)
- [Digital Twin](#digital-twin)
- [Utilities](#utilities)

---

## Configuration

### `BaseConfig`

Базовый класс конфигурации.

```python
from algorithm.TBMD.config import BaseConfig

class BaseConfig:
    device: str = 'cpu'              # 'cpu', 'cuda', 'mps'
    dtype: str = 'float32'           # 'float32', 'float64'
    backend: str = 'torch'           # 'torch', 'numpy'
    random_state: Optional[int] = None
    verbose: bool = False
    eps: float = 1e-8
```

### `DecompositionConfig`

Конфигурация для Tucker декомпозиции.

```python
from algorithm.TBMD.config import DecompositionConfig

config = DecompositionConfig(
    ranks: List[int] = [20, 10],     # [spatial_rank, temporal_rank]
    normalize: bool = True,           # Нормализовать данные
    center: bool = True,              # Центрировать данные
    max_iterations: int = 100,        # Максимум итераций
    tolerance: float = 1e-4,          # Точность сходимости
    **base_config_params
)
```

### `SensorPlacementConfig`

Конфигурация для размещения сенсоров.

```python
from algorithm.TBMD.config import SensorPlacementConfig

config = SensorPlacementConfig(
    n_sensors: int,                   # Количество сенсоров (обязательно)
    min_distance: float = 0.0,        # Минимальное расстояние между сенсорами
    coverage_threshold: float = 0.9,  # Порог покрытия
    **base_config_params
)
```

### `ReconstructionConfig`

Конфигурация для реконструкции.

```python
from algorithm.TBMD.config import ReconstructionConfig

config = ReconstructionConfig(
    solver: str = 'admm',             # 'least_squares', 'admm', 'ista'
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    lambda_reg: float = 0.01,         # Регуляризация
    rho: float = 1.0,                 # ADMM параметр
    alpha: float = 1.0,               # ISTA step size
    **base_config_params
)
```

### `DigitalTwinConfig`

Конфигурация для цифрового двойника.

```python
from algorithm.TBMD.core.digital_twin.system import DigitalTwinConfig

config = DigitalTwinConfig(
    n_spatial_modes: int = 40,
    n_temporal_modes: int = 20,
    n_sensors: int = 30,
    solver: str = 'admm',
    max_iterations: int = 100,
    **base_config_params
)
```

---

## Core Modules

### Decomposition

#### `TuckerDecomposer`

Tucker (HOSVD) декомпозиция.

```python
from algorithm.TBMD.core.decomposition import TuckerDecomposer
from algorithm.TBMD.config import DecompositionConfig

decomposer = TuckerDecomposer(config: DecompositionConfig)
```

**Методы:**

##### `decompose()`

Выполнить декомпозицию.

```python
result = decomposer.decompose(
    tensor: torch.Tensor,              # (I, J, T)
    ranks: Optional[List[int]] = None  # Переопределить ранги
) -> DecompositionResult
```

### Sensor Placement

#### `TensorTubeQRDecomposition`

QR-based sensor placement.

```python
from algorithm.TBMD.core.sensor_placement import TensorTubeQRDecomposition
from algorithm.TBMD.config import SensorPlacementConfig

placer = TensorTubeQRDecomposition(config: SensorPlacementConfig)
```

**Методы:**

##### `place_sensors()`

Разместить сенсоры на основе пространственных мод.

```python
result = placer.place_sensors(
    spatial_modes: torch.Tensor,       # (I*J, R)
    constraints: Optional[Dict] = None # Ограничения размещения
) -> SensorPlacementResult
```

### Reconstruction

#### `TensorCompressiveSensing`

Compressive sensing reconstruction.

```python
from algorithm.TBMD.core.reconstruction import TensorCompressiveSensing
from algorithm.TBMD.config import ReconstructionConfig

reconstructor = TensorCompressiveSensing(config: ReconstructionConfig)
```

**Методы:**

##### `reconstruct()`

Реконструировать поле из измерений.

```python
result = reconstructor.reconstruct(
    dictionary: torch.Tensor,           # (I*J, R) - Пространственные моды
    measurements: torch.Tensor,         # (N, 1) - Измерения
    measurement_matrix: torch.Tensor,   # (N, I*J) - Матрица измерений
    initial_guess: Optional[torch.Tensor] = None
) -> ReconstructionResult
```

---

## Digital Twin

### `DigitalTwinTBMD`

Цифровой двойник с TBMD.

```python
from algorithm.TBMD.core.digital_twin.system import DigitalTwinTBMD, DigitalTwinConfig

twin = DigitalTwinTBMD(config: DigitalTwinConfig)
```

**Методы:**

##### `train()`

Обучить digital twin на исторических данных.

```python
twin.train(
    historical_data: torch.Tensor,  # (I, J, T)
    normalize: bool = True
) -> None
```

##### `predict_next_state()`

Прогнозировать будущие состояния.

```python
forecast = twin.predict_next_state(
    current_state: ReservoirState,
    well_controls: List[WellControl],
    time_horizon: float,
    time_steps: int
) -> List[ReservoirState]
```

##### `update_from_sensors()`

Обновить состояние из измерений.

```python
result = twin.update_from_sensors(
    sensor_readings: torch.Tensor,  # (N,)
    current_time: float
) -> Dict[str, Any]
```

---

## Utilities

### Tensor helpers
```python
from algorithm.TBMD.utils import (
    get_torch_device, to_torch_tensor,
    reconstruct_tensor
)
```

### Metrics
```python
from algorithm.TBMD.utils.metrics import compute_metrics
```

---

**Версия API**: 1.0
