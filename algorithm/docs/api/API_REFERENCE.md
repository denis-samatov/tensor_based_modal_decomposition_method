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
from TBMD.config import BaseConfig

class BaseConfig:
    backend: str = 'pytorch'         # 'pytorch', 'numpy'
    dtype: str = 'float32'           # 'float32', 'float64'
    device: Optional[str] = None     # 'cpu', 'cuda', 'mps' (auto if None)
    seed: Optional[int] = 0
    deterministic: bool = True
    verbose: bool = True
    log_level: str = 'INFO'
```

### `DecompositionConfig`

Конфигурация для Tucker декомпозиции.

```python
from TBMD.config import DecompositionConfig

config = DecompositionConfig(
    ranks: Optional[Union[int, List[int]]] = None,  # Tucker ranks
    method: str = 'hosvd',
    epsilon: float = 1e-2,            # Convergence tolerance
    min_rank: int = 1,                # Minimum rank
    max_workers: Optional[int] = None, # For parallel processing
    **base_config_params
)
```

### `SensorPlacementConfig`

Конфигурация для размещения сенсоров.

```python
from TBMD.config import SensorPlacementConfig

config = SensorPlacementConfig(
    n_sensors: int,                   # Количество сенсоров (обязательно)
    check_orthogonality: bool = False, # Проверять ортогональность
    uniform_distribution: bool = False, # Равномерное распределение
    **base_config_params
)
```

### `CompressiveSensingConfig`

Конфигурация для реконструкции.

```python
from TBMD.config import CompressiveSensingConfig

config = CompressiveSensingConfig(
    max_iter: int = 1000,
    tol: float = 1e-4,
    epsilon_l1: float = 1e-2,         # L1 regularization
    delta_init: float = 1.0,          # ADMM delta parameter
    delta_max: float = 1.0,
    relax_lambda: float = 0.95,       # ADMM relaxation (0 < relax_lambda < 1)
    device: str = 'cpu',
    dtype: str = 'float32'
)
```

### `DigitalTwinConfig`

Конфигурация для цифрового двойника.

```python
from TBMD.config import DigitalTwinConfig

config = DigitalTwinConfig(
    n_spatial_modes: int = 40,
    n_temporal_modes: int = 20,
    n_sensors: int = 30,
    forecaster_type: str = 'lstm', # 'linear', 'mlp', 'lstm', 'persistence'
    proxy_model_type: Optional[str] = None, # 'linear_dynamics', 'neural', 'physics_informed'
    **base_config_params
)
```

---

## Core Modules

### Decomposition

#### `TuckerDecomposer`

Tucker (HOSVD) декомпозиция.

```python
from TBMD.core.decomposition.hosvd import TuckerDecomposer

decomposer = TuckerDecomposer(
    tensors: Union[torch.Tensor, Dict[str, torch.Tensor]],
    config: Optional[DecompositionConfig] = None,
    # Или параметры напрямую:
    ranks=[20, 10, 5],
    device='cpu'
)
```

**Методы и Свойства:**

##### `decompose()`

Выполнить декомпозицию.

```python
decomposer.decompose()
```

##### Результаты

После вызова `decompose()`:

```python
cores = decomposer.cores       # Core tensors
factors = decomposer.factors   # Factor matrices
```

### Sensor Placement

#### `TensorTubeQRDecomposition`

QR-based sensor placement (Algorithm 2).

```python
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition
from TBMD.config import SensorPlacementConfig

placer = TensorTubeQRDecomposition(
    tensor: torch.Tensor,           # Input tensor (e.g. spatial modes)
    config: SensorPlacementConfig
)
```

**Методы:**

##### `factorize()`

Выполнить факторизацию и размещение сенсоров.

```python
P, Q, R = placer.factorize()
# P: Binary mask of sensor locations
# Q: Orthogonal basis
# R: Upper triangular matrix
```

### Reconstruction

#### `TensorCompressiveSensing`

Compressive sensing reconstruction (Algorithm 3).

```python
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

reconstructor = TensorCompressiveSensing(
    A: torch.Tensor,                # Dictionary (spatial modes)
    P: torch.Tensor,                # Sensor mask
    Y: torch.Tensor,                # Measurements
    core_cfg: Optional[CompressiveSensingConfig] = None
)
```

**Методы:**

##### `solve()`

Реконструировать модальные коэффициенты.

```python
x_hat, metrics = reconstructor.solve()
# x_hat: Reconstructed coefficients
# metrics: Reconstruction statistics
```

---

## Digital Twin

### `DigitalTwin`

Цифровой двойник с TBMD.

```python
from TBMD.digital_twin.digital_twin import DigitalTwin
from TBMD.config import DigitalTwinConfig

twin = DigitalTwin(config: DigitalTwinConfig)
```

**Методы:**

##### `train()`

Обучить digital twin на исторических данных.

```python
summary = twin.train(
    historical_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    normalize: bool = False,
    ranks: Optional[List[int]] = None
) -> Dict[str, Any]
```

##### `predict()`

Прогнозировать будущие состояния (автономный режим).

```python
forecast = twin.predict(
    current_state: torch.Tensor,
    n_steps: int = 1,
    return_full_field: bool = True
) -> torch.Tensor
# Returns: (spatial..., n_steps)
```

##### `predict_next_state()`

Прогнозировать следующее состояние с учетом управлений (для сценарного анализа).

```python
prediction = twin.predict_next_state(
    current_state: Union[torch.Tensor, ReservoirState],
    controls: Any
) -> List[ReservoirState]
```

---

## Utilities

### Tensor helpers
```python
from TBMD.core.utils.misc import (
    get_torch_device, to_torch_tensor,
    reconstruct_tensor
)
```

---

**Версия API**: 2.0.0
