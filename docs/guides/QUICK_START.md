# TBMD Quick Start Guide

**Быстрый старт для работы с Tensor-Based Modal Decomposition**

## 📦 Установка

```bash
# Клонировать репозиторий
git clone <repository-url>
cd tensor-based-modal-decomposition-method

# Установить зависимости
pip install -r requirements.txt

```

## 🚀 Первые шаги

### 1. Базовая декомпозиция

```python
import torch
from TBMD.config import DecompositionConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposer

# Ваши данные (I × J × T)
data = torch.randn(100, 3, 50)

# Конфигурация
config = DecompositionConfig(
    ranks=[20, 3, 10],  # (I, J, T) -> три ранга
    verbose=True
)

# Декомпозиция
decomposer = TuckerDecomposer(tensors=data, config=config)
decomposer.decompose()

print(f"Core shape: {decomposer.cores.shape}")
print(f"Factors shapes: {[f.shape for f in decomposer.factors]}")

# Реконструкция
decomposer.reconstruct()
reconstructed = decomposer.reconstructed_tensors
```

### 2. Размещение сенсоров

```python
from TBMD.config import SensorPlacementConfig
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition

# Конфигурация
config = SensorPlacementConfig(
    n_sensors=30,
    verbose=True
)

# Правильный поток: Modal Processor -> Modal Stacker -> A_tensor
from TBMD.config import ModalProcessorConfig, ProcessingStrategy
from TBMD.core.modal_processor.modes import BatchModalProcessor, ModalTensorStacker

modal_config = ModalProcessorConfig(
    device=config.device,
    processing_strategy=ProcessingStrategy.BATCH,
    return_numpy=False
)
processor = BatchModalProcessor(modal_config)
stacker = ModalTensorStacker(modal_config)

modal_tensors = processor.process_multiple_subjects(decomposer.cores, decomposer.factors)
A_tensor = stacker.stack_modal_tensors(modal_tensors)

# Размещение сенсоров по A_tensor
placer = TensorTubeQRDecomposition(
    tensor=A_tensor,
    config=config
)
P, Q, R = placer.factorize()

# Индексы сенсоров
print(f"Measurement matrix P shape: {P.shape}")
```

### 3. Реконструкция из измерений

```python
from TBMD.config import CompressiveSensingConfig
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

# Измерения с сенсоров (симуляция)
# P - бинарная маска, Y - полный тензор измерений с нулями вне сенсоров
true_field = data[..., -1]
Y = torch.zeros_like(true_field)
Y[P.bool()] = true_field[P.bool()]

# Конфигурация
config = CompressiveSensingConfig(
    max_iter=100,
    tol=1e-4,
    epsilon_l1=1e-2,
    relax_lambda=0.95
)

# Реконструкция
reconstructor = TensorCompressiveSensing(
    A=A_tensor,  # Modal basis (Dictionary)
    P=P,         # Sensor Mask
    Y=Y,         # Full-size measurements
    core_cfg=config
)

x_hat, metrics = reconstructor.solve()
```

### 4. Полный пайплайн

```python
from TBMD.config import (
    DecompositionConfig,
    SensorPlacementConfig,
    CompressiveSensingConfig
)
from TBMD.core.decomposition.hosvd import TuckerDecomposer
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

# Исторические данные
historical_data = torch.randn(50, 50, 100)  # (Spatial, Time)

# 1. Декомпозиция
decomposer = TuckerDecomposer(
    tensors=historical_data,
    ranks=[20, 20, 10]
)
decomposer.decompose()

# 1.1 Modal Processor -> A_tensor
from TBMD.config import ModalProcessorConfig, ProcessingStrategy
from TBMD.core.modal_processor.modes import BatchModalProcessor, ModalTensorStacker

modal_config = ModalProcessorConfig(
    device='cpu',
    processing_strategy=ProcessingStrategy.BATCH,
    return_numpy=False
)
processor = BatchModalProcessor(modal_config)
stacker = ModalTensorStacker(modal_config)
modal_tensors = processor.process_multiple_subjects(decomposer.cores, decomposer.factors)
A_tensor = stacker.stack_modal_tensors(modal_tensors)

# 2. Размещение сенсоров
placer = TensorTubeQRDecomposition(
    tensor=A_tensor,
    config=SensorPlacementConfig(n_sensors=30)
)
P, Q, R = placer.factorize()

# 3. В реальном времени: измерения -> реконструкция
# Simulate readings
true_field = historical_data[..., -1] # Last time step
Y = torch.zeros_like(true_field)
Y[P.bool()] = true_field[P.bool()]

reconstructor = TensorCompressiveSensing(
    A=A_tensor,
    P=P,
    Y=Y,
    core_cfg=CompressiveSensingConfig(max_iter=100, tol=1e-4, epsilon_l1=1e-2, relax_lambda=0.95)
)
x_hat, metrics = reconstructor.solve()

# Reconstruct full field from coefficients
from TBMD.core.utils.misc import reconstruct_tensor
reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
```

### 5. Digital Twin

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

# Конфигурация
config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_sensors=30,
    forecaster_type='linear',
    verbose=True
)

# Создать twin
twin = DigitalTwin(config)

# Обучить на исторических данных
twin.train(data, normalize=False)

# Прогноз
forecast = twin.predict(data[..., -1], n_steps=10)
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
    ranks=[40, 3, 20],        # Ранги декомпозиции (пример для (I, J, T))
    device='cuda',            # 'cpu', 'cuda', 'mps'
    dtype='float32',          # 'float32' или 'float64'
    normalize=True,           # Нормализовать данные
    verbose=True,             # Вывод информации
    epsilon=1e-8              # Точность
)
```

### Настройка сенсоров

```python
from TBMD.config import SensorPlacementConfig

config = SensorPlacementConfig(
    n_sensors=50,            # Количество сенсоров
    check_orthogonality=True,
    uniform_distribution=False,
    verbose=True
)
```

### Настройка реконструкции

```python
from TBMD.config import CompressiveSensingConfig

config = CompressiveSensingConfig(
    max_iter=100,            # Максимум итераций
    tol=1e-4,                # Точность сходимости
    delta_init=1.0,          # ADMM параметр
    epsilon_l1=1e-2,         # L1 регуляризация
    relax_lambda=0.95        # ADMM relaxation
)
```

## 🎯 Типичные use cases

### 1. Сжатие больших данных

```python
# Данные: 1000 × 10 × 500 = 5M элементов
large_data = torch.randn(1000, 10, 500)

# Декомпозиция
decomposer = TuckerDecomposer(
    tensors=large_data,
    ranks=[50, 25]
)
decomposer.decompose()

# Получаем сжатые факторы и ядра
core = decomposer.cores
factors = decomposer.factors
```

### 2. Оптимальное размещение сенсоров

```python
# Исторические данные
# Найти моды
decomposer = TuckerDecomposer(
    tensors=data,
    ranks=[40, 20, 10]
)
decomposer.decompose()

from TBMD.config import ModalProcessorConfig, ProcessingStrategy
from TBMD.core.modal_processor.modes import BatchModalProcessor, ModalTensorStacker

modal_config = ModalProcessorConfig(
    device='cpu',
    processing_strategy=ProcessingStrategy.BATCH,
    return_numpy=False
)
processor = BatchModalProcessor(modal_config)
stacker = ModalTensorStacker(modal_config)
modal_tensors = processor.process_multiple_subjects(decomposer.cores, decomposer.factors)
A_tensor = stacker.stack_modal_tensors(modal_tensors)

# Разместить 30 сенсоров
placer = TensorTubeQRDecomposition(
    tensor=A_tensor,
    config=SensorPlacementConfig(n_sensors=30)
)
P, Q, R = placer.factorize()

# Вместо 1000 точек измерений нужно только 30!
print(f"Reduction: {(1 - 30 / 1000) * 100:.1f}%")
```

## 📊 Типичная производительность

| Задача | Данные | Сжатие | Точность | Время |
|--------|--------|--------|----------|-------|
| Декомпозиция | 100×3×50 | 20x | 95% энергии | <1s |
| Размещение | 30 сенсоров | 5% от полных | - | <0.1s |
| Реконструкция | ADMM, 100 iter | - | <5% ошибка | ~1s |

## 📚 Дополнительная документация

- [Core Concepts](tbmd_core.md) - Детали реализации
- [API Reference](API_REFERENCE.md) - Конфигурация и API
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
