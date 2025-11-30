# Цифровой двойник на основе TBMD - Полная документация

## 📖 Оглавление

1. [Что такое цифровой двойник?](#что-такое-цифровой-двойник)
2. [Архитектура и схема TBMD Digital Twin](#архитектура-и-схема-tbmd-digital-twin)
3. [Установка и настройка](#установка-и-настройка)
4. [Как работает Digital Twin](#как-работает-digital-twin)
5. [Пошаговое руководство](#пошаговое-руководство)
6. [Компоненты системы](#компоненты-системы)
7. [Примеры использования](#примеры-использования)
8. [API Reference](#api-reference)
9. [Часто задаваемые вопросы](#faq)

---

## Что такое цифровой двойник?

### Определение

**Цифровой двойник** (Digital Twin) — это виртуальная модель физического объекта или процесса, которая:
- 🔄 Работает в режиме реального времени
- 📡 Обновляется на основе реальных измерений с датчиков
- 🔮 Позволяет прогнозировать будущее поведение системы
- 🎯 Дает возможность оценивать различные сценарии без риска

### Зачем использовать TBMD?

#### Традиционные подходы (полные гидродинамические симуляторы):
- ❌ **Медленные**: расчет занимает часы или дни
- ❌ **Ресурсоемкие**: требуют мощных серверов
- ❌ **Не подходят для реального времени**
- ❌ **Сложная калибровка**

#### TBMD подход:
- ✅ **Быстрые прогнозы**: секунды вместо часов
- ✅ **Компактное представление**: ROM (Reduced Order Model)
- ✅ **Оптимальное размещение сенсоров**
- ✅ **Точная реконструкция** по редким измерениям
- ✅ **Учет геометрии** резервуара
- ✅ **Легкая интеграция** с существующими системами

### Ключевые возможности

1. **Обучение на исторических данных**
   - TBMD-декомпозиция для выделения пространственно-временных паттернов
   - Построение модели сниженного порядка (ROM)
   - Калибровка упрощенной физической модели (proxy model)

2. **Оптимальное размещение датчиков**
   - Тензорный QR алгоритм
   - Минимизация количества сенсоров
   - Автоматический выбор информативных точек

3. **Реконструкция полных полей**
   - Восстановление по разреженным измерениям
   - Компрессивный сенсинг (Compressed Sensing)
   - Сохранение физических свойств

4. **Быстрый сценарный анализ**
   - What-if анализ за секунды
   - Сравнение стратегий разработки
   - Оптимизация управления скважинами

5. **Мониторинг в реальном времени**
   - Сравнение прогноза и факта
   - Обнаружение аномалий
   - Автоматические алерты

---

## Архитектура и схема TBMD Digital Twin

### Основные компоненты

1. **DigitalTwin** — главный класс-оркестратор (TBMD + forecaster + CS).
2. **TuckerDecomposer / ModalProcessor / ModalTensorStacker** — декомпозиция и построение модального базиса `A_tensor`.
3. **TensorTubeQRDecomposition** — выбор сенсоров (DEIM-подобный QR, маска `P`, индексы).
4. **Forecasters** — Linear / MLP / LSTM (auto-regress) в модальном пространстве.
5. **TensorCompressiveSensing** — ADMM-решатель для реконструкции поля по сенсорам.
6. **Proxy Model** (опционально, в ноутбуке выключено) — linear/neural/physics-informed.
7. **Мониторинг/валидация** — метрики MSE/RMSE/MAE/R², относительная ошибка, baseline persistence.

---

## Установка и настройка

### Требования

```bash
Python >= 3.8
PyTorch >= 2.0
NumPy >= 1.20
TensorLy >= 0.7
```

### Установка

```bash
# Клонировать репозиторий
git clone https://github.com/your-repo/tensor-based-modal-decomposition-method.git
cd tensor-based-modal-decomposition-method

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # На Windows: .venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### Быстрый тест

```python
from algorithm.TBMD.modules.DigitalTwinTBMD import DigitalTwinTBMD, DigitalTwinConfig

# Создать конфигурацию
config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=30
)

# Создать цифровой двойник
twin = DigitalTwinTBMD(config)
print("✅ Digital Twin готов к работе!")
```

---

## Как работает Digital Twin

### Этап 1: Обучение (детально)
1. **Загрузка/нормализация**  
   - Данные: тензоры `(H, W[, D], T)` по субъектам.  
   - Нормализация (MinMax/Z-score) с глобальными параметрами train, сплит по времени (ordered).
2. **TBMD-декомпозиция**  
   - Tucker/HOSVD: `X ≈ G ×₁ U₁ ×₂ U₂ ×₃ … ×ₙ Uₙ`, ранги ограничены `n_spatial_modes`, `n_temporal_modes`.  
   - Modal Processor: для каждого временного среза core + spatial factors → модальные срезы.  
   - Modal Stacker: объединяет модальные тензоры в `A_tensor` (базис), последняя ось — число мод.
3. **Размещение сенсоров (TensorTubeQR)**  
   - QR с pivoting по “тюбам” `A_tensor` → бинарная маска `P`, индексы сенсоров.  
   - Число сенсоров авто-поднимается до числа мод, чтобы CS не была сильно недоопределённой.  
   - Проверка ортогональности/ранга для устойчивости.
4. **Обучение forecaster (Linear / MLP / LSTM)**  
   - Проекция всех временных срезов (обычно первого субъекта) в модальное пространство: `x(t) = argmin ||A·x - state_t||`.  
   - Обучение переходов `x(t) → x(t+1)` (auto-regressive).  
   - Hyperparams: `hidden_size`, `num_layers`, `seq_length`, `dropout`, `lr`, gradient clipping (LSTM).
5. **(Опционально) Proxy model**  
   - Linear/Neural/Physics-informed для сценариев со скважинами; в ноутбуке отключена (`proxy=None`).

### Этап 2: Работа / прогноз / реконструкция (детально)
1. **Прогноз**  
   - Нормализация (если заданы mean/std).  
   - Проекция `current_state` в модальное пространство через `torch.linalg.lstsq`.  
   - Forecaster генерирует `x̂(t+1..t+k)` авто-регрессионно (без teacher forcing в валидации).  
   - Реконструкция полей: `X̂ = A_tensor @ x̂` (через `reconstruct_tensor` или матричное умножение) → денормализация.
2. **Реконструкция по сенсорам (Compressive Sensing)**  
   - Измерения приводятся к полной маске `P`.  
   - ADMM-решатель (`TensorCompressiveSensing`) решает `min ||x||₁ s.t. P⊙(A x) = Y` → `x̂`.  
   - Восстановленное поле `X̂ = A_tensor @ x̂`; обновляется состояние twin.
3. **Валидация/метрики**  
   - MSE/RMSE/MAE/R² в норм/денорм шкале, относительная ошибка `||pred-gt|| / ||gt||`.  
   - Baseline persistence (norm/denorm) для сравнения; визуализации GT/Pred/Error.

---

## Пошаговое руководство

👉 **[Перейти к пошаговому руководству (Tutorial)](../tutorials/digital_twin_tutorial.md)**

---

## Компоненты системы

### 1. DigitalTwinTBMD - Главный класс

**Назначение**: Оркестрация всех компонентов цифрового двойника

**Основные методы**:

#### `train()`
Обучение на исторических данных

```python
summary = twin.train(
    historical_data=data,           # torch.Tensor (spatial_dims, time)
    historical_controls=controls,   # List[List[WellControl]]
    mesh=mesh,                      # Optional[MeshGeometry]
    rejection_domain=mask           # Optional[torch.Tensor]
)
```

**Возвращает**:
```python
{
    'decomposition': {
        'reconstruction_error': float,
        'n_modes': (n_spatial, n_temporal)
    },
    'sensor_placement': {
        'n_sensors': int,
        'locations': torch.Tensor
    },
    'calibration': {
        'mse': float,
        'relative_error': float
    }
}
```

#### `predict_next_state()`
Прогноз будущего состояния

```python
predictions = twin.predict_next_state(
    current_state=ReservoirState(...),
    well_controls=[WellControl(...)],
    time_horizon=10.0,     # Горизонт прогноза
    time_steps=10          # Количество шагов
)
```

#### `update_from_sensors()`
Обновление состояния из измерений

```python
result = twin.update_from_sensors(
    sensor_readings=readings,      # torch.Tensor (n_sensors,)
    sensor_locations=locations,    # Optional
    current_time=t
)

# result содержит:
{
    'reconstructed_field': torch.Tensor,  # Восстановленное поле
    'metrics': {
        'mse': float,
        'relative_error': float,
        'max_error': float
    },
    'alert_status': str  # 'normal', 'warning', 'critical'
}
```

#### `evaluate_scenarios()`
Сценарный анализ

```python
scenarios = {
    'Scenario1': [controls1...],
    'Scenario2': [controls2...],
}

results = twin.evaluate_scenarios(
    scenarios=scenarios,
    time_horizon=20.0,
    time_steps=20
)
```

### 2. ReservoirProxyModel - Упрощенные модели

#### 2.1 LinearDynamicsProxyModel

**Описание**: Линейная модель динамики

```
x(t+1) = A·x(t) + B·u(t)
```

**Когда использовать**:
- ✅ Простая/слабо нелинейная динамика
- ✅ Нужна максимальная скорость
- ✅ Мало обучающих данных

**Пример**:
```python
from algorithm.TBMD.models.ReservoirProxyModel import LinearDynamicsProxyModel

proxy = LinearDynamicsProxyModel(
    spatial_shape=(100, 100),
    modal_basis=decomposer.factors[0],
    device='cpu'
)

# Калибровка
metrics = proxy.calibrate(
    historical_states=states,
    historical_controls=controls
)
```

#### 2.2 NeuralProxyModel

**Описание**: Нейросетевая модель для сложной динамики

**Когда использовать**:
- ✅ Сильно нелинейная динамика
- ✅ Много обучающих данных
- ✅ Нужна высокая точность

**Пример**:
```python
from algorithm.TBMD.models.ReservoirProxyModel import NeuralProxyModel

proxy = NeuralProxyModel(
    spatial_shape=(100, 100),
    modal_basis=modal_basis,
    hidden_layers=[128, 64, 32],
    device='cpu'
)
```

#### 2.3 PhysicsInformedProxyModel

**Описание**: Модель с физическими ограничениями

**Особенности**:
- Учет законов сохранения массы
- Учет энергетического баланса
- Физически осмысленные прогнозы

**Когда использовать**:
- ✅ Важно соблюдение физики
- ✅ Мало данных
- ✅ Нужна надежность

### 3. RealtimeMonitor - Мониторинг

**Назначение**: Отслеживание качества прогнозов

```python
from algorithm.TBMD.core.digital_twin.system import RealtimeMonitor

monitor = RealtimeMonitor(alert_threshold=0.15)

# Сравнение
metrics = monitor.compare_prediction_observation(
    predicted=predicted_field,
    observed=observed_field,
    sensor_locations=sensor_mask
)

# Проверка статуса
status = monitor.check_alert_status(metrics)
# 'normal', 'warning', 'critical'
```

### 4. ScenarioAnalyzer - Сценарный анализ

**Назначение**: What-if анализ

```python
from algorithm.TBMD.core.digital_twin.system import ScenarioAnalyzer

analyzer = ScenarioAnalyzer(proxy_model)

# Оценка сценария
result = analyzer.evaluate_scenario(
    scenario_name='High Production',
    initial_state=state,
    well_controls=controls,
    time_horizon=30.0,
    time_steps=30
)

# KPI
kpis = result['kpis']
# {
#     'avg_pressure': float,
#     'total_production': float,
#     'max_drawdown': float
# }
```

---

## Примеры использования

### Пример 1: Базовый workflow

```python
import torch
from algorithm.TBMD.modules.DigitalTwinTBMD import (
    DigitalTwinTBMD, DigitalTwinConfig
)
from algorithm.TBMD.models.ReservoirProxyModel import (
    WellControl, ReservoirState
)

# 1. Конфигурация
config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_temporal_modes=20,
    n_sensors=30
)

# 2. Создание
twin = DigitalTwinTBMD(config)

# 3. Обучение
twin.train(historical_data, historical_controls)

# 4. Использование
current = ReservoirState(pressure=field, time=100.0)
controls = [WellControl('WELL1', 'rate', 1000, (10, 20))]

# Прогноз
forecast = twin.predict_next_state(current, controls)

# Обновление
sensor_data = get_measurements()
twin.update_from_sensors(sensor_data)
```

### Пример 2: Интеграция с Brugge данными

См. полный пример: [`algorithm/scripts/run_brugge_enhanced.py`](algorithm/scripts/run_brugge_enhanced.py)

### Пример 3: Пользовательская proxy модель

```python
from algorithm.TBMD.models.ReservoirProxyModel import ReservoirProxyModelBase

class CustomProxy(ReservoirProxyModelBase):
    def forecast(self, current_state, well_controls, time_horizon, time_steps):
        # Ваша логика
        predictions = []
        # ...
        return predictions
    
    def calibrate(self, states, controls):
        # Ваша калибровка
        return metrics

# Использование
twin = DigitalTwinTBMD(config)
twin.proxy_model = CustomProxy(...)
```

---

## API Reference

### Классы данных

#### ReservoirState
```python
@dataclass
class ReservoirState:
    pressure: torch.Tensor          # Поле давления
    saturation: Optional[torch.Tensor] = None  # Насыщенность
    time: float = 0.0              # Временная метка
    well_rates: Optional[Dict] = None  # Дебиты скважин
```

#### WellControl
```python
@dataclass
class WellControl:
    well_name: str                 # Название скважины
    control_type: str              # 'rate', 'pressure', 'bhp'
    value: float                   # Значение
    location: Tuple[int, ...]      # Координаты (x, y) или (x, y, z)
```

#### DigitalTwinConfig
```python
@dataclass
class DigitalTwinConfig:
    n_spatial_modes: int = 50
    n_temporal_modes: int = 20
    n_sensors: int = 30
    proxy_model_type: str = 'linear'  # 'linear', 'neural', 'physics_informed'
    use_geometry_aware: bool = False
    reconstruction_method: str = 'admm'  # 'admm', 'ista', 'least_squares'
    update_frequency: int = 10
    alert_threshold: float = 0.15
    device: str = 'cpu'
    dtype: str = 'float32'
```

---

## FAQ

### Q: Какие данные нужны для обучения?

**A**: Минимально:
- Временной ряд полей (nx, ny, [nz,] nt)
- Информация о скважинах (опционально, но рекомендуется)

Рекомендуется 50-100 временных снимков для линейной модели, 200+ для нейросетевой.

### Q: Как выбрать количество мод?

**A**: 
1. Начните с `n_spatial_modes = 30-50`
2. Проверьте reconstruction error после обучения
3. Если error > 0.1, увеличьте количество мод
4. Обычно `n_temporal_modes = n_spatial_modes / 2`

### Q: Какую proxy-модель выбрать?

**A**:
- **Linear**: простая динамика, быстро, мало данных
- **Neural**: сложная динамика, много данных
- **Physics-informed**: важна физическая корректность

### Q: Как часто обновлять модель?

**A**:
- Реальное время мониторинг: каждые 1-5 минут
- Оперативное планирование: каждые 1-6 часов
- Стратегическое: каждые 1-7 дней

### Q: Что делать при высоких ошибках?

**A**: Проверьте:
1. Достаточно ли мод? (увеличьте `n_spatial_modes`)
2. Достаточно ли сенсоров? (увеличьте `n_sensors`)
3. Правильно ли размещены сенсоры?
4. Качество исходных данных

### Q: Можно ли использовать GPU?

**A**: Да! Установите `device='cuda'` в конфиге

---

## Дополнительные ресурсы

### Документация
- [TBMD Overview](tbmd_core.md)
- [Geometry-Aware TBMD](geometry_aware_tbmd.md)
- [Основной README](../README.md)

### Примеры
- Базовый: `algorithm/scripts/run_digital_twin_demo.py`
- Brugge: `algorithm/scripts/run_brugge_enhanced.py`
- Notebook: `algorithm/experiments/exp_tbmd_4_digital_twin.ipynb`

### Анализ
- [Brugge Digital Twin Analysis](../examples/brugge_digital_twin_analysis.md)

---

**Версия**: 1.0  
**Дата**: Ноябрь 2025  
**Автор**: TBMD Team
