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
6. **Proxy Model** (опционально) — physics-informed proxy для сценариев.
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
from TBMD.digital_twin.digital_twin import DigitalTwin
from TBMD.config import DigitalTwinConfig

# Создать конфигурацию
config = DigitalTwinConfig(
    n_spatial_modes=20,
    n_temporal_modes=10,
    n_sensors=30
)

# Создать цифровой двойник
twin = DigitalTwin(config)
print("✅ Digital Twin готов к работе!")
```

---

## Как работает Digital Twin

### Этап 1: Обучение (детально)
1. **Загрузка/нормализация**  
   - Данные: тензоры `(H, W[, D], T)` по субъектам.  
   - Если требуется нормализация (MinMax/Z-score), выполните её заранее и передайте данные уже нормализованными.
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
   - Linear/Neural/Physics-informed для сценариев со скважинами.

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

### 1. DigitalTwin - Главный класс

**Назначение**: Оркестрация всех компонентов цифрового двойника

**Основные методы**:

#### `train()`
Обучение на исторических данных

```python
summary = twin.train(
    historical_data=data,           # torch.Tensor (spatial_dims, time)
    normalize=False                 # Нормализуйте данные заранее при необходимости
)
```

**Возвращает**:
```python
{
    'ranks': [...],           # Эффективные ранги Tucker
    'modal_dim': int,         # Число мод
    'n_sensors': int,         # Число сенсоров
    'qr_valid': bool,
    'qr_error': float,
    'qr_metrics': {...},      # Метрики QR
    # + опционально метрики обучения forecaster
}
```

#### `predict()`
Прогноз будущего состояния (автономный)

```python
forecast = twin.predict(
    current_state=state_tensor,
    n_steps=10
)
# Returns: Tensor shape (spatial..., n_steps)
```

#### `predict_next_state()`
Прогноз одного шага с учетом управлений (для сценариев)

```python
predictions = twin.predict_next_state(
    current_state=state_tensor,
    controls=controls
)
```

#### `update_from_sensors()`
Обновление состояния из измерений сенсоров (использует Compressive Sensing).

```python
result = twin.update_from_sensors(
    sensor_readings: torch.Tensor,
    timestamp: Optional[float] = None
)
```

---

## Примеры использования

### Пример 1: Базовый workflow

```python
import torch
from TBMD.digital_twin.digital_twin import DigitalTwin
from TBMD.config import DigitalTwinConfig

# 1. Конфигурация
config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_temporal_modes=20,
    n_sensors=30,
    forecaster_type='linear'
)

# 2. Создание
twin = DigitalTwin(config)

# 3. Обучение
twin.train(historical_data)

# 4. Прогноз
forecast = twin.predict(current_state, n_steps=20)
```

### Пример 2: Интеграция с Brugge данными

См. полный пример: `algorithm/examples/applications/brugge_field/run_brugge_enhanced.py`

---

## API Reference

### Классы данных

#### DigitalTwinConfig
```python
@dataclass
class DigitalTwinConfig(BaseConfig):
    n_spatial_modes: int = 40
    n_temporal_modes: int = 20
    n_sensors: int = 30
    forecaster_type: str = 'lstm'
    proxy_model_type: Optional[str] = None
    forecaster_config: Dict[str, Any] = {...}
    proxy_config: Dict[str, Any] = {...}
    train_test_split: float = 0.8
    validation_split: float = 0.2
    batch_size: int = 32
    epochs: int = 300
    early_stopping_patience: int = 20
```

---

## FAQ

### Q: Какие данные нужны для обучения?

**A**: Минимально:
- Временной ряд полей (nx, ny, [nz,] nt)

### Q: Как выбрать количество мод?

**A**: 
1. Начните с `n_spatial_modes = 30-50`
2. Проверьте reconstruction error после обучения
3. Если error > 0.1, увеличьте количество мод
4. Обычно `n_temporal_modes = n_spatial_modes / 2`

### Q: Можно ли использовать GPU?

**A**: Да! Установите `device='cuda'` в конфиге

---

## Дополнительные ресурсы

### Документация
- [TBMD Overview](tbmd_core.md)
- [Geometry-Aware TBMD](GEOMETRY_AWARE_TBMD.md)
- [Основной README](../README.md)

### Примеры
- Базовый: `algorithm/examples/digital_twin/01_digital_twin_basic.py`
- Brugge: `algorithm/examples/applications/brugge_field/run_brugge_enhanced.py`

---

**Версия**: 2.0.0
**Дата**: Январь 2026
**Автор**: TBMD Team
