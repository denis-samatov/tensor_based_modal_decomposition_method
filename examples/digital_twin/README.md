# Digital Twin Examples

Примеры создания и использования цифровых двойников с TBMD

## 📋 Список примеров

### 01. Digital Twin Basic (`01_digital_twin_basic.py`)
**Описание**: Базовый пример создания digital twin  
**Уровень**: Beginner  
**Основные концепции**:
- Инициализация Digital Twin
- Обучение на исторических данных
- Прогнозирование
- Реконструкция из сенсоров
- Детекция аномалий
- Сценарный анализ

**Использование**:
```bash
python 01_digital_twin_basic.py
```

### 02. Digital Twin Advanced (`02_digital_twin_advanced.py`)
**Описание**: Продвинутые возможности digital twin  
**Уровень**: Advanced  
**Основные концепции**:
- Полный lifecycle Digital Twin
- Real-time monitoring
- Model calibration
- Online learning
- Advanced forecasting
- Multi-scenario analysis

**Использование**:
```bash
python 02_digital_twin_advanced.py
```

## 🎯 Основные компоненты Digital Twin

### 1. Обучение (Training)
```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin import DigitalTwin

config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_sensors=30
)
twin = DigitalTwin(config)
twin.train(historical_data, normalize=True)
```

### 2. Прогнозирование (Forecasting)
```python
# Прогноз на 10 шагов вперед
forecast = twin.predict(
    current_state=current_state,
    n_steps=10,
    return_full_field=True
)
```

### 3. Мониторинг (Monitoring)
```python
# Реконструкция из sensor measurements
reconstructed = twin.update_from_sensors(
    sensor_readings=sensor_data
)
```

### 4. Детекция аномалий (Anomaly Detection)
```python
# Обнаружение отклонений
anomalies = twin.detect_anomalies(
    sensor_data=sensor_time_series,
    threshold=3.0
)
```

### 5. Сценарный анализ (Scenario Analysis)
```python
# Оценка разных сценариев
scenarios = [
    {'name': 'baseline'},
    {'name': 'optimistic'},
    {'name': 'pessimistic'}
]
results = twin.evaluate_scenarios(scenarios, n_steps=10)
```

## 🔄 Типичный workflow

```
1. Historical Data
   ↓
2. Train Digital Twin
   ├─ Tucker Decomposition
   ├─ Sensor Placement
   └─ Calibration
   ↓
3. Real-time Operation
   ├─ Monitor (sensor data)
   ├─ Forecast (future states)
   ├─ Detect anomalies
   └─ Evaluate scenarios
   ↓
4. Decision Making
```

## 📊 Use Cases

### 1. Reservoir Monitoring
```python
# Мониторинг месторождения
twin = DigitalTwin(config)
twin.train(reservoir_history)

# Real-time monitoring
while monitoring:
    sensor_data = read_well_sensors()
    current_field = twin.update_from_sensors(sensor_data)
    
    if check_for_issues(current_field):
        forecast = twin.predict(current_field, n_steps=30)
        alert_operators(forecast)
```

### 2. Predictive Maintenance
```python
# Прогнозирование для предупреждения проблем
forecast = twin.predict(current_state, n_steps=100)

# Проверить критические метрики
if forecast_shows_problem(forecast):
    schedule_maintenance()
```

### 3. What-If Analysis
```python
# Оценка последствий решений
scenarios = [
    {'name': 'increase_production', 'rate': 1.2},
    {'name': 'decrease_production', 'rate': 0.8},
    {'name': 'status_quo', 'rate': 1.0}
]

results = twin.evaluate_scenarios(scenarios, n_steps=365)
best_scenario = select_optimal(results)
```

## ⚙️ Конфигурация

### Базовая
```python
config = DigitalTwinConfig(
    n_spatial_modes=40,
    n_temporal_modes=20,
    n_sensors=30,
    verbose=True
)
```

### Продвинутая
```python
config = DigitalTwinConfig(
    n_spatial_modes=60,
    n_temporal_modes=30,
    n_sensors=50,
    solver='admm',
    max_iterations=200,
    backend='torch',
    device='cuda',
    verbose=True
)
```

## 📈 Метрики производительности

| Операция | Данные (I×J×T) | Время | Точность |
|----------|----------------|-------|----------|
| Training | 200×3×100 | ~5s | - |
| Forecast (10 steps) | - | <0.1s | <5% error |
| Sensor Update | 30 sensors | ~1s | <3% error |
| Anomaly Detection | 30×100 | ~2s | 95% TPR |

## 🔧 Расширение функциональности

### Добавить custom forecasting model
```python
class CustomTwin(DigitalTwin):
    def predict(self, current_state, n_steps):
        # Ваша модель прогнозирования
        return custom_forecast
```

### Добавить custom anomaly detection
```python
def custom_anomaly_detector(twin, data):
    # Ваш алгоритм детекции
    return anomalies
```

## 📚 Дополнительные ресурсы

- [Digital Twin Tutorial](../../docs/tutorials/DIGITAL_TWIN_README.md)
- [API Reference - DigitalTwin](../../docs/api/API_REFERENCE.md#digitaltwin)
- [Quick Start](../../docs/guides/QUICK_START.md#5-digital-twin)

## ⚠️ Ограничения

1. **Статический подход**: Текущая версия не поддерживает online retraining
2. **Простые модели**: Прогнозирование based on persistence
3. **Масштабируемость**: Может быть медленным для очень больших систем

## 🚀 Будущие улучшения

- [ ] Online learning и model updating
- [ ] ML-based forecasting (LSTM, GRU)
- [ ] Multi-fidelity models
- [ ] Uncertainty quantification
- [ ] Distributed computing support

