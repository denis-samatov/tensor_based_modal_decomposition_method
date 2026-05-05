# Руководство: Создание цифрового двойника с нуля

В этом руководстве мы шаг за шагом создадим цифровой двойник для синтетического месторождения, обучим его и выполним прогноз.

## 📋 Предварительные требования

- Установленный Python 3.8+
- Установленные зависимости (`pip install -r requirements.txt`)
- Базовое понимание тензоров (NumPy/PyTorch)

---

## Шаг 1: Подготовка данных

Для начала нам нужны данные. В реальном проекте это будут результаты гидродинамического симулятора (Eclipse, tNavigator). Для урока мы сгенерируем синтетические данные.

```python
import torch
import numpy as np

def generate_synthetic_field(nx=50, ny=50, nt=100):
    """Генерация поля давления с затухающей волной"""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 10, nt)
    
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Физика: затухающая волна от центра
    r = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    field = np.exp(-0.1*T) * np.sin(10*r - T)
    
    return torch.from_numpy(field).float()

# Создаем данные
data = generate_synthetic_field()
print(f"Размер данных: {data.shape}")  # (50, 50, 100)
```

## Шаг 2: Конфигурация Digital Twin

Настроим параметры двойника.

```python
from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin.digital_twin import DigitalTwin

config = DigitalTwinConfig(
    n_spatial_modes=20,    # Количество пространственных мод
    n_temporal_modes=10,   # Количество временных мод
    n_sensors=15,          # Количество сенсоров для размещения
    forecaster_type='linear', # Тип модели прогноза
    device='cpu',
    verbose=True
)

twin = DigitalTwin(config)
```

## Шаг 3: Обучение (Training)

Обучим модель на первых 80 шагах времени.

```python
# Разделение на train/test
train_data = data[..., :80]
test_data = data[..., 80:]

# Запуск обучения
print("Начинаем обучение...")
summary = twin.train(
    historical_data=train_data,
    normalize=False
)

# summary содержит метрики обучения
print(f"Выбрано сенсоров: {summary['n_sensors']}")
if 'qr_error' in summary:
    print(f"Ошибка QR факторизации: {summary['qr_error']:.4f}")
```

## Шаг 4: Прогноз (Forecasting)

Теперь попробуем предсказать следующие 20 шагов.

```python
# Текущее состояние (последний шаг обучения)
current_state = train_data[..., -1]

# Прогноз
# predict_next_state используется для одного шага с controls
# Для многошагового прогноза используем predict
forecast = twin.predict(
    current_state=current_state,
    n_steps=20
)

print(f"Сгенерирован прогноз формы: {forecast.shape}")
```

## Шаг 5: Валидация и Мониторинг

Сравним прогноз с реальными данными (test set).

```python
import matplotlib.pyplot as plt

# Берем последний прогноз
# forecast имеет форму (spatial..., n_steps)
if forecast.ndim > 2:
    predicted_field = forecast[..., -1]
else:
    predicted_field = forecast

true_field = test_data[..., -1]

# Визуализация
plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.title("Прогноз")
plt.imshow(predicted_field)
plt.colorbar()

plt.subplot(132)
plt.title("Истина")
plt.imshow(true_field)
plt.colorbar()

plt.subplot(133)
plt.title("Ошибка")
plt.imshow(torch.abs(predicted_field - true_field))
plt.colorbar()

plt.show()
```

## Шаг 6: Работа с сенсорами

Симулируем получение данных с датчиков и обновление состояния.

```python
# Получаем маску сенсоров (spatial mask)
sensor_mask = twin.sensor_mask

# Симулируем "реальные" измерения в момент t=90
t_idx = 10 # 80 + 10 = 90
real_field_t90 = test_data[..., t_idx]

# Извлекаем значения только в точках сенсоров
# В реальности это придет с физических датчиков
sensor_readings = real_field_t90[sensor_mask]

# Обновляем состояние двойника
update_result = twin.update_from_sensors(
    sensor_readings=sensor_readings,
    timestamp=90.0
)

print(f"Статус системы: {update_result['alert_status']}")
if update_result['sensor_errors']:
    print(f"Ошибка восстановления по сенсорам: {update_result['sensor_errors'][-1]:.4f}")
```

---

## Что дальше?

- Попробуйте изменить `n_spatial_modes` и посмотрите на ошибку.
- Используйте `forecaster_type='lstm'` для более сложных данных.
- Изучите [Geometry-Aware TBMD](../guides/geometry_aware_tbmd.md) для работы с реальными картами.
