# Advanced TBMD Examples

Продвинутые примеры и legacy код для опытных пользователей

## 📋 Список примеров

### Продвинутые техники

#### 01. Modal Tensor Processing (`01_modal_tensor_processing.py`)
**Описание**: Продвинутая обработка модальных тензоров  
**Уровень**: Advanced  
**Основные концепции**:
- Modal analysis
- Tensor algebra
- Mode selection
- Energy analysis

#### 02. Unified Experiments (`02_unified_experiments.py`)
**Описание**: Комплексные эксперименты на реальных данных  
**Уровень**: Advanced  
**Основные концепции**:
- Batch processing
- Multiple datasets
- Comparative analysis
- Performance metrics

#### 03. Sensor Values Fix (`03_sensor_values_fix.py`)
**Описание**: Коррекция и валидация данных сенсоров  
**Уровень**: Intermediate  
**Основные концепции**:
- Data validation
- Outlier detection
- Missing data imputation
- Sensor calibration

### Legacy Examples

> ⚠️ **Примечание**: Эти примеры используют старый API и сохранены для обратной совместимости.  
> Для новых проектов рекомендуется использовать примеры из `basic/`.

#### 04. Legacy TBMD (`04_legacy_tbmd.py`)
**Описание**: Базовый TBMD пример (старый API)  
**Статус**: Legacy  
**Рекомендуется**: `basic/01_tucker_decomposition.py`

#### 05. Legacy QR (`05_legacy_qr.py`)
**Описание**: QR sensor placement (старый API)  
**Статус**: Legacy  
**Рекомендуется**: `basic/02_sensor_placement.py`

#### 06. Legacy CS (`06_legacy_cs.py`)
**Описание**: Compressive sensing (старый API)  
**Статус**: Legacy  
**Рекомендуется**: `basic/03_field_reconstruction.py`

#### 07. Legacy Pipeline (`07_legacy_pipeline.py`)
**Описание**: Полный pipeline (старый API)  
**Статус**: Legacy  
**Рекомендуется**: `basic/04_complete_pipeline.py`

## 🎯 Для кого эти примеры

### Продвинутые техники (01-03)
- Исследователи, разрабатывающие новые методы
- Пользователи, работающие с реальными сложными данными
- Разработчики, оптимизирующие производительность

### Legacy примеры (04-07)
- Пользователи старых версий TBMD
- Проекты, требующие обратной совместимости
- Понимание эволюции API

## 🔄 Миграция с Legacy на новый API

### Старый способ (Legacy):
```python
from TBMD.modules.TensorHOSVD import TuckerDecomposer

decomposer = TuckerDecomposer(device='cpu', verbose=True)
result = decomposer.decompose(data, n_modes=20)
spatial_modes = result['spatial_modes']
```

### Новый способ (Recommended):
```python
from TBMD.config import DecompositionConfig
from TBMD.core.decomposition import TuckerDecomposer

config = DecompositionConfig(
    ranks=[20, 10],
    device='cpu',
    verbose=True
)
decomposer = TuckerDecomposer(config)
result = decomposer.decompose(data)
spatial_modes = result.spatial_modes
```

## 📊 Сравнение Legacy vs New API

| Аспект | Legacy API | New API |
|--------|------------|---------|
| Конфигурация | В аргументах | Dataclass objects |
| Результаты | Словари | Dataclasses |
| Type hints | Частично | Полностью |
| Тестирование | Минимальное | Comprehensive |
| Документация | Базовая | Полная |
| Поддержка | ⚠️ Deprecated | ✅ Active |

## 🛠️ Использование продвинутых техник

### Modal Tensor Processing
```python
from TBMD.core.decomposition import TuckerDecomposer

# Decompose
decomposer = TuckerDecomposer(config)
result = decomposer.decompose(data)

# Analyze modes
mode_energies = analyze_mode_energies(result.spatial_modes)
select_important_modes(mode_energies, threshold=0.95)
```

### Batch Processing
```python
# Process multiple datasets
datasets = load_all_datasets()

results = {}
for name, data in datasets.items():
    result = pipeline.process(data)
    results[name] = evaluate_metrics(result)

compare_results(results)
```

### Sensor Data Validation
```python
# Validate and fix sensor data
sensor_data = load_sensor_data()
validated = validate_sensors(sensor_data)
fixed = fix_outliers(validated)
calibrated = calibrate_sensors(fixed)
```

## 📚 Дополнительные ресурсы

### Для продвинутых техник:
- [Core Modules Guide](../../docs/guides/TBMD_CORE_MODULES.md)
- [Utils Guide](../../docs/guides/TBMD_UTILS.md)
- [API Reference](../../docs/api/API_REFERENCE.md)

### Для миграции:
- [Migration Guide](../../../NOTEBOOK_MIGRATION_GUIDE.md)
- [Quick Start with New API](../../docs/guides/QUICK_START.md)

## ⚠️ Важные замечания

### О Legacy примерах:
1. **Не рекомендуется** для новых проектов
2. **Будут удалены** в будущих версиях
3. **Могут содержать** устаревшие паттерны
4. **Сохранены только** для совместимости

### О продвинутых техниках:
1. Требуют глубокого понимания TBMD
2. Могут быть нестабильными
3. Предназначены для экспериментов
4. Не все функции документированы

## 🚀 Вклад

Если вы разработали продвинутую технику:
1. Убедитесь, что она хорошо документирована
2. Добавьте тесты
3. Создайте pull request
4. Опишите use case

## 📞 Поддержка

Для продвинутых вопросов:
- Изучите исходный код в `TBMD/core/`
- Обратитесь к исследовательским статьям
- Создайте detailed issue
- Свяжитесь с мейнтейнерами напрямую

