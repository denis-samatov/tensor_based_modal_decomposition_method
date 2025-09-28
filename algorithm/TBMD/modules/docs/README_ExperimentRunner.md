# ExperimentRunner: Унифицированная система анализа TBMD

## 🎯 Ключевое обновление: Унифицированный метод `run_experiments`

Вместо двух запутывающих методов (`run_standard_experiments` и `run_full_dataset_experiments`) теперь есть **один универсальный метод** с автоматическим определением режима!

## 🔄 До и После

### ❌ **Было: Путаница с двумя методами**
```python
# Непонятно, какой использовать и в чем разница
df1 = runner.run_standard_experiments(...)      # Простые значения
df2 = runner.run_full_dataset_experiments(...)  # Со статистикой
```

### ✅ **Стало: Один понятный метод**
```python
# Автоматическое определение режима по конфигурации
df = runner.run_experiments(A_tensor, test_tensors, sensor_values)

# Или явное управление
df_fast = runner.run_experiments(..., statistical_analysis=False)    # Быстро
df_precise = runner.run_experiments(..., statistical_analysis=True)  # Точно
```

## Автоматическое определение режима

```python
# Конфигурация без шума → простой режим
config1 = ExperimentConfig(noise_level=0.0)
runner1 = ExperimentRunner(config1)
df1 = runner1.run_experiments(...)
# Результат: ['sensors', 'error', 'ssim', 'psnr']

# Конфигурация с шумом → статистический режим  
config2 = ExperimentConfig(noise_level=0.1, num_noise_samples=5)
runner2 = ExperimentRunner(config2)
df2 = runner2.run_experiments(...)
# Результат: ['sensors', 'error_mean', 'error_std', 'error_ci_lower', 'error_ci_upper', ...]
```

## Структура классов

### ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    # Основные параметры
    solver_method: str = "triangular"
    seed: int = SEED
    device: str = 'cpu'
    
    # Параметры компрессивного зондирования
    max_iter: int = 1000
    epsilon: float = 1e-2
    lambd: float = 0.95
    delta_0: float = 0.1
    delta_max: float = 1.0
    
    # Параметры шума для автоматического определения режима
    noise_level: float = 0.0        # > 0 → статистический режим
    num_noise_samples: int = 0      # > 0 → статистический режим
    noise_threshold: float = 1e-10  # Порог для определения "нулевых" значений при зашумлении
    
    # Параметры анализа
    confidence_level: float = 0.95
    subject_axis: bool = False
    
    # Валидационные параметры
    valid_mask: Optional[np.ndarray] = None
    wells: Optional[Dict[str, List[Tuple[int, int]]]] = None
    
    # Параметры вывода
    verbose: bool = True
```

### ExperimentRunner

```python
class ExperimentRunner:
    """Унифицированная система экспериментов для TBMD анализа."""
    
    def __init__(self, config: ExperimentConfig = None)
    
    # 🎯 ГЛАВНЫЙ УНИВЕРСАЛЬНЫЙ МЕТОД
    def run_experiments(self, A_tensor, test_tensors, sensor_values, 
                       statistical_analysis=None) -> pd.DataFrame
    
    # Специализированные методы
    def run_single_slice_experiments(self, A_tensor, test_tensors, subject_name, slice_idx, sensor_values) -> pd.DataFrame
    def run_wells_experiments(self, A_tensor, test_tensors, sensor_values, slice_idx) -> pd.DataFrame
    
    # Алиасы для совместимости  
    def run_full_dataset_experiments(self, A_tensor, test_tensors, sensor_values) -> pd.DataFrame  # = run_experiments(..., statistical_analysis=True)
    def run_standard_experiments(self, A_tensor, test_tensors, sensor_values) -> pd.DataFrame     # = run_experiments(..., statistical_analysis=False)
```

## Примеры использования

### 1. Автоматический режим (рекомендуется)

```python
from TBMD.utils.analytics import ExperimentRunner, ExperimentConfig

# Простая конфигурация → автоматически простой режим
config_simple = ExperimentConfig(device='cpu')
runner = ExperimentRunner(config_simple)
df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
# Результат: простые агрегированные метрики

# Конфигурация с шумом → автоматически статистический режим
config_statistical = ExperimentConfig(
    device='cpu',
    noise_level=0.1,
    num_noise_samples=5,
    confidence_level=0.95
)
runner = ExperimentRunner(config_statistical)
df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
# Результат: детальная статистика с доверительными интервалами
```

### 2. Явное управление режимом

```python
config = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=10)
runner = ExperimentRunner(config)

# Принудительно простой режим (быстро, без статистики)
df_fast = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                statistical_analysis=False)

# Принудительно статистический режим (медленно, с доверительными интервалами)
df_precise = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                  statistical_analysis=True)

# Автоматический режим (использует конфигурацию: noise_level > 0 → статистический)
df_auto = runner.run_experiments(A_tensor, test_tensors, sensor_values)
```

### 3. Анализ одного среза

```python
config = ExperimentConfig(noise_level=0.05, num_noise_samples=3)
runner = ExperimentRunner(config)

df_slice = runner.run_single_slice_experiments(
    A_tensor, test_tensors, subject_name='subject_1', slice_idx=0, sensor_values=sensor_values
)
```

### 4. Анализ по скважинам

```python
wells_config = {
    'subject_1': [(10, 10), (20, 20), (30, 30)],
    'subject_2': [(15, 15), (25, 25), (35, 35)]
}

config = ExperimentConfig(wells=wells_config, subject_axis=True)
runner = ExperimentRunner(config)

df_wells = runner.run_wells_experiments(A_tensor, test_tensors, sensor_values, slice_idx=0)
```

### 5. Построение графиков

```python
from TBMD.utils.analytics import plot_analytics

# Получаем данные (автоматический режим)
config = ExperimentConfig(noise_level=0.1, num_noise_samples=5)
runner = ExperimentRunner(config)
df = runner.run_experiments(A_tensor, test_tensors, sensor_values)

# 🎯 НОВАЯ УЛУЧШЕННАЯ ФУНКЦИЯ plot_analytics
# Поддерживает 4 типа графиков: individual, combined, normalized, all

# Индивидуальные графики для каждой метрики
plot_analytics(df, plot_type='individual', title_prefix="TBMD Individual")

# Нормализованный график (Error инвертирован + SSIM)
plot_analytics(df, metrics=['error', 'ssim'], plot_type='normalized')

# Комбинированный график всех метрик
plot_analytics(df, plot_type='combined')

# Все типы графиков сразу + сохранение
plot_analytics(df, plot_type='all', save_path="tbmd_complete_analysis")
```

## Структура DataFrame

### Простой режим
```
| sensors | error | mse   | ssim  | psnr  |
|---------|-------|-------|-------|-------|
| 5       | 0.123 | 0.015 | 0.856 | 18.2  |
| 10      | 0.089 | 0.008 | 0.912 | 21.5  |
```

### Статистический режим (с доверительными интервалами)
```
| sensors | error_mean | error_std | ssim_mean | psnr_mean | error_ci_lower | error_ci_upper | num_samples |
|---------|------------|-----------|-----------|-----------|----------------|----------------|-------------|
| 5       | 0.145      | 0.023     | 0.834     | 17.1      | 0.131          | 0.159          | 50          |
| 10      | 0.102      | 0.018     | 0.889     | 20.3      | 0.092          | 0.112          | 50          |
```

## Обратная совместимость

Старые методы продолжают работать:

```python
# Все эти вызовы эквивалентны
df1 = runner.run_experiments(A_tensor, test_tensors, sensor_values, statistical_analysis=False)
df2 = runner.run_standard_experiments(A_tensor, test_tensors, sensor_values)  # Алиас

# Эти тоже эквивалентны
df3 = runner.run_experiments(A_tensor, test_tensors, sensor_values, statistical_analysis=True)
df4 = runner.run_full_dataset_experiments(A_tensor, test_tensors, sensor_values)  # Алиас
```

## Рекомендации по использованию

### 🎯 **Главное правило: Используйте `run_experiments()`**

```python
# ✅ Рекомендуется
df = runner.run_experiments(A_tensor, test_tensors, sensor_values)

# ⚠️ Работает, но устарело
df = runner.run_standard_experiments(A_tensor, test_tensors, sensor_values)
df = runner.run_full_dataset_experiments(A_tensor, test_tensors, sensor_values)
```

### 🎛️ **Выбор режима:**

1. **Автоматический (по умолчанию)** - настройте `noise_level`/`num_noise_samples` в конфигурации
2. **Быстрый анализ** - `statistical_analysis=False`
3. **Точный анализ** - `statistical_analysis=True`

### ⚡ **Производительность:**

- **Простой режим**: ~1x скорость, базовые метрики
- **Статистический режим**: ~5-10x медленнее, полная статистика

### 📊 **Совместимость с графиками:**

Функция `plot_analytics` автоматически определяет формат данных и строит соответствующие графики.

## Расширенные возможности plot_analytics

### Типы графиков

#### 1. `individual` - Индивидуальные графики
- Отдельный график для каждой метрики
- Полная детализация с доверительными интервалами

#### 2. `normalized` - Нормализованный график  
- **Error инвертирован** (выше = лучше качество)
- SSIM нормализован к диапазону [0,1]
- Позволяет сравнивать разнонаправленные метрики

#### 3. `combined` - Комбинированный график
- Все метрики на одном графике
- Сохраняет оригинальные значения

#### 4. `all` - Все типы графиков
- Создает все вышеперечисленные графики
- Автоматическое сохранение с суффиксами

### Гибкость форматов данных

Функция автоматически определяет формат DataFrame и работает с обоими режимами.

## 🛢️ Умное зашумление для резервуарных данных

### Проблема традиционного зашумления

В резервуарном моделировании **нулевые значения имеют физический смысл** - они означают отсутствие флюида или породу. Добавление шума к таким значениям может исказить интерпретацию данных.

### ❌ Старый подход (некорректный)
```python
# Зашумляет ВСЕ значения, включая нули
noise = torch.randn_like(Y) * noise_level
noisy_Y = Y + noise  # Некорректно для резервуарных данных!
```

### ✅ Новый умный подход
```python
# Зашумляет ТОЛЬКО ненулевые значения
non_zero_mask = torch.abs(Y) > noise_threshold
noisy_Y = Y.clone()
noisy_Y[non_zero_mask] = Y[non_zero_mask] + noise[non_zero_mask]
```

### Конфигурация умного зашумления

```python
config = ExperimentConfig(
    noise_level=0.1,           # Уровень шума (10%)
    num_noise_samples=5,       # Количество зашумленных образцов
    noise_threshold=1e-10      # Порог для определения "нулевых" значений
)
```

**Параметр `noise_threshold`:**
- `1e-10` (по умолчанию): Стандартный выбор для численной стабильности
- `1e-6`: Для данных с очень малыми значениями  
- `0.1`: Для зашумления только значительных значений

### Демонстрация

Запустите полную демонстрацию:
```bash
cd algorithm/TBMD/examples
python noise_demonstration.py
```

### Преимущества умного зашумления

| Аспект | Старый подход | Умный подход |
|--------|---------------|--------------|
| **Физический смысл** | ❌ Нарушается | ✅ Сохраняется |
| **Нулевые значения** | ❌ Зашумляются | ✅ Остаются нулевыми |
| **Статистика** | ❌ Искажается | ✅ Корректная |
| **Резервуарная интерпретация** | ❌ Затруднена | ✅ Сохранена |

## Заключение

Новая архитектура `ExperimentRunner` + унифицированный `run_experiments` + улучшенная `plot_analytics` + умное зашумление предоставляет:

1. **🎯 Единый интерфейс**: Один метод для всех базовых экспериментов
2. **🤖 Автоматизация**: Умное определение режима анализа
3. **⚡ Гибкость**: Явное управление через параметры
4. **🔄 Совместимость**: Полная обратная совместимость
5. **📊 Мощная визуализация**: 4 типа графиков с адаптацией логики из plots.py
6. **📈 Статистическая корректность**: Автоматические доверительные интервалы
7. **🚀 Простота использования**: Минимум параметров для начала работы
8. **🛢️ Физически корректное зашумление**: Сохранение смысла нулевых значений

Это устраняет путаницу между `run_standard_experiments` и `run_full_dataset_experiments`, предоставляя понятный и мощный интерфейс для всех типов анализа TBMD! 