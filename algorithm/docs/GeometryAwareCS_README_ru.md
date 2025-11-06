# Geometry-Aware Compressive Sensing (Компрессивное восстановление с учётом геометрии)

## Обзор

`GeometryAwareTensorCS` — это расширение стандартного тензорного компрессивного восстановления (Algorithm 3) с добавлением **лапласовой регуляризации** для получения пространственно гладких решений на неструктурированных сетках.

### Зачем это нужно?

При восстановлении физических полей (температура, давление, скорость) из неполных измерений:
- **Стандартный CS**: восстанавливает разреженные коэффициенты, но **не гарантирует** пространственную гладкость
- **Geometry-aware CS**: добавляет регуляризацию через граф-Лапласиан → физически реалистичные, **гладкие** поля

## Математическая формулировка

### Стандартный CS (без геометрии)
```
min ||Ax - y||² + ε||d||₁
```

### Geometry-aware CS (с геометрией)
```
min ||Ax - y||² + ε||d||₁ + α||L·x||²
```

где:
- **A**: матрица модальных форм (forward model)
- **x**: коэффициенты для восстановления
- **y**: измерения на датчиках
- **d**: вспомогательная переменная для L1-штрафа
- **L**: граф-Лапласиан сетки
- **α**: сила регуляризации (чем больше → тем глаже)
- **ε**: параметр разреженности

## Использование

### Базовый пример

```python
import numpy as np
from TBMD.modules import GeometryAwareTensorCS, GeometryAwareCSConfig
from TBMD.utils.geometry import MeshGraphBuilder

# 1. Построить сетку
builder = MeshGraphBuilder(connectivity_type='grid')
mesh = builder.build_from_shape((100, 100))  # 100x100 сетка

# 2. Настроить параметры
config = GeometryAwareCSConfig(
    alpha=0.1,           # Сила регуляризации Лапласиана
    epsilon_l1=1e-2,     # Порог разреженности
    max_iter=500,        # Максимум итераций
    tol=1e-4,            # Критерий сходимости
    laplacian_type='normalized',  # Тип Лапласиана
    auto_alpha=True      # Автоматическая настройка α
)

# 3. Создать solver
solver = GeometryAwareTensorCS(
    A=mode_shapes,       # (n_cells, n_modes)
    P=sensor_mask,       # (n_cells,) bool маска
    Y=measurements,      # (n_cells,) измерения
    mesh=mesh,           # Геометрия сетки
    core_cfg=config
)

# 4. Решить
x_recovered, metrics = solver.solve()

print(f"Восстановлено за {metrics.iterations} итераций")
print(f"Сошлось: {metrics.converged}")
print(f"Время: {metrics.time_sec:.2f}с")
```

### Сравнение со стандартным CS

```python
from TBMD.modules import TensorCompressiveSensing, CompressiveSensingConfig

# Стандартный CS (без геометрии)
config_std = CompressiveSensingConfig(epsilon_l1=1e-2)
solver_std = TensorCompressiveSensing(A, P, Y, core_cfg=config_std)
x_std, _ = solver_std.solve()

# Geometry-aware CS
config_geo = GeometryAwareCSConfig(alpha=0.1, epsilon_l1=1e-2)
solver_geo = GeometryAwareTensorCS(A, P, Y, mesh, core_cfg=config_geo)
x_geo, _ = solver_geo.solve()

# Geometry-aware версия даст более гладкое решение!
```

## Параметры конфигурации

### `GeometryAwareCSConfig`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `alpha` | 0.01 | Сила лапласовой регуляризации. ↑ → глаже |
| `laplacian_type` | 'normalized' | Тип Лапласиана: 'standard' или 'normalized' |
| `auto_alpha` | True | Автоматическая настройка α по спектральной норме |
| `alpha_max` | 1.0 | Максимальное значение α при auto_alpha=True |
| `epsilon_l1` | 0.01 | Порог L1-разреженности |
| `max_iter` | 1000 | Максимум ADMM итераций |
| `tol` | 1e-4 | Критерий сходимости |
| `delta_init` | 1.0 | Начальное значение ADMM штрафа δ |
| `relax_lambda` | 0.95 | Over-relaxation параметр |

## Когда использовать geometry-aware версию?

### ✅ **Используйте** когда:
1. **Физические поля**: температура, давление, скорость в CFD
2. **Неструктурированные сетки**: FEM/FVM сетки со сложной топологией
3. **Мало датчиков**: < 30% покрытие → регуляризация критична
4. **Шумные измерения**: Лапласиан действует как denoise-фильтр
5. **Нужна гладкость**: требования к физической реалистичности

### ❌ **Не используйте** когда:
1. **Не физические данные**: изображения, временные ряды без пространственной структуры
2. **Много датчиков**: > 70% покрытие → стандартный CS достаточно хорош
3. **Резкие фронты**: ударные волны, разрывы → Лапласиан их сгладит!

## Архитектура системы

Теперь у вас **полный набор geometry-aware методов**:

```
Алгоритм                        | Стандарт                        | Geometry-aware
--------------------------------|---------------------------------|----------------------------------
Tucker декомпозиция (HOSVD)     | TuckerDecomposer               | GeometryAwareTuckerDecomposer
Размещение датчиков (QR)        | TensorTubeQRDecomposition      | GeometryAwareTensorQR
Восстановление из измерений (CS)| TensorCompressiveSensing       | GeometryAwareTensorCS ← НОВОЕ!
```

## Примеры

### Пример 1: Восстановление температурного поля

```python
# Измерения температуры на 20% сетки
temperature_field = ...  # CFD симуляция
sensor_locations = ...    # 20% ячеек

# Выполнить POD/Tucker для получения базисных функций
from TBMD.modules import GeometryAwareTuckerDecomposer
decomposer = GeometryAwareTuckerDecomposer(
    tensor=temperature_field,
    mesh=mesh,
    ranks=[50, 50, 100]
)
decomposer.decompose()
A = decomposer.factors[0]  # Пространственные моды

# Восстановить из неполных измерений
solver = GeometryAwareTensorCS(A, sensor_mask, measurements, mesh)
coefficients, metrics = solver.solve()

# Реконструировать поле
reconstructed = A @ coefficients
```

### Пример 2: Автоматический подбор α

```python
# auto_alpha=True автоматически балансирует:
# - Data fidelity term: ||Ax - y||²
# - Smoothness term: α||Lx||²

config = GeometryAwareCSConfig(
    auto_alpha=True,     # Включить автонастройку
    alpha=0.01,          # Начальное значение
    alpha_max=1.0        # Максимальная граница
)

solver = GeometryAwareTensorCS(A, P, Y, mesh, core_cfg=config)
x, metrics = solver.solve()

# α выбирается автоматически на основе:
# α ≈ ||A^T A|| / ||L^T L A A^T L^T L||
```

## Полный рабочий пример

Запустите:
```bash
python TBMD/examples/geometry_aware_cs_example.py
```

Это создаст:
- Синтетическую задачу с гладким полем
- Сравнение стандартного и geometry-aware CS
- Визуализацию восстановленных полей
- Графики сходимости

## Ссылки

- **Algorithm 3**: Tensor-based compressive sensing via ADMM (ваша статья)
- **Boyd et al. (2011)**: "Distributed Optimization and Statistical Learning via ADMM"
- **Jiang et al. (2017)**: "Smooth Tucker decomposition" — регуляризация Лапласианом

## Совместимость

Работает с:
- ✅ `GeometryAwareTuckerDecomposer` — используйте одну и ту же сетку
- ✅ `GeometryAwareTensorQR` — сначала разместите датчики с QR, затем восстанавливайте с CS
- ✅ PyTorch GPU acceleration — установите `device='cuda'`
- ✅ Sparse matrices — Лапласиан автоматически обрабатывается как разреженная матрица

## FAQ

**Q: Какое значение α выбрать?**  
A: Начните с `auto_alpha=True`. Если нужен ручной контроль: α ∈ [0.001, 0.1] для большинства задач.

**Q: Медленнее ли geometry-aware версия?**  
A: На ~20-30% из-за доп. матричного умножения `A^T L^T L A`, но результаты физически корректнее.

**Q: Можно ли использовать с неструктурированными сетками?**  
A: Да! Это главное преимущество — `MeshGraphBuilder` строит граф для любой топологии.

**Q: Что делать с ударными волнами/разрывами?**  
A: Уменьшите α или используйте стандартный CS. Лапласиан сглаживает разрывы.

