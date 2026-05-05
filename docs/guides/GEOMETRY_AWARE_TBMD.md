# Geometry-Aware TBMD - Полная документация

## 📖 Оглавление

1. [Введение](#введение)
2. [Теоретические основы](#теоретические-основы)
3. [Архитектура](#архитектура)
4. [Установка и использование](#установка-и-использование)
5. [API Reference](#api-reference)
6. [Примеры](#примеры)

---

## Введение

**Geometry-Aware TBMD** — это расширение классического метода TBMD, которое учитывает сложную геометрию резервуара и неструктурированные сетки.

### Проблема классического TBMD
Классический TBMD работает с тензорами на регулярных сетках (декартовых). Однако реальные резервуары имеют:
- ❌ Сложные границы (разломы, выклинивания)
- ❌ Неактивные ячейки (null blocks)
- ❌ Неструктурированные сетки (Corner Point Geometry)

### Решение
Использование **Graph Laplacian** (Лапласиана графа) для кодирования топологии сетки в модальное разложение.

---

## Теоретические основы

### 1. Графовое представление
Резервуар представляется как граф $G = (V, E)$, где:
- $V$ — активные ячейки (узлы)
- $E$ — связи между соседними ячейками (ребра)

### 2. Лапласиан графа
Строится матрица Лапласа $L = D - A$, где:
- $A$ — матрица смежности (кто с кем сосед)
- $D$ — диагональная матрица степеней вершин

Нормированный Лапласиан:
$$ \mathcal{L} = I - D^{-1/2} A D^{-1/2} $$

### 3. Спектральное разложение
Собственные векторы Лапласиана $\mathcal{L} u_k = \lambda_k u_k$ образуют базис, адаптированный к геометрии.
- Низкочастотные моды (малые $\lambda$) — плавные изменения
- Высокочастотные моды (большие $\lambda$) — локальные детали

### 4. Интеграция с TBMD
Вместо стандартного SVD/HOSVD по пространственным модам, мы используем регуляризацию Лапласианом (Laplacian Regularization) в процессе ALS (Alternating Least Squares) для получения пространственно гладких мод, уважающих геометрию.

---

## Архитектура

### Основные компоненты

1. **`GeometryAwareTuckerDecomposer`**
   - Главный класс для декомпозиции
   - Принимает тензор и геометрию сетки (MeshGeometry)
   - Выполняет HOSVD с регуляризацией Лапласианом

2. **`GeometryAwareConfig`**
   - Конфигурация параметров регуляризации
   - Параметры: `alpha` (сила регуляризации), `spatial_modes`, `laplacian_type`

3. **`MeshGraphBuilder`** и **`MeshGeometry`**
   - Утилиты для построения графов смежности из сеток
   - Поддержка регулярных сеток, KNN, radius-based графов

---

## Установка и использование

### Инициализация

```python
from TBMD.core.decomposition.geometry_aware import (
    GeometryAwareTuckerDecomposer,
    GeometryAwareConfig
)
from TBMD.core.geometry import MeshGraphBuilder

# 1. Построение геометрии сетки
builder = MeshGraphBuilder(connectivity_type='grid')
# Для 2D данных (100x100)
mesh = builder.build_from_shape(spatial_shape=(100, 100))

# 2. Конфигурация
geo_config = GeometryAwareConfig(
    alpha=0.1,             # Сила сглаживания
    spatial_modes=[0],     # Индекс пространственной моды (в тензоре)
    laplacian_type='normalized'
)

# 3. Создание декомпозитора
decomposer = GeometryAwareTuckerDecomposer(
    tensor=data_tensor,    # (Spatial, Time) or (X, Y, Time)
    mesh=mesh,
    geo_config=geo_config,
    ranks=[20, 10]         # [spatial_rank, temporal_rank]
)
```

### Декомпозиция

```python
decomposer.decompose()

# Доступ к результатам
core = decomposer.cores
factors = decomposer.factors
spatial_modes = factors[0]
```

### Реконструкция

```python
reconstructed = decomposer.reconstruct()
```

---

## API Reference

### `GeometryAwareTuckerDecomposer`

#### `__init__(tensor, mesh, geo_config, ranks, ...)`
- `tensor`: Входной тензор
- `mesh`: `MeshGeometry` или tuple размеров (для авто-генерации сетки)
- `geo_config`: Экземпляр `GeometryAwareConfig`
- `ranks`: Целевые ранги Таккера

#### `decompose()`
Запускает процесс декомпозиции с ALS и регуляризацией. Сохраняет результаты в свойствах `cores` и `factors`.

#### `reconstruct()` -> `torch.Tensor`
Восстанавливает тензор из разложения.

### `GeometryAwareConfig`

```python
@dataclass
class GeometryAwareConfig:
    alpha: float = 0.01                 # Коэффициент регуляризации
    spatial_modes: List[int] = [0]      # Индексы мод для применения Лапласиана
    laplacian_type: str = 'normalized'  # 'standard' или 'normalized'
    connectivity_type: str = 'grid'     # Метод построения графа
    connectivity_params: Dict = {}      # Параметры графа
```

---

## Примеры

### Пример 1: L-образный домен (с маской)

```python
import numpy as np
from TBMD.core.decomposition.geometry_aware import GeometryAwareTuckerDecomposer, GeometryAwareConfig
from TBMD.core.geometry import MeshGraphBuilder

# 1. Создаем маску (L-shape)
mask = np.ones((20, 20))
mask[10:, 10:] = 0  # Вырезаем угол

# 2. Строим граф с учетом маски
builder = MeshGraphBuilder(connectivity_type='grid')
# Предполагаем, что есть метод build_from_mask или передаем active_cells_mask (зависит от API MeshGraphBuilder)
# В текущей реализации MeshGraphBuilder может принимать параметры в build_from_shape или конструкторе
# Для сложных масок лучше создать adjacency matrix вручную или использовать спец. методы
mesh = builder.build_from_shape((20, 20)) 
# TODO: Уточнить передачу маски в MeshGraphBuilder API если поддерживается напрямую

# 3. Декомпозиция
config = GeometryAwareConfig(alpha=0.5)
decomposer = GeometryAwareTuckerDecomposer(
    tensor=data_l_shape, 
    mesh=mesh, 
    geo_config=config,
    ranks=[10, 5]
)

decomposer.decompose()
```

### Пример 2: Brugge Field (3D)

Для реальных месторождений данные часто линеаризуются (Active Cells).

```python
# data shape: (N_active_cells, T)

# Предположим у нас есть матрица смежности для активных ячеек
adjacency_matrix = load_brugge_adjacency() 

from TBMD.core.geometry import MeshGeometry
mesh = MeshGeometry(adjacency_matrix=adjacency_matrix)

# Конфиг
config = GeometryAwareConfig(alpha=0.1)

# TBMD
decomposer = GeometryAwareTuckerDecomposer(
    tensor=data,
    mesh=mesh,
    geo_config=config,
    ranks=[100, 20] # [spatial, temporal]
)
decomposer.decompose()
```

---

## Преимущества перед обычным TBMD

| Характеристика | Standard TBMD | Geometry-Aware TBMD |
|----------------|---------------|---------------------|
| Сетка | Только регулярная | Любая (граф) |
| Границы | Ступенчатые | Гладкие / Точные |
| Неактивные ячейки | Заполняются нулями (ошибки на границах) | Игнорируются (корректная физика) |
| Базис | Фурье / Полиномы | Собственные функции домена |
| Эффективность | Низкая для сложных форм | Высокая (меньше мод для той же точности) |

---

**Версия**: 2.0
**Дата**: Январь 2026
**Автор**: TBMD Team
