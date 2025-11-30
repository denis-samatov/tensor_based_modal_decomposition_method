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
Вместо стандартного SVD/HOSVD по пространственным модам, мы используем проекцию на собственные векторы графа или регуляризацию с учетом $L$.

---

## Архитектура

### Основные компоненты

1. **`GeometryAwareTBMD`**
   - Главный класс
   - Строит граф по сетке
   - Вычисляет собственные векторы Лапласиана
   - Выполняет декомпозицию и реконструкцию

2. **`UnstructuredMesh`**
   - Утилита для работы с неструктурированными сетками
   - Импорт из GRDECL / EGRID (через `opm` или парсеры)
   - Определение соседей

3. **`GraphLaplacian`**
   - Построение матрицы смежности
   - Разреженные вычисления (scipy.sparse)
   - Вычисление собственных чисел/векторов

---

## Установка и использование

### Инициализация

```python
from algorithm.TBMD.core.geometry_aware import GeometryAwareTBMD, MeshConfig

# Конфигурация сетки
mesh_config = MeshConfig(
    nx=60, ny=60, nz=1,
    active_cells_mask=mask_2d  # Бинарная маска активных ячеек
)

# Создание модели
geo_tbmd = GeometryAwareTBMD(
    n_modes=50,
    mesh_config=mesh_config
)
```

### Обучение

```python
# data: (n_cells, n_timesteps) - данные только в активных ячейках
# или (nx, ny, nt) - полные данные с нулями

geo_tbmd.fit(data)
```

### Реконструкция

```python
reconstructed = geo_tbmd.reconstruct(reduced_state)
```

---

## API Reference

### `GeometryAwareTBMD`

#### `__init__(n_modes, mesh_config, method='spectral')`
- `n_modes`: Количество сохраняемых мод
- `mesh_config`: Конфигурация сетки
- `method`: 'spectral' (через Лапласиан) или 'regularized' (TBMD с регуляризацией)

#### `fit(data)`
Обучает модель.
- Строит граф смежности
- Вычисляет собственные векторы
- Проецирует данные на базис

#### `transform(data)`
Сжимает данные в модальное пространство.

#### `inverse_transform(reduced_data)`
Восстанавливает исходные данные.

### `MeshConfig`

```python
@dataclass
class MeshConfig:
    nx: int
    ny: int
    nz: int = 1
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    active_cells_mask: Optional[np.ndarray] = None
    connectivity_matrix: Optional[sp.spmatrix] = None
```

---

## Примеры

### Пример 1: L-образный домен

```python
import numpy as np
from algorithm.TBMD.core.geometry_aware import GeometryAwareTBMD, MeshConfig

# 1. Создаем маску (L-shape)
mask = np.ones((20, 20))
mask[10:, 10:] = 0  # Вырезаем угол

# 2. Данные (симуляция диффузии)
data = generate_diffusion_data(mask, n_steps=100)

# 3. Модель
config = MeshConfig(20, 20, active_cells_mask=mask)
model = GeometryAwareTBMD(n_modes=10, mesh_config=config)

# 4. Обучение
model.fit(data)

# 5. Проверка
rec = model.inverse_transform(model.transform(data))
print(f"Reconstruction Error: {np.linalg.norm(data - rec) / np.linalg.norm(data):.4f}")
```

### Пример 2: Brugge Field

Для реальных месторождений используется маска активных ячеек (`ACTNUM`).

```python
# Загрузка маски Brugge
actnum = load_brugge_actnum()  # (nx, ny, nz)

# Конфиг
config = MeshConfig(
    nx=139, ny=48, nz=9,
    active_cells_mask=actnum
)

# TBMD
model = GeometryAwareTBMD(n_modes=100, mesh_config=config)
model.fit(pressure_history)
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

**Версия**: 1.0  
**Дата**: Ноябрь 2025  
**Автор**: TBMD Team
