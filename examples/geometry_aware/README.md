# Geometry-Aware TBMD Examples

Примеры использования TBMD с учетом геометрии данных

## 📋 Список примеров

### 01. Graph-Based TBMD (`01_graph_based_tbmd.py`)
**Описание**: Полный пайплайн TBMD с использованием графовых структур  
**Уровень**: Advanced  
**Основные концепции**:
- Построение графа из геометрии mesh
- Geometry-aware Tucker decomposition
- Графовая регуляризация в реконструкции
- Laplacian smoothing

**Использование**:
```bash
python 01_graph_based_tbmd.py
```

### 02. Geometry-Aware Compressive Sensing (`02_geometry_aware_cs.py`)
**Описание**: Compressive sensing с учетом геометрической структуры  
**Уровень**: Intermediate  
**Основные концепции**:
- GeometryAwareTensorCS
- Spatial regularization
- Anisotropic diffusion
- Graph Laplacian penalties

**Использование**:
```bash
python 02_geometry_aware_cs.py
```

### 03. Geometry-Aware Decomposition (`03_geometry_aware_decomposition.py`)
**Описание**: Tucker декомпозиция с геометрическими ограничениями  
**Уровень**: Intermediate  
**Основные концепции**:
- GeometryAwareTuckerDecomposer
- Spatially coherent modes
- Geometry-constrained optimization

**Использование**:
```bash
python 03_geometry_aware_decomposition.py
```

### 04. Geometry Utilities (`04_geometry_utils.py`)
**Описание**: Утилиты для работы с геометрией  
**Уровень**: Beginner  
**Основные концепции**:
- MeshGeometry
- MeshGraphBuilder
- Distance computation
- Neighbor finding

**Использование**:
```bash
python 04_geometry_utils.py
```

### 05. Test Components (`05_test_components.py`)
**Описание**: Тестирование geometry-aware компонентов  
**Уровень**: Advanced  
**Основные концепции**:
- Unit testing
- Integration testing
- Performance benchmarking
- Validation

**Использование**:
```bash
python 05_test_components.py
```

## 🎯 Типичные use cases

### 1. Работа с нерегулярными сетками
Если ваши данные на нерегулярной сетке:
```python
from TBMD.utils.geometry import MeshGeometry, MeshGraphBuilder
from TBMD.modules.GeometryAwareTensorHOSVD import GeometryAwareTuckerDecomposer

# Создать геометрию
geometry = MeshGeometry(vertices, faces)
graph = MeshGraphBuilder().build_graph(geometry)

# Decomposition с учетом геометрии
decomposer = GeometryAwareTuckerDecomposer(geometry=geometry)
result = decomposer.decompose(data)
```

### 2. Анизотропные поля
Для полей с направленной структурой:
```python
from TBMD.modules.GeometryAwareTensorCS import GeometryAwareTensorCS

reconstructor = GeometryAwareTensorCS(
    geometry=geometry,
    use_anisotropic=True
)
```

### 3. Сохранение пространственной когерентности
Для smooth полей:
```python
config = GeometryAwareConfig(
    spatial_smoothness_weight=1.0,
    use_graph_laplacian=True
)
```

## 📊 Сравнение методов

| Метод | Точность | Скорость | Сложность | Use Case |
|-------|----------|----------|-----------|----------|
| Standard TBMD | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Регулярные сетки |
| Geometry-Aware | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Нерегулярные сетки |
| Graph-Based | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Сложные геометрии |

## 🔧 Конфигурация

### Базовая конфигурация
```python
from TBMD.modules.GeometryAwareTensorHOSVD import GeometryAwareConfig

config = GeometryAwareConfig(
    spatial_smoothness_weight=1.0,
    use_graph_laplacian=True,
    laplacian_weight=0.1
)
```

### Продвинутая конфигурация
```python
config = GeometryAwareConfig(
    spatial_smoothness_weight=1.0,
    temporal_smoothness_weight=0.5,
    use_graph_laplacian=True,
    laplacian_weight=0.1,
    use_anisotropic=True,
    anisotropy_tensor=custom_tensor
)
```

## 📚 Дополнительные ресурсы

- [Geometry-Aware TBMD Guide](../../docs/guides/GEOMETRY_AWARE_TBMD.md)
- [Geometry-Aware Quick Start](../../docs/tutorials/GEOMETRY_AWARE_QUICKSTART.md)
- [Geometry-Aware CS Tutorial](../../docs/tutorials/GeometryAwareCS_README.md)

## ⚠️ Важные замечания

1. **Производительность**: Geometry-aware методы медленнее стандартных
2. **Память**: Хранение графов требует дополнительной памяти
3. **Конвергенция**: Может требоваться больше итераций
4. **Параметры**: Требуют тщательной настройки весов

## 🐛 Устранение проблем

### Проблема: Слишком медленно
**Решение**: Уменьшите количество вершин в графе или используйте coarsening

### Проблема: Результаты слишком smooth
**Решение**: Уменьшите `spatial_smoothness_weight`

### Проблема: Не сохраняется структура
**Решение**: Увеличьте `laplacian_weight`

