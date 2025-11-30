"""
Конфигурация для размещения сенсоров

Модуль содержит:
- SensorPlacementConfig: конфигурация для размещения сенсоров (включает все параметры QR)
- GeometricSensorConfig: расширенная конфигурация с геометрией

References:
- Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.)
- Algorithm 2: Tensor-based tube fiber-pivot QR factorization
"""
from dataclasses import dataclass
from typing import Optional, Literal
from .base_config import BaseConfig


@dataclass
class SensorPlacementConfig(BaseConfig):
    """
    Конфигурация для размещения сенсоров методом QR-разложения с пивотированием.
    
    Включает все параметры для Tensor QR decomposition с научно обоснованными 
    константами численной стабильности.
    
    Наследуется от BaseConfig, который предоставляет:
    - seed: Optional[int] = 0 — для воспроизводимости
    - device: Optional[str] = None — 'cuda', 'cpu', или auto
    - dtype: Literal['float32', 'float64'] = 'float32'
    - verbose: bool = True
    
    Attributes
    ----------
    n_sensors : int, default=200
        Количество сенсоров для размещения (параметр N в алгоритме).
    uniform_distribution : bool, default=False
        Включить ограничения на равномерное пространственное распределение.
    check_orthogonality : bool, default=False
        Проверять ортогональность Q на каждом шаге (медленнее, но полезно для отладки).
    random_state : int, optional
        Альтернативный параметр seed для воспроизводимости (совместимость с sklearn API).
        Если указан, переопределяет seed из BaseConfig.
    
    Numerical Stability Constants
    -----------------------------
    machine_epsilon_factor : float, default=1e-6
        Реалистичный допуск для float32. Используется для определения 
        численной незначимости величин.
    householder_threshold : float, default=1e-6
        Порог для вычисления вектора Хаусхолдера. Если норма меньше,
        преобразование пропускается.
    orthogonality_tolerance : float, default=1e-4
        Допуск для проверки ортогональности матрицы Q.
        Учитывает накопление ошибок в float32.
    condition_number_threshold : float, default=1e12
        Максимально допустимое число обусловленности тензора.
        При превышении выдается предупреждение о возможной нестабильности.
    
    Distribution Penalty Weights
    ----------------------------
    slice_penalty_weight : float, default=0.8
        Вес штрафа за дисбаланс между срезами (для 3D+ тензоров).
    distribution_penalty_weight : float, default=0.5
        Вес штрафа за неравномерное пространственное распределение.
    similarity_grouping_decimals : int, default=3
        Точность округления для группировки похожих регионов.
    
    Examples
    --------
    >>> from TBMD.config import SensorPlacementConfig
    >>> config = SensorPlacementConfig(n_sensors=100, seed=42)
    >>> config.n_sensors
    100
    
    References
    ----------
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.)
    - Algorithm 2: Tensor-based tube fiber-pivot QR factorization
    """
    
    # === Основные параметры ===
    n_sensors: int = 200  # Количество сенсоров (N в алгоритме)
    uniform_distribution: bool = False  # Равномерное распределение по слоям
    check_orthogonality: bool = False  # Проверка ортогональности Q
    random_state: Optional[int] = None  # Альтернативный seed для воспроизводимости
    
    # === Численные константы стабильности ===
    machine_epsilon_factor: float = 1e-6  # Realistic tolerance for float32
    householder_threshold: float = 1e-6   # Threshold for Householder vector computation
    orthogonality_tolerance: float = 1e-4  # Realistic tolerance for float32 with accumulation
    condition_number_threshold: float = 1e12  # Maximum acceptable condition number
    
    # === Веса штрафов распределения ===
    slice_penalty_weight: float = 0.8      # Weight for inter-slice balance penalty
    distribution_penalty_weight: float = 0.5  # Weight for spatial distribution penalty
    similarity_grouping_decimals: int = 3   # Precision for similarity grouping
    
    def __post_init__(self):
        super().__post_init__()
        self._validate()
    
    def _validate(self):
        """Валидация параметров"""
        if self.n_sensors <= 0:
            raise ValueError("n_sensors должен быть положительным")
        
        if not (0 < self.slice_penalty_weight <= 1):
            raise ValueError("slice_penalty_weight must be in (0, 1]")
        
        if not (0 < self.distribution_penalty_weight <= 1):
            raise ValueError("distribution_penalty_weight must be in (0, 1]")
        
        if self.condition_number_threshold < 1:
            raise ValueError("condition_number_threshold must be >= 1")


@dataclass
class GeometricSensorConfig(SensorPlacementConfig):
    """
    Конфигурация для geometry-aware размещения сенсоров.
    
    Расширяет SensorPlacementConfig геометрическими параметрами для
    улучшенного размещения на неструктурированных сетках.
    
    Attributes
    ----------
    gradient_weight : float, default=0.5
        Вес для геометрических градиентов (β в формуле).
        Приоритет ячейкам с высокими пространственными градиентами.
    proximity_weight : float, default=1.0
        Вес штрафа за близость к существующим сенсорам (γ).
        Больше = больше расстояние между сенсорами.
    amplitude_weight : float, default=1.0
        Вес для амплитуды поля (приоритет высокоамплитудным регионам).
    energy_weight : float, default=0.5
        Вес для локальной энергии в пространственной окрестности.
    min_distance_factor : float, default=2.0
        Минимальное расстояние между сенсорами как множитель 
        характерной длины сетки: min_distance = min_distance_factor * h_char.
    gradient_method : {'fd', 'graph'}, default='graph'
        Метод вычисления пространственных градиентов.
    adaptive_weights : bool, default=True
        Автоматически нормализовать и масштабировать веса.
    use_graph_distance : bool, default=False
        Использовать геодезическое расстояние по графу вместо евклидова.
    k_neighbors : int, default=6
        Количество соседей для построения графа сетки.
    
    Examples
    --------
    >>> from TBMD.config import GeometricSensorConfig
    >>> config = GeometricSensorConfig(
    ...     n_sensors=50,
    ...     seed=42,
    ...     gradient_weight=0.8,
    ...     proximity_weight=1.5
    ... )
    """
    
    # === Геометрические веса ===
    gradient_weight: float = 0.5  # β - вес градиентов
    proximity_weight: float = 1.0  # γ - штраф за близость
    amplitude_weight: float = 1.0  # Приоритет высокоамплитудным регионам
    energy_weight: float = 0.5  # Локальная энергия в окрестности
    
    # === Параметры расстояния ===
    min_distance_factor: float = 2.0  # min_distance = factor * h_char
    
    # === Методы вычисления ===
    gradient_method: Literal['fd', 'graph'] = 'graph'
    adaptive_weights: bool = True
    use_graph_distance: bool = False
    
    # === Параметры графа ===
    k_neighbors: int = 6
    
    def _validate(self):
        """Дополнительная валидация для геометрических параметров."""
        super()._validate()
        
        if self.gradient_weight < 0:
            raise ValueError("gradient_weight должен быть >= 0")
        
        if self.proximity_weight < 0:
            raise ValueError("proximity_weight должен быть >= 0")
        
        if self.amplitude_weight < 0:
            raise ValueError("amplitude_weight должен быть >= 0")
        
        if self.energy_weight < 0:
            raise ValueError("energy_weight должен быть >= 0")
        
        if self.min_distance_factor <= 0:
            raise ValueError("min_distance_factor должен быть > 0")
        
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors должен быть >= 1")

