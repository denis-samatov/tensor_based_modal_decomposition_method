"""
Конфигурация для тензорной декомпозиции
"""
from dataclasses import dataclass
from typing import List, Optional, Literal, Union
from enum import Enum
import torch
from .base_config import BaseConfig


class ProcessingStrategy(Enum):
    """Strategy for processing modal tensors."""
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    MEMORY_EFFICIENT = "memory_efficient"


@dataclass
class ModalProcessorConfig:
    """Configuration for modal tensor processing."""
    device: str = 'cpu'
    return_numpy: bool = True
    processing_strategy: ProcessingStrategy = ProcessingStrategy.BATCH
    batch_size: Optional[int] = None
    memory_limit_gb: float = 4.0
    enable_progress_logging: bool = True
    validation_enabled: bool = True
    numerical_precision: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")


@dataclass
class DecompositionConfig(BaseConfig):
    """Конфигурация для HOSVD/Tucker декомпозиции"""
    
    # Параметры декомпозиции
    ranks: Optional[Union[int, List[int]]] = None  # [spatial_rank, temporal_rank] or single int
    method: Literal['hosvd', 'tucker', 'st_hosvd'] = 'hosvd'
    
    # Пороги отсечения
    energy_threshold: float = 0.99  # Порог энергии для автоматического выбора рангов
    singular_value_threshold: float = 1e-10  # Порог для сингулярных значений
    
    # Оптимизация
    max_iterations: int = 100  # Для итеративных методов
    convergence_tol: float = 1e-6
    
    # Центрирование данных
    center_data: bool = False
    normalize: bool = False
    
    # Численные ограничения (из hosvd.py)
    min_rank: int = 1  # Минимальный допустимый ранг
    epsilon: float = 1e-2  # Epsilon для Tucker
    
    # Параллелизм
    max_workers: Optional[int] = None # Number of parallel workers (used in hosvd.py)
    
    # Дополнительные поля из hosvd.py
    random_state: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self._validate()
    
    def _validate(self):
        """Валидация параметров"""
        if self.ranks is not None:
            if isinstance(self.ranks, list):
                if len(self.ranks) != 2:
                    # Allow more than 2 ranks for general Tucker, but warn or just pass if not strict
                    pass 
                if any(r <= 0 for r in self.ranks):
                    raise ValueError("Все ранги должны быть положительными")
            elif isinstance(self.ranks, int):
                if self.ranks <= 0:
                    raise ValueError("Ранг должен быть положительным")
        
        if not 0 < self.energy_threshold <= 1:
            raise ValueError("energy_threshold должен быть в диапазоне (0, 1]")
        
        if self.singular_value_threshold < 0:
            raise ValueError("singular_value_threshold должен быть неотрицательным")


@dataclass
class GeometryAwareDecompositionConfig(DecompositionConfig):
    """Конфигурация для geometry-aware декомпозиции"""
    
    # Геометрические параметры
    alpha: float = 0.1  # Вес геометрической регуляризации
    alpha_adaptive: bool = False  # Адаптивный выбор alpha
    alpha_min: float = 0.01
    alpha_max: float = 0.5
    
    # Параметры графа
    graph_metric: Literal['euclidean', 'geodesic'] = 'euclidean'
    k_neighbors: int = 6  # Количество соседей для построения графа
    
    # Laplacian
    laplacian_type: Literal['unnormalized', 'symmetric', 'random_walk'] = 'symmetric'
    
    # Веса
    weight_function: Literal['inverse_distance', 'gaussian', 'uniform'] = 'gaussian'
    gaussian_sigma: Optional[float] = None  # None = автоматический выбор
    
    def _validate(self):
        """Дополнительная валидация"""
        super()._validate()
        
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha должен быть в диапазоне [0, 1]")
        
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors должен быть >= 1")
        
        if self.alpha_adaptive:
            if not 0 <= self.alpha_min <= self.alpha_max <= 1:
                raise ValueError("Должно выполняться: 0 <= alpha_min <= alpha_max <= 1")

