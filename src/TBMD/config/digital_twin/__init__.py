"""
Конфигурация для цифрового двойника
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
from ..base import BaseConfig


@dataclass
class DigitalTwinConfig(BaseConfig):
    """Конфигурация для цифрового двойника"""
    
    # Архитектура
    n_spatial_modes: int = 40
    n_temporal_modes: int = 20
    n_sensors: int = 30
    
    # Компоненты - Forecaster (для модальных коэффициентов)
    forecaster_type: Literal['linear', 'mlp', 'lstm', 'persistence'] = 'linear'
    
    # Компоненты - Proxy Model (для сценарного анализа с well controls)
    proxy_model_type: Optional[Literal['linear_dynamics', 'neural', 'physics_informed']] = None
    proxy_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    
    # Параметры forecaster
    forecaster_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'seq_length': 5,  # Для LSTM/Transformer
        'learning_rate': 0.001,
        'weight_decay': 1e-5  # L2 регуляризация
    })
    
    # Параметры proxy model
    proxy_config: Dict[str, Any] = field(default_factory=lambda: {
        'regularization': 1e-4,  # Для LinearDynamics
        'learning_rate': 1e-3,   # Для Neural
        'batch_size': 32,        # Для Neural
    })
    
    # Обучение (согласовано с моделями)
    train_test_split: float = 0.8
    validation_split: float = 0.2  # Обновлено с 0.1 (согласовано с моделями)
    batch_size: int = 32
    epochs: int = 300  # Обновлено с 100 (согласовано с моделями)
    early_stopping_patience: int = 20  # Обновлено с 10 (согласовано с моделями)
    
    # Data Loading
    shuffle_training_data: bool = True  # Перемешивание данных при обучении
    num_workers: int = 0  # Количество workers для DataLoader
    pin_memory: bool = True  # Закрепление памяти для GPU
    
    # Режимы работы
    online_learning: bool = False  # Обучение на лету
    uncertainty_estimation: bool = False  # Оценка неопределенности
    anomaly_detection: bool = False  # Детекция аномалий
    
    # Оптимизация
    use_data_assimilation: bool = True  # Ассимиляция данных с сенсоров
    assimilation_method: Literal['kalman', 'ensemble', 'variational'] = 'ensemble'
    
    # Производительность
    cache_decomposition: bool = True
    parallel_scenarios: bool = True
    max_workers: Optional[int] = None  # None = автоматически
    
    def __post_init__(self):
        super().__post_init__()
        self._validate()
    
    def _validate(self):
        """Валидация параметров"""
        if self.n_spatial_modes <= 0:
            raise ValueError("n_spatial_modes должен быть положительным")
        
        if self.n_temporal_modes <= 0:
            raise ValueError("n_temporal_modes должен быть положительным")
        
        if self.n_sensors <= 0:
            raise ValueError("n_sensors должен быть положительным")
        
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split должен быть в (0, 1)")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split должен быть в (0, 1)")
        
        # validation_split применяется к тренировочным данным, поэтому проверяем только индивидуально
        # Например: train_test_split=0.8 берет 80% для обучения, validation_split=0.2 берет 20% из этих 80%
        
        if self.batch_size <= 0:
            raise ValueError("batch_size должен быть положительным")
        
        if self.epochs <= 0:
            raise ValueError("epochs должен быть положительным")
