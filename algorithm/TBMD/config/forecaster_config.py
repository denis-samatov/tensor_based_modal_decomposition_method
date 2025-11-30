"""
Конфигурация для forecaster моделей
"""
from dataclasses import dataclass, field
from typing import Literal, Optional
from .base_config import BaseConfig


@dataclass
class ForecasterConfig(BaseConfig):
    """Базовая конфигурация для forecaster моделей"""
    
    # Архитектура
    model_type: Literal['linear', 'mlp', 'lstm', 'transformer'] = 'lstm'
    in_dim: Optional[int] = None  # Входная размерность (устанавливается при инициализации)
    out_dim: Optional[int] = None  # Выходная размерность (устанавливается при инициализации)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    seq_length: int = 5  # Для LSTM/Transformer
    
    # Оптимизация
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: Literal['adam', 'sgd', 'adamw'] = 'adam'
    
    # Обучение
    num_epochs: int = 300
    batch_size: int = 32
    val_split: float = 0.2
    early_stopping_patience: int = 20
    
    # Data Loading
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self._validate()
    
    def _validate(self):
        """Валидация параметров"""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size должен быть положительным")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers должен быть положительным")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout должен быть в диапазоне [0, 1]")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным")
        
        if not 0 < self.val_split < 1:
            raise ValueError("val_split должен быть в диапазоне (0, 1)")


@dataclass
class LinearForecasterConfig(ForecasterConfig):
    """Конфигурация для Linear forecaster"""
    model_type: Literal['linear'] = 'linear'
    
    # Linear не использует эти параметры
    hidden_size: int = 0  # Не используется
    num_layers: int = 0  # Не используется
    dropout: float = 0.0  # Не используется
    seq_length: int = 1  # Linear работает с одним шагом
    
    # Упрощенное обучение для линейной модели
    num_epochs: int = 1  # Обучается за одну итерацию (pseudoinverse)
    early_stopping_patience: int = 0  # Не используется
    
    def _validate(self):
        """Переопределить валидацию - для linear модели не проверяем hidden_size"""
        # Linear модель не использует большинство параметров, skip validation
        pass


@dataclass
class MLPForecasterConfig(ForecasterConfig):
    """Конфигурация для MLP forecaster"""
    model_type: Literal['mlp'] = 'mlp'
    
    # MLP использует больший hidden size и больше эпох
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    seq_length: int = 1  # MLP работает с одним состоянием
    
    # MLP требует больше эпох для сходимости
    num_epochs: int = 500
    early_stopping_patience: int = 20


@dataclass
class LSTMForecasterConfig(ForecasterConfig):
    """Конфигурация для LSTM forecaster"""
    model_type: Literal['lstm'] = 'lstm'
    
    # LSTM параметры
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0  # Применяется только если num_layers > 1
    seq_length: int = 5
    
    # LSTM обучение
    num_epochs: int = 300
    early_stopping_patience: int = 20


# Для обратной совместимости с DigitalTwinConfig
def create_forecaster_config_from_dict(config_dict: dict, model_type: str = 'lstm') -> ForecasterConfig:
    """
    Создать ForecasterConfig из словаря (для совместимости с DigitalTwinConfig.forecaster_config)
    
    Args:
        config_dict: Словарь параметров
        model_type: Тип модели ('linear', 'mlp', 'lstm')
    
    Returns:
        Соответствующий ForecasterConfig
    """
    if model_type == 'linear':
        return LinearForecasterConfig(**config_dict)
    elif model_type == 'mlp':
        return MLPForecasterConfig(**config_dict)
    elif model_type == 'lstm':
        return LSTMForecasterConfig(**config_dict)
    else:
        return ForecasterConfig(**config_dict)

