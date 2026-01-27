"""
Базовая конфигурация для всех модулей TBMD
"""
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import tensorly as tl
import random
import numpy as np


@dataclass
class BaseConfig:
    """Базовая конфигурация для всех TBMD модулей"""
    
    # Вычислительные параметры
    backend: Literal['pytorch', 'numpy'] = 'pytorch'
    dtype: Literal['float32', 'float64'] = 'float32'
    device: Optional[str] = None  # 'cuda', 'cpu', или None (auto)
    
    # Воспроизводимость
    seed: Optional[int] = 0
    deterministic: bool = True
    
    # Логирование
    verbose: bool = True
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    
    def __post_init__(self):
        """Валидация и автоматическая настройка"""
        # Автоматический выбор устройства
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Установка seed
        if self.seed is not None:
            self._set_seed()
    
    def _set_seed(self):
        """Установка seed для воспроизводимости.

        Delegates to the central `set_seed` helper in TBMD.utils.tbmd_utils
        for consistent seeding across NumPy, random, torch and Tensorly.
        Also enables deterministic algorithms when requested.
        """
        try:
            # Import local helper to centralise seed logic
            from TBMD.utils.tbmd_utils import set_seed as _set_seed_helper
            _set_seed_helper(int(self.seed))
        except Exception:
            # Fallback: set seeds manually if the helper isn't available
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            try:
                tl.set_random_state(self.seed)
            except Exception:
                pass

            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            # Prefer new API when available (PyTorch 1.8+)
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def to_dict(self):
        """Преобразовать конфигурацию в словарь"""
        return {
            'backend': self.backend,
            'dtype': self.dtype,
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'verbose': self.verbose,
            'log_level': self.log_level
        }
