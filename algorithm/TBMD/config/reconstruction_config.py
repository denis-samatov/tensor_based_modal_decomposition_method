"""
Конфигурация для реконструкции (TBMD-CS Algorithm 3)

Модуль содержит конфигурации для ADMM-based компрессивного сенсинга:
- CompressiveSensingConfig: основные гиперпараметры алгоритма
- ExtensionCompressiveSensingConfig: расширенные настройки (solver, policies)
- ReconstructionConfig: legacy конфиг для обратной совместимости
- GeometryAwareReconstructionConfig: geometry-aware реконструкция

References:
- Algorithm 3: TBMD-CS (formulas 32-36)
- Boyd et al. (2011): Distributed Optimization and Statistical Learning via ADMM
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Union
import torch
from .base_config import BaseConfig


def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        if dtype in dtype_map:
            return dtype_map[dtype]
        raise ValueError(f"Unknown dtype string: {dtype}. Use 'float32', 'float64', etc.")
    raise TypeError(f"dtype must be str or torch.dtype, got {type(dtype)}")


@dataclass
class CompressiveSensingConfig:
    """
    Core hyperparameters of the TBMD-CS algorithm.
    
    Это основной конфиг для Algorithm 3 (ADMM-based compressive sensing).
    
    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of ADMM iterations.
    tol : float, default=1e-4
        Termination threshold for max(primal_residual, dual_residual).
    epsilon_l1 : float, default=1e-2
        ε in equation (28); L1 shrinkage parameter for soft-thresholding.
    delta_init : float, default=1.0
        Initial value of the ADMM penalty parameter δ₀.
    delta_max : float, default=1.0
        Maximum cap for δ as in (36).
    relax_lambda : float, default=0.95
        Over-relaxation mixing coefficient for x and d. Must be in (0, 1).
    device : str, default="cpu"
        Torch device string (e.g., "cpu", "cuda:0").
    dtype : torch.dtype or str, default=torch.float32
        Torch dtype used for tensors. Accepts both torch.dtype and string ('float32', 'float64').
    
    Examples
    --------
    >>> config = CompressiveSensingConfig(max_iter=500, tol=1e-5)
    >>> cs = TensorCompressiveSensing(A, P, Y, core_cfg=config)
    
    >>> # Using string dtype
    >>> config = CompressiveSensingConfig(dtype='float64')
    """
    max_iter: int = 1000
    tol: float = 1e-4                 # stop criterion on max(primal, dual)
    epsilon_l1: float = 1e-2          # ε in (28)
    delta_init: float = 1.0           # δ₀
    delta_max: float = 1.0            # δ_max (36)
    relax_lambda: float = 0.95        # mixing x and d
    device: str = "cpu"
    dtype: Union[torch.dtype, str] = torch.float32

    def __post_init__(self) -> None:
        """Validate parameter ranges right after dataclass construction."""
        # Convert string dtype to torch.dtype
        self.dtype = _resolve_dtype(self.dtype)
        
        if not (0 < self.relax_lambda < 1):
            raise ValueError("relax_lambda must be in (0, 1)")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.epsilon_l1 <= 0:
            raise ValueError("epsilon_l1 must be > 0")
        if self.delta_init <= 0 or self.delta_max <= 0:
            raise ValueError("delta values must be > 0")


@dataclass
class ExtensionCompressiveSensingConfig:
    """
    Convenience switches for features outside the strict TBMD-CS core.
    
    Parameters
    ----------
    solver : {"cholesky", "direct", "svd"}, default="cholesky"
        Linear system solver used in the x-update.
    reg : float, default=1e-8
        Small diagonal regularization added to lhs for numerical stability.
    delta_policy : {"boyd", "cap_only"}, default="boyd"
        Strategy for adapting δ during iterations.
    stop_policy : {"residual", "relative", "both"}, default="residual"
        Termination rule.
    relative_window : int, default=5
        Window size (iterations) for the relative stopping rule.
    relative_drop : float, default=1e-3
        Required relative decrease within relative_window iterations.
    collect_history : bool, default=True
        Whether to store residual history for diagnostics and plotting.
    
    Examples
    --------
    >>> ext_cfg = ExtensionCompressiveSensingConfig(solver="svd", collect_history=True)
    >>> cs = TensorCompressiveSensing(A, P, Y, ext_cfg=ext_cfg)
    """
    # Linear solver
    solver: Literal["cholesky", "direct", "svd"] = "cholesky"
    reg: float = 1e-8                  # diagonal regularization
    
    # δ policy
    delta_policy: Literal["boyd", "cap_only"] = "boyd"
    
    # Stop conditions
    stop_policy: Literal["residual", "relative", "both"] = "residual"
    relative_window: int = 5           # window for relative criterion
    relative_drop: float = 1e-3        # required relative drop
    
    # Metrics/logging
    collect_history: bool = True


# =============================================================================
# Legacy configs for backward compatibility
# =============================================================================

@dataclass
class ReconstructionConfig(BaseConfig):
    """
    Legacy конфигурация для компрессивного сенсинга.
    
    DEPRECATED: Используйте CompressiveSensingConfig + ExtensionCompressiveSensingConfig.
    Этот класс сохранён для обратной совместимости.
    """
    
    # Алгоритм решения
    solver: Literal['admm', 'ista', 'fista', 'least_squares'] = 'admm'
    
    # Параметры оптимизации
    max_iterations: int = 100
    convergence_eps: float = 1e-2
    
    # ADMM параметры
    damping_factor: float = 0.95  # ρ (rho)
    over_relaxation: float = 1.0  # α для over-relaxation
    
    # Regularization
    l1_lambda: float = 0.01  # L1 регуляризация
    l2_lambda: float = 0.0   # L2 регуляризация
    
    # Адаптивный шаг
    adaptive_step_size: bool = True
    initial_step_size: float = 1.0
    max_step_size: float = 1.0
    
    # Warm start
    warm_start: bool = False
    initial_guess: Optional[Literal['zero', 'least_squares', 'previous']] = None
    
    def __post_init__(self):
        super().__post_init__()
        self._validate()
    
    def _validate(self):
        """Валидация параметров"""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations должен быть положительным")
        
        if self.convergence_eps <= 0:
            raise ValueError("convergence_eps должен быть положительным")
        
        if not 0 < self.damping_factor <= 1:
            raise ValueError("damping_factor должен быть в (0, 1]")
        
        if self.l1_lambda < 0 or self.l2_lambda < 0:
            raise ValueError("lambda параметры должны быть неотрицательными")
    
    def to_core_config(self) -> CompressiveSensingConfig:
        """Convert to new CompressiveSensingConfig format."""
        return CompressiveSensingConfig(
            max_iter=self.max_iterations,
            tol=self.convergence_eps,
            epsilon_l1=self.l1_lambda,
            delta_init=self.initial_step_size,
            delta_max=self.max_step_size,
            relax_lambda=self.damping_factor,
            device=self.device or "cpu",
            dtype=torch.float64 if self.dtype == 'float64' else torch.float32
        )


@dataclass
class GeometryAwareReconstructionConfig(ReconstructionConfig):
    """Конфигурация для geometry-aware реконструкции"""
    
    # Геометрическая регуляризация
    geometric_lambda: float = 0.1  # Вес Laplacian регуляризации
    adaptive_lambda: bool = False  # Адаптивный выбор
    
    # Laplacian параметры
    laplacian_type: Literal['unnormalized', 'symmetric', 'random_walk'] = 'symmetric'
    laplacian_power: int = 1  # Степень Laplacian (1 = градиент, 2 = лапласиан)
    
    # Локальная гладкость
    enforce_local_smoothness: bool = True
    smoothness_weight: float = 0.5
    
    def _validate(self):
        """Дополнительная валидация"""
        super()._validate()
        
        if self.geometric_lambda < 0:
            raise ValueError("geometric_lambda должен быть неотрицательным")
        
        if self.laplacian_power < 1:
            raise ValueError("laplacian_power должен быть >= 1")
        
        if not 0 <= self.smoothness_weight <= 1:
            raise ValueError("smoothness_weight должен быть в диапазоне [0, 1]")


# Aliases for backward compatibility
TensorCSConfig = CompressiveSensingConfig
CSConfig = CompressiveSensingConfig

