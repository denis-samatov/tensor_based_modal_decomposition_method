"""
Digital Twin для мониторинга и прогнозирования месторождений

Объединяет TBMD с прогнозными моделями для создания цифрового двойника.

Полный workflow:
1. Tucker Decomposition → cores + factors
2. Modal Tensor Processing → A_tensor (modal basis)
3. QR Factorization → sensor placement (P matrix)
4. Compressive Sensing → reconstruction from sparse sensors
5. Forecasting → предсказание следующих состояний (Linear/MLP/LSTM)
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

# Config imports
from TBMD.config.digital_twin_config import DigitalTwinConfig
from TBMD.config.decomposition_config import DecompositionConfig, ModalProcessorConfig
from TBMD.config.sensor_placement_config import SensorPlacementConfig
from TBMD.config.reconstruction_config import CompressiveSensingConfig, ExtensionCompressiveSensingConfig

# Core TBMD imports
from TBMD.core.decomposition.hosvd import TuckerDecomposer
from TBMD.core.modal_processor.modes import ProcessingStrategy, BatchModalProcessor, ModalTensorStacker
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

# Forecasting Models
from TBMD.models.LinearForecaster import LinearForecaster
from TBMD.models.MLPForecaster import MLPForecaster
from TBMD.models.LSTMForecaster import LSTMForecaster

# Reservoir Proxy Models (как в system.py)
from TBMD.models.ReservoirProxyModel import (
    ReservoirProxyModelBase,
    LinearDynamicsProxyModel,
    NeuralProxyModel,
    PhysicsInformedProxyModel,
    ReservoirState,
    WellControl
)

# Forecaster Configs
from TBMD.config.forecaster_config import (
    LinearForecasterConfig,
    MLPForecasterConfig,
    LSTMForecasterConfig
)

# Utils
from TBMD.utils.tbmd_utils import reconstruct_tensor, to_torch_tensor

logger = logging.getLogger(__name__)


class ForecasterType(Enum):
    """Типы прогнозных моделей для модальных коэффициентов"""
    LINEAR = "linear"
    MLP = "mlp"
    LSTM = "lstm"
    PERSISTENCE = "persistence"  # Просто повторяет текущее состояние


class ProxyModelType(Enum):
    """
    Типы proxy-моделей для физического прогнозирования резервуара.
    
    Как в system.py - для сценарного анализа с well controls.
    """
    LINEAR_DYNAMICS = "linear_dynamics"   # x(t+1) = A @ x(t) + B @ u(t)
    NEURAL = "neural"                     # Neural network proxy
    PHYSICS_INFORMED = "physics_informed" # С физическими ограничениями


# Type alias for ranks
from typing import Union as UnionType
RanksType = UnionType[int, list]



@dataclass
class DigitalTwinState:
    """
    Текущее состояние цифрового двойника
    
    Attributes:
        current_time: Текущее время
        modal_coefficients: Текущие модальные коэффициенты
        prediction_error: Ошибка прогнозирования
        is_calibrated: Откалиброван ли twin
        alert_status: Статус алерта ('normal', 'warning', 'critical')
        history: История измерений и прогнозов
    """
    current_time: float = 0.0
    modal_coefficients: Optional[torch.Tensor] = None
    prediction_error: float = 0.0
    is_calibrated: bool = False
    alert_status: str = 'normal'
    history: Dict[str, List] = field(default_factory=lambda: {
        'times': [],
        'errors': [],
        'predictions': [],
        'observations': []
    })


class DigitalTwin:
    """
    Цифровой двойник месторождения с TBMD
    
    Объединяет:
    1. TBMD декомпозицию для снижения размерности
    2. Оптимальное размещение сенсоров
    3. Реконструкцию полных полей по измерениям
    4. Прогнозирование будущих состояний
    5. Мониторинг и детекцию аномалий
    
    Examples:
        >>> config = DigitalTwinConfig(
        ...     n_spatial_modes=40,
        ...     n_sensors=30
        ... )
        >>> twin = DigitalTwin(config)
        >>> twin.train(historical_data)
        >>> forecast = twin.predict(current_state, n_steps=10)
    """
    
    def __init__(self, config: DigitalTwinConfig):
        """
        Args:
            config: Конфигурация цифрового двойника
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # Состояние
        self.state = DigitalTwinState()
        
        # Компоненты TBMD
        self.decomposer = None
        self.sensor_placer = None
        self.reconstructor = None
        
        # Forecaster (прогнозная модель для модальных коэффициентов)
        self.forecaster = None
        self.forecaster_type = ForecasterType(config.forecaster_type) if hasattr(config, 'forecaster_type') else ForecasterType.PERSISTENCE
        self._modal_history = None  # История модальных коэффициентов для обучения forecaster
        
        # Proxy Model (для сценарного анализа с well controls, как в system.py)
        self.proxy_model: Optional[ReservoirProxyModelBase] = None
        # Проверяем на None перед преобразованием в enum
        _proxy_type = getattr(config, 'proxy_model_type', None)
        self.proxy_model_type = ProxyModelType(_proxy_type) if _proxy_type is not None else None
        self._spatial_shape: Optional[Tuple[int, ...]] = None
        
        # Обученные параметры
        self.spatial_modes = None
        self.temporal_modes = None
        self.core_tensor = None
        self.sensor_mask = None          # Boolean mask на пространственной сетке
        self.sensor_indices = None       # Линейные индексы сенсоров
        self.measurement_matrix = None   # Совместимый с CS формат маски
        
        # Статистика
        self.mean = None
        self.std = None
        
        if config.verbose:
            logger.info(f"Digital Twin инициализирован: {config.n_spatial_modes} мод, {config.n_sensors} сенсоров")
    
    def _validate_tensor_shape(self, tensor: torch.Tensor, expected_dims: int, param_name: str):
        """
        Валидировать форму входного тензора.
        
        Args:
            tensor: Тензор для проверки
            expected_dims: Ожидаемое количество размерностей
            param_name: Название параметра для сообщения об ошибке
            
        Raises:
            ValueError: Если форма неправильная
        """
        if tensor.ndim != expected_dims:
            raise ValueError(
                f"{param_name} должен иметь {expected_dims} размерностей, "
                f"получено {tensor.ndim} с формой {tensor.shape}"
            )
        
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{param_name} содержит NaN или Inf значения")
        
        if tensor.numel() == 0:
            raise ValueError(f"{param_name} не может быть пустым")
    
    def train(
        self,
        historical_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        normalize: bool = False,
        ranks: Optional[RanksType] = None
    ):
        """
        Обучить digital twin на исторических данных
        
        Правильная последовательность (как в new_tbmd.ipynb):
        1. Tucker Decomposition с config → cores + factors
        2. Modal Tensor Processing → A_tensor
        3. QR Factorization с config → sensor placement
        
        Args:
            historical_data: Исторические данные - torch.Tensor (любой размерности)
                            или Dict[str, torch.Tensor] для нескольких subjects
            normalize: Нормализовать данные внутри метода (по умолчанию False).
                       Если данные уже нормализованы извне, оставьте False.
            ranks: Ranks для Tucker decomposition. Если None, автоматически
                   создаётся [n_spatial_modes, n_temporal_modes] или 
                   используется размерность данных
        """
        # Обработка входных данных
        # Сброс статистики нормализации (нормализуйте данные заранее, если нужно)
        self.mean, self.std = None, None
        if isinstance(historical_data, dict):
            data_dict = {
                k: to_torch_tensor(v, device=self.device, dtype=self.dtype)
                for k, v in historical_data.items()
            }
        else:
            historical_data = to_torch_tensor(historical_data, device=self.device, dtype=self.dtype)
            data_dict = {"train": historical_data}
        
        # Все тензоры должны иметь одинаковую форму
        shapes = {v.shape for v in data_dict.values()}
        if len(shapes) != 1:
            raise ValueError(f"Все тензоры должны иметь одинаковую форму, получены формы: {shapes}")
        
        sample_tensor = next(iter(data_dict.values()))
        
        # Валидация входных данных
        if sample_tensor.ndim < 3:
            raise ValueError(
                f"historical_data должен иметь как минимум 3 размерности (spatial_dims..., time), "
                f"получено {sample_tensor.ndim}"
            )
        
        self._original_ndim = sample_tensor.ndim
        self._spatial_shape = sample_tensor.shape[:-1]
        
        if self.config.verbose:
            logger.info(f"Начало обучения Digital Twin на данных формы {sample_tensor.shape}")
        
        # ========================================================================
        # Step 1: TBMD Tucker Декомпозиция (как в new_tbmd.ipynb)
        # ========================================================================
        # Определить ranks
        if ranks is not None:
            effective_ranks = ranks if isinstance(ranks, list) else [ranks] * sample_tensor.ndim
        else:
            # Auto-determine ranks based on data dimensions
            if sample_tensor.ndim == 3:
                effective_ranks = [
                    min(self.config.n_spatial_modes, sample_tensor.shape[0]),
                    min(self.config.n_spatial_modes, sample_tensor.shape[1]),
                    min(self.config.n_temporal_modes, sample_tensor.shape[2])
                ]
            else:
                # Для 4D+ данных: первые N-1 = spatial, последний = temporal
                effective_ranks = [
                    min(self.config.n_spatial_modes, sample_tensor.shape[i])
                    for i in range(sample_tensor.ndim - 1)
                ] + [min(self.config.n_temporal_modes, sample_tensor.shape[-1])]
        
        decomp_config = DecompositionConfig(
            ranks=effective_ranks,
            epsilon=1e-2,
            random_state=self.config.seed if hasattr(self.config, 'seed') else None,
            device=self.config.device,
            dtype=self.config.dtype
        )
        
        # Создать decomposer с config (как в new_tbmd.ipynb)
        self.decomposer = TuckerDecomposer(
            tensors=data_dict,
            device=self.config.device,
            config=decomp_config
        )
        
        self.decomposer.decompose()
        
        # Извлечь cores и factors
        cores = self.decomposer.cores
        factors = self.decomposer.factors
        
        if self.config.verbose:
            logger.info(f"✅ Декомпозиция завершена, ranks={effective_ranks}")
        
        # ========================================================================
        # Step 2: Modal Tensor Processing (как в new_tbmd.ipynb)
        # ========================================================================
        modal_config = ModalProcessorConfig(
            device=self.config.device,
            processing_strategy=ProcessingStrategy.BATCH,
            enable_progress_logging=self.config.verbose,
            return_numpy=False
        )
        
        batch_processor = BatchModalProcessor(modal_config)
        stacker = ModalTensorStacker(modal_config)
        
        # Вычислить modal tensors
        modal_tensors = batch_processor.process_multiple_subjects(cores, factors)
        
        # Сложить в A_tensor (время‑инвариантные моды)
        A_tensor = stacker.stack_modal_tensors(modal_tensors)
        
        # Сохранить для использования
        self.spatial_modes = A_tensor  # Modal basis
        self.core_tensor = cores
        self.temporal_modes = factors
        # Количество мод = последняя размерность A_tensor
        modal_dim = A_tensor.shape[-1]
        # Автоподстройка числа сенсоров, чтобы не решать недоопределённую задачу CS
        max_sensors = int(np.prod(self._spatial_shape))
        if self.config.n_sensors < modal_dim:
            adjusted = min(modal_dim, max_sensors)
            logger.warning(
                f"n_sensors={self.config.n_sensors} меньше числа мод {modal_dim}; "
                f"устанавливаю n_sensors={adjusted} для устойчивой реконструкции."
            )
            self.config.n_sensors = adjusted
        
        if self.config.verbose:
            logger.info(f"✅ Modal tensor вычислен: {A_tensor.shape}")
        
        # ========================================================================
        # Step 3: Размещение сенсоров через QR с config (как в new_tbmd.ipynb)
        # ========================================================================
        sensor_config = SensorPlacementConfig(
            n_sensors=self.config.n_sensors,
            random_state=self.config.seed if hasattr(self.config, 'seed') else None,
            device=self.config.device,
            dtype=self.config.dtype,
            check_orthogonality=True,
            uniform_distribution=False
        )
        
        qr_decomposer = TensorTubeQRDecomposition(
            tensor=A_tensor,
            config=sensor_config
        )
        
        if self.config.verbose:
            logger.info("Выполняется QR факторизация...")
        
        P, Q, R = qr_decomposer.factorize()
        
        # Проверка
        is_valid, error, metrics = qr_decomposer.check_factorization()
        
        if self.config.verbose:
            logger.info(f"✅ QR факторизация: valid={is_valid}, error={error:.2e}")
            logger.info(f"   Orthogonality deviation: {metrics['orthogonality_deviation']:.2e}")
            logger.info(f"   Sensors placed: {metrics['sensor_count']}/{qr_decomposer.N}")
        
        # Сохранить результаты
        self.sensor_mask = P.bool()
        self.sensor_indices = torch.nonzero(self.sensor_mask.reshape(-1), as_tuple=False).squeeze(-1)
        self.measurement_matrix = self.sensor_mask
        
        # ========================================================================
        # Step 4: Обучение прогнозной модели (Forecaster)
        # ========================================================================
        self._train_forecaster(data_dict, sample_tensor)
        
        # ========================================================================
        # Step 5: Инициализация Proxy Model (если указан, как в system.py)
        # ========================================================================
        if self.proxy_model_type is not None:
            self._init_proxy_model()
            try:
                hist_states, hist_controls = self._build_proxy_training_sets(data_dict)
                self.calibrate_proxy_model(
                    hist_states,
                    hist_controls,
                    **getattr(self.config, "proxy_config", {})
                )
            except Exception as e:
                logger.warning(f"Proxy calibration skipped/failed: {e}")
        
        # Обновить состояние
        self.state.is_calibrated = True
        self.state.current_time = 0.0
        
        if self.config.verbose:
            logger.info("✅ Digital Twin обучен успешно")
    
    def _train_forecaster(
        self,
        data_dict: Dict[str, torch.Tensor],
        sample_tensor: torch.Tensor
    ):
        """
        Обучить прогнозную модель на модальных коэффициентах.
        
        Модели работают в модальном пространстве:
        - Вход: x(t) - модальные коэффициенты в момент t
        - Выход: x(t+1) - модальные коэффициенты в момент t+1
        
        Args:
            data_dict: Словарь с нормализованными данными
            sample_tensor: Пример тензора для определения размерностей
        """
        if self.forecaster_type == ForecasterType.PERSISTENCE:
            if self.config.verbose:
                logger.info("📊 Forecaster: persistence (без обучения)")
            return
        
        if self.config.verbose:
            logger.info(f"📊 Обучение forecaster ({self.forecaster_type.value})...")
        
        # Проецировать данные в модальное пространство.
        # Используем только одного субъекта, чтобы не склеивать разные траектории.
        first_key = next(iter(data_dict))
        if self.config.verbose and len(data_dict) > 1:
            logger.info(f"Forecaster обучается только на subject '{first_key}' (без склейки траекторий).")
        data = data_dict[first_key]
        T = data.shape[-1]
        modal_seq = []
        for t in range(T):
            state_t = data[..., t]
            modal_t = self._project_to_modal_space(state_t)
            modal_seq.append(modal_t)
        modal_history = torch.stack(modal_seq, dim=0)  # (T, n_modes)
        self._modal_history = modal_history
        self._modal_history_subject = first_key
        
        n_modes = modal_history.shape[1]
        
        # Получить параметры из config
        forecaster_config = getattr(self.config, 'forecaster_config', {})
        
        # Создать и обучить forecaster
        if self.forecaster_type == ForecasterType.LINEAR:
            self._train_linear_forecaster(modal_history, forecaster_config)
        elif self.forecaster_type == ForecasterType.MLP:
            self._train_mlp_forecaster(modal_history, n_modes, forecaster_config)
        elif self.forecaster_type == ForecasterType.LSTM:
            self._train_lstm_forecaster(modal_history, n_modes, forecaster_config)
        
        if self.config.verbose:
            logger.info(f"✅ Forecaster ({self.forecaster_type.value}) обучен")
    
    def _train_linear_forecaster(
        self,
        modal_history: torch.Tensor,
        config: Dict[str, Any]
    ):
        """Обучить линейный forecaster: x(t+1) = A @ x(t)"""
        # LinearForecaster работает с numpy
        x_history = modal_history.cpu().numpy()
        
        self.forecaster = LinearForecaster(use_torch=True)
        metrics = self.forecaster.train(x_history, verbose=self.config.verbose)
        
        if self.config.verbose:
            r2 = metrics.get('r2_score', 'N/A')
            if isinstance(r2, (int, float)):
                logger.info(f"   Linear forecaster R²: {r2:.4f}")
            else:
                logger.info(f"   Linear forecaster R²: {r2}")
    
    def _train_mlp_forecaster(
        self,
        modal_history: torch.Tensor,
        n_modes: int,
        config: Dict[str, Any]
    ):
        """Обучить MLP forecaster"""
        x_history = modal_history.cpu().numpy()
        
        # Параметры из config
        hidden_dim = config.get('hidden_size', 256)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.3)
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-5)
        
        self.forecaster = MLPForecaster(
            in_dim=n_modes,
            out_dim=n_modes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            device=self.config.device
        )
        
        # Обучение
        history = self.forecaster.train(
            x_history,
            num_epochs=self.config.epochs if hasattr(self.config, 'epochs') else 300,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            val_split=self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2,
            early_stopping_patience=self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 20,
            verbose=self.config.verbose
        )
        
        if self.config.verbose:
            final_loss = history.get('train_losses', [0])[-1] if history else 0
            logger.info(f"   MLP forecaster final loss: {final_loss:.6f}")
    
    def _train_lstm_forecaster(
        self,
        modal_history: torch.Tensor,
        n_modes: int,
        config: Dict[str, Any]
    ):
        """Обучить LSTM forecaster"""
        x_history = modal_history.cpu().numpy()
        
        # Параметры из config
        hidden_dim = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 1)
        dropout = config.get('dropout', 0.0)
        seq_length = config.get('seq_length', 5)
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-5)
        
        self.forecaster = LSTMForecaster(
            in_dim=n_modes,
            out_dim=n_modes,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            lr=lr,
            weight_decay=weight_decay,
            device=self.config.device
        )
        
        # Обучение
        history = self.forecaster.train(
            x_history,
            num_epochs=self.config.epochs if hasattr(self.config, 'epochs') else 300,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            val_split=self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2,
            early_stopping_patience=self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 20,
            verbose=self.config.verbose
        )
        
        if self.config.verbose:
            final_loss = history.get('train_losses', [0])[-1] if history else 0
            logger.info(f"   LSTM forecaster final loss: {final_loss:.6f}")
    
    def _project_to_modal_space(self, state: torch.Tensor) -> torch.Tensor:
        """
        Проецировать состояние на модальное пространство.
        
        Использует правильную tensorly операцию для проекции на A_tensor.
        
        Args:
            state: Пространственное состояние (spatial_shape)
            
        Returns:
            Модальные коэффициенты (n_modes,)
        """
        # A_tensor это modal basis, нужно найти коэффициенты x такие что:
        # state ≈ A_tensor @ x
        # Решаем через least squares: x = (A^T A)^{-1} A^T state
        
        A_tensor = self.spatial_modes
        state_flat = state.reshape(-1)
        
        # Solve least squares
        try:
            # Используем torch.linalg.lstsq для численно стабильного решения
            if A_tensor.ndim == 2:
                # Если A_tensor это матрица (spatial_points, n_modes)
                x_modal = torch.linalg.lstsq(A_tensor, state_flat.unsqueeze(-1)).solution.squeeze(-1)
            else:
                # Если A_tensor это тензор, flatten первую размерность
                A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                x_modal = torch.linalg.lstsq(A_flat, state_flat.unsqueeze(-1)).solution.squeeze(-1)
        except Exception as e:
            logger.warning(f"Least squares failed, using transpose: {e}")
            # Fallback: простая проекция через транспонирование
            if A_tensor.ndim == 2:
                x_modal = A_tensor.T @ state_flat
            else:
                A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                x_modal = A_flat.T @ state_flat
        
        return x_modal
    
    def _reconstruct_from_modal(self, modal_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Реконструировать пространственное поле из модальных коэффициентов.
        
        Args:
            modal_coeffs: Модальные коэффициенты (n_modes,) или (n_modes, n_steps)
            
        Returns:
            Реконструированное поле (spatial_shape) или (spatial_shape, n_steps)
        """
        A_tensor = self.spatial_modes
        
        # Если modal_coeffs это вектор
        if modal_coeffs.ndim == 1:
            reconstructed = reconstruct_tensor(
                A_tensor=A_tensor,
                x_hat=modal_coeffs,
                zero_threshold=1e-6,
                decimals=4
            )
            
            if reconstructed is None:
                # Fallback: простое умножение
                if A_tensor.ndim == 2:
                    reconstructed = A_tensor @ modal_coeffs
                else:
                    A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                    reconstructed = A_flat @ modal_coeffs
        else:
            # Если modal_coeffs это матрица (n_modes, n_steps)
            n_steps = modal_coeffs.shape[1]
            reconstructed_list = []
            
            for t in range(n_steps):
                rec_t = reconstruct_tensor(
                    A_tensor=A_tensor,
                    x_hat=modal_coeffs[:, t],
                    zero_threshold=1e-6,
                    decimals=4
                )
                
                if rec_t is None:
                    # Fallback
                    if A_tensor.ndim == 2:
                        rec_t = A_tensor @ modal_coeffs[:, t]
                    else:
                        A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                        rec_t = A_flat @ modal_coeffs[:, t]
                
                reconstructed_list.append(rec_t)
            
            reconstructed = torch.stack(reconstructed_list, dim=-1)
        
        return reconstructed
    
    def predict(
        self,
        current_state: torch.Tensor,
        n_steps: int = 1,
        return_full_field: bool = True,
        use_history: Optional[bool] = None,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прогнозировать будущие состояния используя обученную модель
        
        Правильный workflow:
        1. Проекция на модальное пространство
        2. Прогноз модальных коэффициентов через forecaster (Linear/MLP/LSTM)
        3. Реконструкция полного поля
        
        Args:
            current_state: Текущее состояние (spatial_shape)
            n_steps: Количество шагов прогноза
            return_full_field: Вернуть полное поле или только коэффициенты
            use_history: Для LSTM - использовать историю (если доступна)
            history: История состояний (spatial_shape, history_len) для инициализации LSTM
            
        Returns:
            Прогноз (spatial_shape, n_steps) если return_full_field=True
            или модальные коэффициенты (n_modes, n_steps)
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin не обучен. Сначала вызовите train()")
        
        current_state = current_state.to(device=self.device, dtype=self.dtype)
        
        # Нормализация
        if self.mean is not None:
            state_norm = (current_state - self.mean.squeeze(-1)) / self.std.squeeze(-1)
        else:
            state_norm = current_state
            
        # Обработка истории если передана
        modal_history_tensor = None
        if history is not None:
            history = history.to(device=self.device, dtype=self.dtype)
            if self.mean is not None:
                history_norm = (history - self.mean) / self.std # Broadcast dimensions?
                # mean/std shape depends on data. usually (spatial, 1)
                # history shape (spatial, T)
                # (spatial, T) - (spatial, 1) works.
            else:
                history_norm = history
            
            # Project history
            T_hist = history_norm.shape[-1]
            modal_seq = []
            for t in range(T_hist):
                st = history_norm[..., t]
                mc = self._project_to_modal_space(st)
                modal_seq.append(mc)
            modal_history_tensor = torch.stack(modal_seq, dim=0) # (T, n_modes)
        
        # ========================================================================
        # Step 1: Проекция на модальное пространство
        # ========================================================================
        modal_current = self._project_to_modal_space(state_norm)
        
        # ========================================================================
        # Step 2: Прогноз модальных коэффициентов через Forecaster
        # ========================================================================
        modal_forecast = self._forecast_modal_coefficients(
            modal_current, 
            n_steps, 
            use_history, 
            external_history=modal_history_tensor
        )
        
        # Сохранить в состояние
        self.state.modal_coefficients = modal_forecast
        
        if not return_full_field:
            return modal_forecast
        
        # ========================================================================
        # Step 3: Реконструкция полного поля
        # ========================================================================
        forecast = self._reconstruct_from_modal(modal_forecast)
        
        # Reshape если нужно
        if forecast.ndim > current_state.ndim:
            pass  # forecast уже имеет форму (spatial_shape, n_steps)
        else:
            forecast = forecast.unsqueeze(-1).repeat(1, 1, n_steps) if current_state.ndim == 2 else forecast.unsqueeze(-1)
        
        # Денормализация
        if self.mean is not None:
            if forecast.ndim == 3:
                forecast = forecast * self.std + self.mean
            else:
                forecast = forecast * self.std.squeeze(-1) + self.mean.squeeze(-1)
        
        return forecast

    def _forecast_modal_coefficients(
        self,
        modal_current: torch.Tensor,
        n_steps: int,
        use_history: Optional[bool] = None,
        external_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прогнозировать модальные коэффициенты используя обученный forecaster.
        
        Args:
            modal_current: Текущие модальные коэффициенты (n_modes,)
            n_steps: Количество шагов прогноза
            use_history: Использовать историю для LSTM
            external_history: Внешняя история (T, n_modes) для инициализации
            
        Returns:
            Прогноз модальных коэффициентов (n_modes, n_steps)
        """
        # Автовыбор использования истории для LSTM, если параметр не задан явно
        if use_history is None:
            use_history = self.forecaster_type == ForecasterType.LSTM

        # Persistence (fallback или если forecaster не обучен)
        if self.forecaster is None or self.forecaster_type == ForecasterType.PERSISTENCE:
            return modal_current.unsqueeze(1).repeat(1, n_steps)
        
        # Используем forecaster
        x_current = modal_current.cpu().numpy()
        
        try:
            if self.forecaster_type == ForecasterType.LINEAR:
                # Linear forecaster: predict_sequence
                future_seq = self.forecaster.predict_sequence(x_current, n_steps=n_steps)
                # future_seq shape: (n_steps, n_modes)
                
            elif self.forecaster_type == ForecasterType.MLP:
                # MLP forecaster: predict_sequence
                future_seq = self.forecaster.predict_sequence(x_current, n_steps=n_steps)
                # future_seq shape: (n_steps, n_modes)
                
            elif self.forecaster_type == ForecasterType.LSTM:
                # LSTM forecaster: нужна последовательность для входа
                seq_length = self.forecaster.seq_length if hasattr(self.forecaster, 'seq_length') else 5
                
                if external_history is not None:
                    # Use external history
                    history = external_history.cpu().numpy()
                    if len(history) >= seq_length:
                        x_window = history[-seq_length:]
                    else:
                        x_window = np.tile(x_current, (seq_length, 1))
                        x_window[-len(history):] = history
                elif use_history and self._modal_history is not None:
                    history = self._modal_history.cpu().numpy()
                    if len(history) >= seq_length:
                        x_window = history[-seq_length:]
                    else:
                        x_window = np.tile(x_current, (seq_length, 1))
                        x_window[-len(history):] = history
                else:
                    # Создать окно из текущего состояния
                    x_window = np.tile(x_current, (seq_length, 1))
                
                future_seq = self.forecaster.predict_sequence(x_window, n_steps=n_steps)
                # future_seq shape: (n_steps, n_modes)
            else:
                # Fallback to persistence
                return modal_current.unsqueeze(1).repeat(1, n_steps)
            
            # Конвертировать в torch и транспонировать в (n_modes, n_steps)
            modal_forecast = torch.tensor(
                future_seq, 
                device=self.device, 
                dtype=self.dtype
            ).T  # (n_steps, n_modes) -> (n_modes, n_steps)
            
            return modal_forecast
            
        except Exception as e:
            logger.warning(f"Forecaster prediction failed: {e}. Using persistence.")
            return modal_current.unsqueeze(1).repeat(1, n_steps)
    
    def _prepare_sensor_measurements(self, sensor_readings: torch.Tensor) -> torch.Tensor:
        """
        Унифицировать формат измерений сенсоров.
        
        Поддерживаемые форматы:
        - Тензор формы (spatial_shape[, ...]) с ненулевыми значениями на позициях сенсоров.
        - Тензор формы (n_sensors[, ...]) где порядок соответствует self.sensor_indices.
        
        Возвращает тензор формы spatial_shape или spatial_shape + trailing_dims.
        """
        if self.sensor_mask is None or self.sensor_indices is None:
            raise ValueError("Сенсоры еще не размещены. Выполните train().")
        
        readings = sensor_readings.to(device=self.device, dtype=self.dtype)
        spatial_shape = self.sensor_mask.shape
        flat_mask = self.sensor_mask.reshape(-1)
        n_sensors = int(flat_mask.sum().item())
        
        # Если уже передана полная маска (возможно с временным измерением)
        if readings.shape[:len(spatial_shape)] == spatial_shape:
            return readings
        
        # Формат (n_sensors, ...) – раскладываем по маске
        if readings.shape[0] == n_sensors:
            trailing = readings.shape[1:]
            full_flat = torch.zeros((flat_mask.numel(),) + trailing, device=self.device, dtype=self.dtype)
            full_flat[flat_mask.bool()] = readings
            return full_flat.reshape(spatial_shape + trailing)
        
        raise ValueError(
            f"sensor_readings форма {readings.shape} не совместима: ожидалась {spatial_shape} "
            f"или ({n_sensors}, ...)"
        )
    
    def update_from_sensors(
        self,
        sensor_readings: torch.Tensor,
        timestamp: Optional[float] = None
    ) -> torch.Tensor:
        """
        Обновить состояние из измерений сенсоров
        
        Правильная логика TBMD CS:
        1. Построить Y (full-size tensor с измерениями на позициях сенсоров)
        2. Создать solver с A_tensor, P, Y
        3. Решить x_hat = solver.solve()
        4. Реконструировать поле: X_reconstructed = A_tensor @ x_hat
        
        Args:
            sensor_readings: Измерения с сенсоров. Поддерживается либо полный тензор 
                             формы spatial_shape (ненулевые значения только на сенсорах),
                             либо массив длиной n_sensors (или n_sensors × ... для батча/времени).
            timestamp: Временная метка
            
        Returns:
            Реконструированное полное поле (spatial_shape)
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin не обучен")
        
        if self.mean is None:
            logger.warning(
                "Нормализация не настроена внутри DigitalTwin. "
                "Ожидаются заранее нормализованные измерения сенсоров."
            )
        
        # Приводим измерения к полной форме spatial_shape (ненулевые значения на сенсорах)
        Y = self._prepare_sensor_measurements(sensor_readings)
        
        # P - это бинарная маска сенсоров
        P = self.sensor_mask
        
        # A_tensor - это modal basis (уже есть в self.spatial_modes)
        A_tensor = self.spatial_modes
        
        spatial_shape = self._spatial_shape if self._spatial_shape is not None else Y.shape
        spatial_ndim = len(spatial_shape)
        
        # Поддержка батчей/времени: раскладываем trailing_dims в последовательность срезов
        if Y.ndim == spatial_ndim:
            Y_slices = [Y]
            trailing_shape: Tuple[int, ...] = ()
        else:
            trailing_shape = Y.shape[spatial_ndim:]
            Y_flat = Y.reshape(spatial_shape + (-1,))
            Y_slices = [Y_flat[..., i] for i in range(Y_flat.shape[-1])]
        
        reconstructed_slices = []
        
        sensor_errors: List[float] = []
        for idx, Y_slice in enumerate(Y_slices):
            # ====================================================================
            # Создать Compressive Sensing solver
            # ====================================================================
            cs_config = CompressiveSensingConfig(
                max_iter=self.config.max_iterations if hasattr(self.config, 'max_iterations') else 1000,
                tol=1e-4,
                epsilon_l1=1e-2,
                delta_init=1.0,
                delta_max=1.0,
                relax_lambda=0.95,
                device=self.config.device,
                dtype=self.dtype
            )
            
            ext_config = ExtensionCompressiveSensingConfig(
                solver="cholesky",
                reg=1e-8,
                delta_policy="boyd",
                stop_policy="residual",
                relative_window=5,
                relative_drop=1e-3,
                collect_history=True
            )
            
            solver = TensorCompressiveSensing(
                A_tensor,  # Modal basis
                P,         # Sensor mask  
                Y_slice,   # Measurements (full-size с нулями)
                cs_config,
                ext_config
            )
            
            # Решить и реконструировать
            x_hat, metrics = solver.solve()
            
            if self.config.verbose:
                logger.info(f"CS Reconstruction slice {idx}: converged={metrics.converged}, "
                           f"iters={metrics.iterations}, obj={metrics.objective:.4e}")
            
            X_reconstructed = reconstruct_tensor(
                A_tensor=A_tensor,
                x_hat=x_hat,
                zero_threshold=1e-4,
                decimals=3
            )
            
            if X_reconstructed is None:
                raise RuntimeError("Reconstruction failed")
            
            # Ошибка на сенсорах
            sensor_err = torch.norm((X_reconstructed - Y_slice)[self.sensor_mask]).item()
            sensor_errors.append(sensor_err)
            
            reconstructed_slices.append(X_reconstructed)
        
        if len(reconstructed_slices) == 1:
            reconstructed = reconstructed_slices[0]
        else:
            reconstructed = torch.stack(reconstructed_slices, dim=-1).reshape(spatial_shape + trailing_shape)
        
        # Денормализация
        if self.mean is not None:
            if reconstructed.ndim == len(spatial_shape):
                reconstructed = reconstructed * self.std.squeeze(-1) + self.mean.squeeze(-1)
            else:
                reconstructed = reconstructed * self.std + self.mean
        
        # Обновить состояние
        if timestamp is not None:
            self.state.current_time = timestamp
        
        # Обновить метрики/коэффициенты по последнему срезу
        last_reconstructed = reconstructed if reconstructed.ndim == len(spatial_shape) else reconstructed[..., -1]
        if self.mean is not None:
            last_norm = (last_reconstructed - self.mean.squeeze(-1)) / self.std.squeeze(-1)
        else:
            last_norm = last_reconstructed
        self.state.modal_coefficients = self._project_to_modal_space(last_norm).unsqueeze(1)
        if sensor_errors:
            self.state.prediction_error = sensor_errors[-1]
            self.state.history['errors'].append(sensor_errors[-1])
        self.state.history['observations'].append(reconstructed.cpu().numpy())
        
        # Обновить proxy model, если есть
        if self.proxy_model is not None:
            observed_state = self.create_reservoir_state(
                pressure=last_reconstructed,
                time=self.state.current_time
            )
            try:
                self.proxy_model.update_from_observations(observed_state, self.sensor_mask)
            except Exception as e:
                logger.warning(f"Proxy update failed: {e}")
        
        return reconstructed
    
    def evaluate_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        n_steps: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Оценить несколько сценариев развития
        
        Args:
            scenarios: Список сценариев с параметрами
                      Каждый сценарий должен содержать:
                      - 'name': название сценария
                      - 'initial_state': начальное состояние (optional)
            n_steps: Длина прогноза
            
        Returns:
            Словарь {scenario_name: metrics}
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin не обучен")
        
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', f"scenario_{len(results)}")
            
            # Получить начальное состояние для сценария
            if 'initial_state' in scenario:
                initial_state = scenario['initial_state']
            elif self.state.modal_coefficients is not None:
                # Реконструировать текущее состояние из модальных коэффициентов
                initial_state = self._reconstruct_from_modal(
                    self.state.modal_coefficients[:, 0]
                )
            else:
                # Нет данных для прогноза
                logger.warning(f"No initial state for scenario {scenario_name}, skipping")
                continue
            
            try:
                # Сделать прогноз
                forecast = self.predict(
                    current_state=initial_state,
                    n_steps=n_steps,
                    return_full_field=True
                )
                
                # Вычислить метрики
                metrics = {
                    'mean_value': forecast.mean().item(),
                    'std_value': forecast.std().item(),
                    'max_value': forecast.max().item(),
                    'min_value': forecast.min().item(),
                    'final_mean': forecast[..., -1].mean().item()
                }
                
                results[scenario_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating scenario {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results
    
    def detect_anomalies(
        self,
        sensor_data: torch.Tensor,
        threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Детектировать аномалии в данных сенсоров
        
        Использует правильный TBMD CS workflow для каждого временного шага.
        
        Args:
            sensor_data: Данные с сенсоров (spatial_shape, n_timesteps)
                        где ненулевые значения только на позициях сенсоров
            threshold: Порог для детекции (в сигмах)
            
        Returns:
            Список обнаруженных аномалий с timestamp, residual, severity
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin не обучен")
        
        anomalies = []
        # Приводим данные к полной форме (spatial_shape[, time])
        sensor_tensor = self._prepare_sensor_measurements(
            sensor_data.to(device=self.device, dtype=self.dtype)
        )
        spatial_ndim = len(self._spatial_shape) if self._spatial_shape is not None else sensor_tensor.ndim
        if sensor_tensor.ndim == spatial_ndim:
            sensor_tensor = sensor_tensor.unsqueeze(-1)
        else:
            sensor_tensor = sensor_tensor.reshape(self._spatial_shape + (-1,))
        
        # Prepare CS configs (создаем один раз для эффективности)
        cs_config = CompressiveSensingConfig(
            max_iter=self.config.max_iterations if hasattr(self.config, 'max_iterations') else 1000,
            tol=1e-4,
            epsilon_l1=1e-2,
            delta_init=1.0,
            delta_max=1.0,
            relax_lambda=0.95,
            device=self.config.device,
            dtype=self.dtype
        )
        
        ext_config = ExtensionCompressiveSensingConfig(
            solver="cholesky",
            reg=1e-8,
            delta_policy="boyd",
            stop_policy="residual",
            relative_window=5,
            relative_drop=1e-3,
            collect_history=False  # Не собираем историю для скорости
        )
        
        A_tensor = self.spatial_modes
        P = self.sensor_mask
        
        # Реконструкция для каждого временного шага
        n_timesteps = sensor_tensor.shape[-1]
        
        for t in range(n_timesteps):
            try:
                # Извлечь измерения для текущего шага
                Y = sensor_tensor[..., t]
                
                # Создать solver
                solver = TensorCompressiveSensing(
                    A_tensor,
                    P,
                    Y,
                    cs_config,
                    ext_config
                )
                
                # Решить
                x_hat, metrics = solver.solve()
                
                # Вычислить reconstruction error
                X_reconstructed = reconstruct_tensor(
                    A_tensor=A_tensor,
                    x_hat=x_hat,
                    zero_threshold=1e-6,
                    decimals=4
                )
                
                if X_reconstructed is not None:
                    # Вычислить residual
                    residual = torch.norm((X_reconstructed - Y)[self.sensor_mask]).item()
                    
                    # Определить порог аномалии
                    if self.std is not None:
                        threshold_value = threshold * self.std.mean().item()
                    else:
                        threshold_value = threshold
                    
                    # Проверить на аномалию
                    if residual > threshold_value:
                        severity = 'high' if residual > 5 * threshold_value else 'medium'
                        anomalies.append({
                            'timestamp': t,
                            'residual': residual,
                            'severity': severity,
                            'threshold': threshold_value,
                            'converged': metrics.converged
                        })
                
            except Exception as e:
                logger.warning(f"Ошибка реконструкции на шаге {t}: {e}")
                anomalies.append({
                    'timestamp': t,
                    'error': str(e),
                    'severity': 'error'
                })
        
        return anomalies
    
    def get_sensor_locations(self) -> np.ndarray:
        """Получить индексы размещенных сенсоров"""
        if self.sensor_indices is None:
            raise ValueError("Сенсоры еще не размещены")
        return self.sensor_indices.cpu().numpy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику работы twin"""
        return {
            'is_calibrated': self.state.is_calibrated,
            'current_time': self.state.current_time,
            'n_spatial_modes': self.config.n_spatial_modes,
            'n_sensors': self.config.n_sensors,
            'modal_dim': int(self.spatial_modes.shape[-1]) if self.spatial_modes is not None else None,
            'sensors_placed': int(self.sensor_indices.numel()) if self.sensor_indices is not None else 0,
            'alert_status': self.state.alert_status,
            'history_length': len(self.state.history['observations']),
            'proxy_model_type': self.proxy_model_type.value if self.proxy_model_type else None,
            'forecaster_type': self.forecaster_type.value if self.forecaster_type else None
        }
    
    # ==========================================================================
    # PROXY MODEL METHODS (как в system.py)
    # ==========================================================================
    
    def _init_proxy_model(self):
        """
        Инициализировать proxy model для сценарного анализа с well controls.
        
        Как в system.py - создаёт LinearDynamicsProxyModel, NeuralProxyModel 
        или PhysicsInformedProxyModel на основе modal basis.
        """
        if self.spatial_modes is None:
            raise ValueError("Modal basis не вычислен. Сначала выполните decomposition.")
        
        # Flatten modal basis если нужно
        if self.spatial_modes.ndim == 2:
            modal_basis = self.spatial_modes
        else:
            modal_basis = self.spatial_modes.reshape(-1, self.spatial_modes.shape[-1])
        
        spatial_shape = self._spatial_shape if self._spatial_shape else (modal_basis.shape[0],)
        
        if self.proxy_model_type == ProxyModelType.LINEAR_DYNAMICS:
            self.proxy_model = LinearDynamicsProxyModel(
                spatial_shape=spatial_shape,
                modal_basis=modal_basis,
                device=self.config.device,
                dtype=self.dtype
            )
            if self.config.verbose:
                logger.info("✅ LinearDynamicsProxyModel инициализирован")
                
        elif self.proxy_model_type == ProxyModelType.NEURAL:
            hidden_layers = getattr(self.config, 'proxy_hidden_layers', [128, 64])
            self.proxy_model = NeuralProxyModel(
                spatial_shape=spatial_shape,
                modal_basis=modal_basis,
                hidden_layers=hidden_layers,
                device=self.config.device,
                dtype=self.dtype
            )
            if self.config.verbose:
                logger.info(f"✅ NeuralProxyModel инициализирован (hidden={hidden_layers})")
                
        elif self.proxy_model_type == ProxyModelType.PHYSICS_INFORMED:
            porosity = getattr(self.config, 'porosity', None)
            permeability = getattr(self.config, 'permeability', None)
            self.proxy_model = PhysicsInformedProxyModel(
                spatial_shape=spatial_shape,
                modal_basis=modal_basis,
                porosity=porosity,
                permeability=permeability,
                device=self.config.device,
                dtype=self.dtype
            )
            if self.config.verbose:
                logger.info("✅ PhysicsInformedProxyModel инициализирован")
    
    def _build_proxy_training_sets(
        self,
        data_dict: Dict[str, torch.Tensor]
    ) -> Tuple[List[ReservoirState], List[List[WellControl]]]:
        """
        Подготовить исторические состояния и простые well controls для калибровки proxy.
        Используется первый субъект в data_dict.
        """
        first_key = next(iter(data_dict))
        data = data_dict[first_key]
        T = data.shape[-1]
        states: List[ReservoirState] = []
        controls: List[List[WellControl]] = []
        zero_location = tuple(0 for _ in range(len(self._spatial_shape))) if self._spatial_shape else (0, 0)
        
        for t in range(T):
            pressure = data[..., t]
            state = self.create_reservoir_state(pressure=pressure, time=float(t))
            states.append(state)
            controls.append([
                self.create_well_control(
                    well_name="dummy",
                    control_type="rate",
                    value=0.0,
                    location=zero_location
                )
            ])
        
        return states, controls
    
    def calibrate_proxy_model(
        self,
        historical_states: List[ReservoirState],
        historical_controls: List[List[WellControl]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Калибровать proxy model на исторических данных.
        
        Как в system.py - обучает proxy model предсказывать динамику резервуара
        с учётом well controls.
        
        Args:
            historical_states: Список исторических состояний резервуара
            historical_controls: Список well controls для каждого временного шага
            **kwargs: Дополнительные параметры для калибровки
                - regularization: float для LinearDynamicsProxyModel
                - epochs, learning_rate, batch_size: для NeuralProxyModel
            
        Returns:
            Метрики калибровки (mse, relative_error, и т.д.)
        """
        if self.proxy_model is None:
            raise ValueError("Proxy model не инициализирован. Установите proxy_model_type в config.")
        
        if self.config.verbose:
            logger.info(f"Калибровка {self.proxy_model_type.value} proxy model...")
        
        if isinstance(self.proxy_model, LinearDynamicsProxyModel):
            regularization = kwargs.get('regularization', 1e-4)
            metrics = self.proxy_model.calibrate(
                historical_states,
                historical_controls,
                regularization=regularization
            )
        elif isinstance(self.proxy_model, NeuralProxyModel):
            epochs = kwargs.get('epochs', getattr(self.config, 'epochs', 100))
            learning_rate = kwargs.get('learning_rate', 1e-3)
            batch_size = kwargs.get('batch_size', 32)
            metrics = self.proxy_model.train_model(
                historical_states,
                historical_controls,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
        elif isinstance(self.proxy_model, PhysicsInformedProxyModel):
            regularization = kwargs.get('regularization', 1e-4)
            metrics = self.proxy_model.calibrate(
                historical_states,
                historical_controls,
                regularization=regularization
            )
        else:
            raise ValueError(f"Неизвестный тип proxy model: {type(self.proxy_model)}")
        
        if self.config.verbose:
            logger.info(f"✅ Proxy model откалиброван: {metrics}")
        
        return metrics
    
    def predict_with_controls(
        self,
        current_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float = 1.0,
        time_steps: int = 10
    ) -> List[ReservoirState]:
        """
        Прогнозировать состояние резервуара с учётом well controls.
        
        Как в system.py - использует proxy model для быстрого сценарного анализа.
        
        Args:
            current_state: Текущее состояние резервуара (ReservoirState)
            well_controls: Управление скважинами (WellControl)
            time_horizon: Горизонт прогноза
            time_steps: Количество временных шагов
            
        Returns:
            Список прогнозированных состояний
        """
        if self.proxy_model is None:
            raise ValueError("Proxy model не инициализирован или не откалиброван.")
        
        forecasted_states = self.proxy_model.forecast(
            current_state=current_state,
            well_controls=well_controls,
            time_horizon=time_horizon,
            time_steps=time_steps
        )
        
        return forecasted_states
    
    def evaluate_well_scenarios(
        self,
        initial_state: ReservoirState,
        scenarios: Dict[str, List[WellControl]],
        time_horizon: float = 10.0,
        time_steps: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Оценить несколько сценариев управления скважинами.
        
        Как ScenarioAnalyzer в system.py - быстрый what-if анализ.
        
        Args:
            initial_state: Начальное состояние резервуара
            scenarios: Словарь {scenario_name: well_controls}
            time_horizon: Горизонт прогноза
            time_steps: Количество шагов
            
        Returns:
            Словарь {scenario_name: {forecasted_states, kpis}}
        """
        if self.proxy_model is None:
            raise ValueError("Proxy model не инициализирован.")
        
        results = {}
        
        for scenario_name, well_controls in scenarios.items():
            if self.config.verbose:
                logger.info(f"Оценка сценария: {scenario_name}")
            
            try:
                # Прогноз
                forecasted_states = self.predict_with_controls(
                    current_state=initial_state,
                    well_controls=well_controls,
                    time_horizon=time_horizon,
                    time_steps=time_steps
                )
                
                # Вычислить KPIs
                kpis = self._compute_scenario_kpis(forecasted_states, well_controls)
                
                results[scenario_name] = {
                    'forecasted_states': forecasted_states,
                    'kpis': kpis,
                    'well_controls': well_controls
                }
                
            except Exception as e:
                logger.error(f"Ошибка в сценарии {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results
    
    def _compute_scenario_kpis(
        self,
        forecasted_states: List[ReservoirState],
        well_controls: List[WellControl]
    ) -> Dict[str, float]:
        """
        Вычислить Key Performance Indicators для сценария.
        
        Как в system.py ScenarioAnalyzer._compute_kpis().
        """
        kpis = {}
        
        # Статистика давления
        pressures = [state.pressure for state in forecasted_states]
        avg_pressures = [torch.mean(p).item() for p in pressures]
        
        kpis['avg_pressure'] = float(np.mean(avg_pressures))
        kpis['min_pressure'] = float(np.min(avg_pressures))
        kpis['max_pressure'] = float(np.max(avg_pressures))
        kpis['pressure_std'] = float(np.std(avg_pressures))
        
        # Производство и инжекция
        production_wells = [ctrl for ctrl in well_controls if ctrl.value < 0]
        injection_wells = [ctrl for ctrl in well_controls if ctrl.value > 0]
        
        kpis['total_production'] = abs(sum(ctrl.value for ctrl in production_wells)) * len(forecasted_states)
        kpis['total_injection'] = sum(ctrl.value for ctrl in injection_wells) * len(forecasted_states)
        kpis['net_production'] = kpis['total_production'] - kpis['total_injection']
        
        # Количество активных скважин
        kpis['n_production_wells'] = len(production_wells)
        kpis['n_injection_wells'] = len(injection_wells)
        
        return kpis
    
    def create_reservoir_state(
        self,
        pressure: torch.Tensor,
        saturation: Optional[torch.Tensor] = None,
        time: float = 0.0,
        well_rates: Optional[Dict[str, float]] = None
    ) -> ReservoirState:
        """
        Создать ReservoirState из данных.
        
        Вспомогательный метод для удобного создания состояний.
        
        Args:
            pressure: Поле давления
            saturation: Поле насыщенности (опционально)
            time: Время
            well_rates: Дебиты скважин
            
        Returns:
            ReservoirState
        """
        return ReservoirState(
            pressure=pressure.to(device=self.device, dtype=self.dtype),
            saturation=saturation.to(device=self.device, dtype=self.dtype) if saturation is not None else None,
            time=time,
            well_rates=well_rates
        )
    
    def create_well_control(
        self,
        well_name: str,
        control_type: str,
        value: float,
        location: Tuple[int, ...]
    ) -> WellControl:
        """
        Создать WellControl.
        
        Вспомогательный метод для удобного создания well controls.
        
        Args:
            well_name: Имя скважины
            control_type: Тип контроля ('rate', 'pressure', 'bhp')
            value: Значение (положительное = инжекция, отрицательное = добыча)
            location: Координаты скважины
            
        Returns:
            WellControl
        """
        return WellControl(
            well_name=well_name,
            control_type=control_type,
            value=value,
            location=location
        )
    
    def update_from_observations(
        self,
        observed_state: ReservoirState,
        sensor_locations: torch.Tensor
    ) -> None:
        """
        Обновить proxy model по новым наблюдениям (data assimilation).
        
        Как в system.py - онлайн обновление модели.
        
        Args:
            observed_state: Наблюдаемое состояние
            sensor_locations: Позиции сенсоров
        """
        if self.proxy_model is not None:
            self.proxy_model.update_from_observations(observed_state, sensor_locations)
            if self.config.verbose:
                logger.info("Proxy model обновлён по наблюдениям")


# Alias для обратной совместимости
DigitalTwinTBMD = DigitalTwin
