"""Digital twin support for reservoir monitoring and forecasting.

This module combines TBMD with forecasting models to create a digital twin.

Complete workflow:
1. Tucker Decomposition -> cores + factors
2. Modal Tensor Processing -> A_tensor (modal basis)
3. QR Factorization -> sensor placement (P matrix)
4. Compressive Sensing -> reconstruction from sparse sensors
5. Forecasting -> next-state prediction (Linear/MLP/LSTM)
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

# Config imports
from TBMD.config import DigitalTwinConfig
from TBMD.config import (
    DecompositionConfig, 
    DigitalTwinConfig, 
    ModalProcessorConfig,
    ProcessingStrategy
)
from TBMD.config import SensorPlacementConfig
from TBMD.config import CompressiveSensingConfig, ExtensionCompressiveSensingConfig

# Core TBMD imports
from TBMD.core.decomposition.hosvd import TuckerDecomposer
from TBMD.core.modal_processor.modes import ProcessingStrategy, BatchModalProcessor, ModalTensorStacker
from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition
from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing

# Forecasting Models
from TBMD.core.forecasting.LinearForecaster import LinearForecaster
from TBMD.core.forecasting.MLPForecaster import MLPForecaster
from TBMD.core.forecasting.LSTMForecaster import LSTMForecaster

# Reservoir proxy models
from TBMD.core.forecasting.ReservoirProxyModel import (
    ReservoirProxyModelBase,
    LinearDynamicsProxyModel,
    NeuralProxyModel,
    PhysicsInformedProxyModel,
    ReservoirState,
    WellControl
)

# Forecaster Configs
from TBMD.config import (
    LinearForecasterConfig,
    MLPForecasterConfig,
    LSTMForecasterConfig
)

# Utils
from TBMD.core.utils.misc import reconstruct_tensor, to_torch_tensor

logger = logging.getLogger(__name__)


class ForecasterType(Enum):
    """Forecasting model types for modal coefficients."""
    LINEAR = "linear"
    MLP = "mlp"
    LSTM = "lstm"
    PERSISTENCE = "persistence"  # Repeats the current state.


class ProxyModelType(Enum):
    """Proxy model types for physics-oriented reservoir forecasting."""
    LINEAR_DYNAMICS = "linear_dynamics"   # x(t+1) = A @ x(t) + B @ u(t)
    NEURAL = "neural"                     # Neural network proxy
    PHYSICS_INFORMED = "physics_informed" # Uses physical constraints


# Type alias for ranks
from typing import Union as UnionType
RanksType = UnionType[int, list]



@dataclass
class DigitalTwinState:
    """Current digital twin state.
    
    Attributes:
        current_time: Current time.
        modal_coefficients: Current modal coefficients.
        prediction_error: Prediction error.
        is_calibrated: Whether the twin has been calibrated.
        alert_status: Alert status ('normal', 'warning', 'critical').
        history: Measurement and forecast history.
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
    """Reservoir digital twin built on TBMD.
    
    Combines:
    1. TBMD decomposition for dimensionality reduction.
    2. Optimal sensor placement.
    3. Full-field reconstruction from measurements.
    4. Future-state forecasting.
    5. Monitoring and anomaly detection.
    
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
            config: Digital twin configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # State
        self.state = DigitalTwinState()
        
        # TBMD components
        self.decomposer = None
        self.sensor_placer = None
        self.reconstructor = None
        
        # Forecaster for modal coefficients
        self.forecaster = None
        self.forecaster_type = ForecasterType(config.forecaster_type) if hasattr(config, 'forecaster_type') else ForecasterType.PERSISTENCE
        self._modal_history = None  # Modal coefficient history for forecaster training
        
        # Proxy model for scenario analysis with well controls
        self.proxy_model: Optional[ReservoirProxyModelBase] = None
        # Check for None before enum conversion
        _proxy_type = getattr(config, 'proxy_model_type', None)
        self.proxy_model_type = ProxyModelType(_proxy_type) if _proxy_type is not None else None
        self._spatial_shape: Optional[Tuple[int, ...]] = None
        
        # Trained parameters
        self.spatial_modes = None
        self.temporal_modes = None
        self.core_tensor = None
        self.sensor_mask = None          # Boolean mask on the spatial grid
        self.sensor_indices = None       # Linear sensor indices
        self.measurement_matrix = None   # Sensor mask in a CS-compatible format
        
        # Statistics
        self.mean = None
        self.std = None
        
        # Monitor
        self.monitor = None
        
        if config.verbose:
            logger.info(f"Digital Twin initialized: {config.n_spatial_modes} modes, {config.n_sensors} sensors")
    
    def _validate_tensor_shape(self, tensor: torch.Tensor, expected_dims: int, param_name: str):
        """Validate an input tensor shape.
        
        Args:
            tensor: Tensor to validate.
            expected_dims: Expected number of dimensions.
            param_name: Parameter name used in error messages.
            
        Raises:
            ValueError: If the tensor shape or values are invalid.
        """
        if tensor.ndim != expected_dims:
            raise ValueError(
                f"{param_name} must have {expected_dims} dimensions, "
                f"got {tensor.ndim} with shape {tensor.shape}"
            )
        
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{param_name} contains NaN or Inf values")
        
        if tensor.numel() == 0:
            raise ValueError(f"{param_name} cannot be empty")
    
    def train(
        self,
        historical_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        normalize: bool = False,
        ranks: Optional[RanksType] = None
    ):
        """Train the digital twin on historical data.
        
        Expected sequence:
        1. Tucker Decomposition with config -> cores + factors
        2. Modal Tensor Processing -> A_tensor
        3. QR Factorization with config -> sensor placement
        
        Args:
            historical_data: Historical data as a torch.Tensor of arbitrary
                dimensionality or a dict of named tensors for multiple subjects.
            normalize: Normalize data inside the method. Keep False when data
                has already been normalized externally.
            ranks: Tucker decomposition ranks. If None, ranks are derived from
                configuration and data dimensions.
        """
        # Input conversion
        # Reset normalization statistics. Normalize data before calling this method if needed.
        self.mean, self.std = None, None
        if isinstance(historical_data, dict):
            data_dict = {
                k: to_torch_tensor(v, device=self.device, dtype=self.dtype)
                for k, v in historical_data.items()
            }
        else:
            historical_data = to_torch_tensor(historical_data, device=self.device, dtype=self.dtype)
            data_dict = {"train": historical_data}
        
        # All tensors must have the same shape.
        shapes = {v.shape for v in data_dict.values()}
        if len(shapes) != 1:
            raise ValueError(f"All tensors must have the same shape, got: {shapes}")
        
        sample_tensor = next(iter(data_dict.values()))
        
        # Input validation
        if sample_tensor.ndim < 3:
            raise ValueError(
                f"historical_data must have at least 3 dimensions (spatial_dims..., time), "
                f"got {sample_tensor.ndim}"
            )
        
        self._original_ndim = sample_tensor.ndim
        self._spatial_shape = sample_tensor.shape[:-1]
        
        if self.config.verbose:
            logger.info(f"Starting Digital Twin training on data with shape {sample_tensor.shape}")
        
        # ========================================================================
        # Step 1: TBMD Tucker decomposition
        # ========================================================================
        # Determine ranks.
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
                # For 4D+ data, the first N-1 dimensions are spatial and the last is temporal.
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
        
        # Create decomposer with config.
        self.decomposer = TuckerDecomposer(
            tensors=data_dict,
            device=self.config.device,
            config=decomp_config
        )
        
        self.decomposer.decompose()
        
        # Extract cores and factors.
        cores = self.decomposer.cores
        factors = self.decomposer.factors
        
        if self.config.verbose:
            logger.info(f"Decomposition complete, ranks={effective_ranks}")
        
        # ========================================================================
        # Step 2: Modal tensor processing
        # ========================================================================
        modal_config = ModalProcessorConfig(
            device=self.config.device,
            processing_strategy=ProcessingStrategy.BATCH,
            enable_progress_logging=self.config.verbose,
            return_numpy=False
        )
        
        batch_processor = BatchModalProcessor(modal_config)
        stacker = ModalTensorStacker(modal_config)
        
        # Compute modal tensors.
        modal_tensors = batch_processor.process_multiple_subjects(cores, factors)
        
        # Stack into A_tensor (time-invariant modes).
        A_tensor = stacker.stack_modal_tensors(modal_tensors)
        
        # Store for later use.
        self.spatial_modes = A_tensor  # Modal basis
        self.core_tensor = cores
        self.temporal_modes = factors
        # Number of modes equals the last A_tensor dimension.
        modal_dim = A_tensor.shape[-1]
        # Adjust the sensor count to avoid an underdetermined CS problem.
        max_sensors = int(np.prod(self._spatial_shape))
        if self.config.n_sensors < modal_dim:
            adjusted = min(modal_dim, max_sensors)
            logger.warning(
                f"n_sensors={self.config.n_sensors} is smaller than the number of modes {modal_dim}; "
                f"setting n_sensors={adjusted} for stable reconstruction."
            )
            self.config.n_sensors = adjusted
        
        if self.config.verbose:
            logger.info(f"Modal tensor computed: {A_tensor.shape}")
        
        # ========================================================================
        # Step 3: Sensor placement through QR factorization
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
            logger.info("Running QR factorization...")
        
        P, Q, R = qr_decomposer.factorize()
        
        # Validation
        is_valid, error, metrics = qr_decomposer.check_factorization()
        
        if self.config.verbose:
            logger.info(f"QR factorization: valid={is_valid}, error={error:.2e}")
            logger.info(f"   Orthogonality deviation: {metrics['orthogonality_deviation']:.2e}")
            logger.info(f"   Sensors placed: {metrics['sensor_count']}/{qr_decomposer.N}")
        
        # Store results.
        self.sensor_mask = P.bool()
        self.sensor_indices = torch.nonzero(self.sensor_mask.reshape(-1), as_tuple=False).squeeze(-1)
        self.measurement_matrix = self.sensor_mask
        
        # ========================================================================
        # Step 4: Forecasting model training
        # ========================================================================
        # Prepare summary
        summary = {
            "ranks": effective_ranks,
            "modal_dim": modal_dim,
            "n_sensors": self.config.n_sensors,
            "qr_valid": is_valid,
            "qr_error": error,
            "qr_metrics": metrics
        }
        
        # Add forecaster metrics
        forecaster_metrics = self._train_forecaster(data_dict, sample_tensor)
        if forecaster_metrics:
            summary.update(forecaster_metrics)
            
        # Update state
        self.state.is_calibrated = True
        self.state.current_time = 0.0

        if self.config.verbose:
            logger.info("Digital Twin trained successfully")
            
        return summary
    
    def _train_forecaster(
        self,
        data_dict: Dict[str, torch.Tensor],
        sample_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Train the forecasting model on modal coefficients.
        
        Returns:
             Dict with training metrics
        """
        metrics = {}
        if self.forecaster_type == ForecasterType.PERSISTENCE:
            if self.config.verbose:
                logger.info("Forecaster: persistence (no training)")
            return {"forecaster": "persistence"}
        
        if self.config.verbose:
            logger.info(f"Training forecaster ({self.forecaster_type.value})...")
        
        # Project data into modal space.
        first_key = next(iter(data_dict))
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
        forecaster_config = getattr(self.config, 'forecaster_config', {})
        
        # Create and train the forecaster.
        if self.forecaster_type == ForecasterType.LINEAR:
            m = self._train_linear_forecaster(modal_history, forecaster_config)
            metrics.update(m)
        elif self.forecaster_type == ForecasterType.MLP:
            m = self._train_mlp_forecaster(modal_history, n_modes, forecaster_config)
            metrics.update(m)
        elif self.forecaster_type == ForecasterType.LSTM:
            m = self._train_lstm_forecaster(modal_history, n_modes, forecaster_config)
            metrics.update(m)
        
        if self.config.verbose:
            logger.info(f"Forecaster ({self.forecaster_type.value}) trained")
            
        return metrics
    
    def _train_linear_forecaster(
        self,
        modal_history: torch.Tensor,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a linear forecaster: x(t+1) = A @ x(t)."""
        # LinearForecaster expects NumPy input.
        x_history = modal_history.cpu().numpy()
        
        self.forecaster = LinearForecaster(use_torch=True)
        metrics = self.forecaster.train(x_history, verbose=self.config.verbose)
        
        if self.config.verbose:
            r2 = metrics.get('r2_score', 'N/A')
            if isinstance(r2, (int, float)):
                logger.info(f"   Linear forecaster R2: {r2:.4f}")
            else:
                logger.info(f"   Linear forecaster R2: {r2}")
        return metrics
    
    def _train_mlp_forecaster(
        self,
        modal_history: torch.Tensor,
        n_modes: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train an MLP forecaster."""
        x_history = modal_history.cpu().numpy()
        
        # Parameters from config.
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
        
        # Training
        history = self.forecaster.train(
            x_history,
            num_epochs=self.config.epochs if hasattr(self.config, 'epochs') else 300,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            val_split=self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2,
            early_stopping_patience=self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 20,
            verbose=self.config.verbose
        )
        
        final_loss = history.get('train_losses', [0])[-1] if history else 0
        if self.config.verbose:
            logger.info(f"   MLP forecaster final loss: {final_loss:.6f}")
            
        return {"mlp_history": history, "final_loss": final_loss}
    
    def _train_lstm_forecaster(
        self,
        modal_history: torch.Tensor,
        n_modes: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train an LSTM forecaster."""
        x_history = modal_history.cpu().numpy()
        
        # Parameters from config.
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
        
        # Training
        history = self.forecaster.train(
            x_history,
            num_epochs=self.config.epochs if hasattr(self.config, 'epochs') else 300,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            val_split=self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2,
            early_stopping_patience=self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 20,
            verbose=self.config.verbose
        )
        
        final_loss = history.get('train_losses', [0])[-1] if history else 0
        if self.config.verbose:
            logger.info(f"   LSTM forecaster final loss: {final_loss:.6f}")
            
        return {"lstm_history": history, "final_loss": final_loss}
    
    def _project_to_modal_space(self, state: torch.Tensor) -> torch.Tensor:
        """Project a state into modal space.
        
        Uses a tensor operation to project onto A_tensor.
        
        Args:
            state: Spatial state with shape spatial_shape.
            
        Returns:
            Modal coefficients with shape (n_modes,).
        """
        # A_tensor is the modal basis. Find coefficients x such that:
        # state ~= A_tensor @ x.
        # Solve via least squares: x = (A^T A)^{-1} A^T state.
        
        A_tensor = self.spatial_modes
        state_flat = state.reshape(-1)
        
        # Solve least squares
        try:
            # Use torch.linalg.lstsq for a numerically stable solution.
            if A_tensor.ndim == 2:
                # A_tensor is a matrix with shape (spatial_points, n_modes).
                x_modal = torch.linalg.lstsq(A_tensor, state_flat.unsqueeze(-1)).solution.squeeze(-1)
            else:
                # A_tensor is a tensor; flatten spatial dimensions.
                A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                x_modal = torch.linalg.lstsq(A_flat, state_flat.unsqueeze(-1)).solution.squeeze(-1)
        except Exception as e:
            logger.warning(f"Least squares failed, using transpose: {e}")
            # Fallback: simple transpose-based projection.
            if A_tensor.ndim == 2:
                x_modal = A_tensor.T @ state_flat
            else:
                A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                x_modal = A_flat.T @ state_flat
        
        return x_modal
    
    def _reconstruct_from_modal(self, modal_coeffs: torch.Tensor) -> torch.Tensor:
        """Reconstruct a spatial field from modal coefficients.
        
        Args:
            modal_coeffs: Modal coefficients with shape (n_modes,) or (n_modes, n_steps).
            
        Returns:
            Reconstructed field with shape spatial_shape or (spatial_shape, n_steps).
        """
        A_tensor = self.spatial_modes
        
        # modal_coeffs is a vector.
        if modal_coeffs.ndim == 1:
            reconstructed = reconstruct_tensor(
                A_tensor=A_tensor,
                x_hat=modal_coeffs,
                zero_threshold=1e-6,
                decimals=4
            )
            
            if reconstructed is None:
                # Fallback: simple multiplication.
                if A_tensor.ndim == 2:
                    reconstructed = A_tensor @ modal_coeffs
                else:
                    A_flat = A_tensor.reshape(-1, A_tensor.shape[-1])
                    reconstructed = A_flat @ modal_coeffs
        else:
            # modal_coeffs is a matrix with shape (n_modes, n_steps).
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
    
    def predict_next_state(self, current_state: torch.Tensor, controls: Any) -> torch.Tensor:
        """
        Predict the next state with control inputs.
        Delegates to proxy_model if available, otherwise uses internal forecaster (ignoring controls).
        """
        if self.proxy_model is not None:
             # Proxy model works with full state
             # Ensure state is tensor
             if not isinstance(current_state, torch.Tensor):
                 current_state = to_torch_tensor(current_state, device=self.device, dtype=self.dtype)
             
             prediction = self.proxy_model.predict_step(current_state, controls)
             return prediction
        else:
             # Fallback to internal forecaster (ignores controls)
             # Handle ReservoirState wrapper
             is_reservoir_state = isinstance(current_state, ReservoirState)
             if is_reservoir_state:
                 if current_state.saturation is not None:
                     # Stack pressure and saturation if separated
                     state_tensor = torch.stack([current_state.pressure, current_state.saturation], dim=-1)
                 else:
                     # Assume pressure holds the full state (e.g. multi-channel tensor)
                     state_tensor = current_state.pressure
             else:
                 state_tensor = current_state

             prediction = self.predict(state_tensor, n_steps=1, return_full_field=True)
             
             # Prediction is (spatial..., 1) if n_steps=1
             if prediction.ndim == state_tensor.ndim + 1:
                 prediction = prediction.squeeze(-1)
                 
             if is_reservoir_state:
                 # Unpack back to ReservoirState
                 if current_state.saturation is not None:
                     # Split back
                     return [ReservoirState(
                         pressure=prediction[..., 0],
                         saturation=prediction[..., 1]
                     )]
                 else:
                     # Keep combined
                     return [ReservoirState(pressure=prediction)]
             return [prediction]

    @property
    def sensor_locations(self) -> torch.Tensor:
        """
        Return sensor locations as a spatial mask.
        Used by compatibility scripts.
        """
        if self.sensor_indices is None:
            spatial_shape = self._spatial_shape if self._spatial_shape else self.config.spatial_shape
            return torch.zeros(spatial_shape, dtype=torch.bool, device=self.device)
            
        # Create mask
        # Flattened shape size
        # Use helper
        if self._spatial_shape is not None:
             dims = self._spatial_shape
        elif hasattr(self.config, 'spatial_shape'):
             dims = self.config.spatial_shape
        else:
             # Fallback: infer from sensor indices max? No.
             # Assume it was set during training
             raise ValueError("Spatial shape not defined. Digital Twin not trained?")

        n_points = int(np.prod(dims))
        mask_flat = torch.zeros(n_points, dtype=torch.bool, device=self.device)
        mask_flat[self.sensor_indices] = True
        
        return mask_flat
    
    def predict(
        self,
        current_state: torch.Tensor,
        n_steps: int = 1,
        return_full_field: bool = True,
        use_history: Optional[bool] = None,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forecast future states using the trained model.
        
        Workflow:
        1. Project into modal space.
        2. Forecast modal coefficients with the forecaster (Linear/MLP/LSTM).
        3. Reconstruct the full field.
        
        Args:
            current_state: Current state with shape spatial_shape.
            n_steps: Number of forecast steps.
            return_full_field: Return the full field instead of modal coefficients.
            use_history: For LSTM, use history when available.
            history: State history with shape (spatial_shape, history_len) for LSTM initialization.
            
        Returns:
            Forecast with shape (spatial_shape, n_steps) when return_full_field=True,
            otherwise modal coefficients with shape (n_modes, n_steps).
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin is not trained. Call train() first.")
        
        current_state = current_state.to(device=self.device, dtype=self.dtype)
        
        # Normalization
        if self.mean is not None:
            state_norm = (current_state - self.mean.squeeze(-1)) / self.std.squeeze(-1)
        else:
            state_norm = current_state
            
        # Process history when provided.
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
        # Step 1: Project into modal space
        # ========================================================================
        modal_current = self._project_to_modal_space(state_norm)
        
        # ========================================================================
        # Step 2: Forecast modal coefficients with the forecaster
        # ========================================================================
        modal_forecast = self._forecast_modal_coefficients(
            modal_current, 
            n_steps, 
            use_history, 
            external_history=modal_history_tensor
        )
        
        # Save into state.
        self.state.modal_coefficients = modal_forecast
        
        if not return_full_field:
            return modal_forecast
        
        # ========================================================================
        # Step 3: Reconstruct the full field
        # ========================================================================
        forecast = self._reconstruct_from_modal(modal_forecast)
        
        # Reshape if needed.
        if forecast.ndim > current_state.ndim:
            pass  # forecast already has shape (spatial_shape, n_steps).
        else:
            forecast = forecast.unsqueeze(-1).repeat(1, 1, n_steps) if current_state.ndim == 2 else forecast.unsqueeze(-1)
        
        # Denormalization
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
        """Forecast modal coefficients using the trained forecaster.
        
        Args:
            modal_current: Current modal coefficients with shape (n_modes,).
            n_steps: Number of forecast steps.
            use_history: Whether to use history for LSTM.
            external_history: External history with shape (T, n_modes) for initialization.
            
        Returns:
            Modal coefficient forecast with shape (n_modes, n_steps).
        """
        # Automatically use history for LSTM unless explicitly configured.
        if use_history is None:
            use_history = self.forecaster_type == ForecasterType.LSTM

        # Persistence fallback or untrained forecaster.
        if self.forecaster is None or self.forecaster_type == ForecasterType.PERSISTENCE:
            return modal_current.unsqueeze(1).repeat(1, n_steps)
        
        # Use the forecaster.
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
                # LSTM forecaster requires an input sequence.
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
                    # Create a window from the current state.
                    x_window = np.tile(x_current, (seq_length, 1))
                
                future_seq = self.forecaster.predict_sequence(x_window, n_steps=n_steps)
                # future_seq shape: (n_steps, n_modes)
            else:
                # Fallback to persistence
                return modal_current.unsqueeze(1).repeat(1, n_steps)
            
            # Convert to torch and transpose to (n_modes, n_steps).
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
        """Normalize the sensor measurement format.
        
        Supported formats:
        - Tensor with shape (spatial_shape[, ...]) and nonzero values at sensor positions.
        - Tensor with shape (n_sensors[, ...]) where order matches self.sensor_indices.
        
        Returns a tensor with shape spatial_shape or spatial_shape + trailing_dims.
        """
        if self.sensor_mask is None or self.sensor_indices is None:
            raise ValueError("Sensors have not been placed. Run train().")
        
        readings = sensor_readings.to(device=self.device, dtype=self.dtype)
        spatial_shape = self.sensor_mask.shape
        flat_mask = self.sensor_mask.reshape(-1)
        n_sensors = int(flat_mask.sum().item())
        
        # A full mask was already passed, possibly with a time dimension.
        if readings.shape[:len(spatial_shape)] == spatial_shape:
            return readings
        
        # Format (n_sensors, ...): scatter values into the full mask.
        if readings.shape[0] == n_sensors:
            trailing = readings.shape[1:]
            full_flat = torch.zeros((flat_mask.numel(),) + trailing, device=self.device, dtype=self.dtype)
            full_flat[flat_mask.bool()] = readings
            return full_flat.reshape(spatial_shape + trailing)
        
        raise ValueError(
            f"sensor_readings shape {readings.shape} is incompatible: expected {spatial_shape} "
            f"or ({n_sensors}, ...)"
        )
    
    def update_from_sensors(
        self,
        sensor_readings: torch.Tensor,
        timestamp: Optional[float] = None
    ) -> torch.Tensor:
        """Update state from sensor measurements.
        
        TBMD CS workflow:
        1. Build Y as a full-size tensor with measurements at sensor positions.
        2. Create a solver with A_tensor, P, and Y.
        3. Solve x_hat = solver.solve().
        4. Reconstruct the field: X_reconstructed = A_tensor @ x_hat.
        
        Args:
            sensor_readings: Sensor measurements. Supports either a full tensor
                with shape spatial_shape and nonzero values only at sensors, or
                an array with length n_sensors (or n_sensors x ... for batch/time).
            timestamp: Timestamp.
            
        Returns:
            Reconstructed full field with shape spatial_shape.
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin is not trained")
        
        if self.mean is None:
            logger.warning(
                "Internal normalization is not configured in DigitalTwin. "
                "Sensor measurements are expected to be normalized beforehand."
            )
        
        # Convert measurements to full spatial_shape with nonzero values at sensors.
        Y = self._prepare_sensor_measurements(sensor_readings)
        
        # P is the binary sensor mask.
        P = self.sensor_mask
        
        # A_tensor is the modal basis stored in self.spatial_modes.
        A_tensor = self.spatial_modes
        
        spatial_shape = self._spatial_shape if self._spatial_shape is not None else Y.shape
        spatial_ndim = len(spatial_shape)
        
        # Batch/time support: flatten trailing dimensions into a sequence of slices.
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
            # Create a Compressive Sensing solver.
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
                Y_slice,   # Full-size measurements with zeros away from sensors
                cs_config,
                ext_config
            )
            
            # Solve and reconstruct.
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
            
            # Sensor error
            sensor_err = torch.norm((X_reconstructed - Y_slice)[self.sensor_mask]).item()
            sensor_errors.append(sensor_err)
            
            reconstructed_slices.append(X_reconstructed)
        
        if len(reconstructed_slices) == 1:
            reconstructed = reconstructed_slices[0]
        else:
            reconstructed = torch.stack(reconstructed_slices, dim=-1).reshape(spatial_shape + trailing_shape)
        
        # Denormalization
        if self.mean is not None:
            if reconstructed.ndim == len(spatial_shape):
                reconstructed = reconstructed * self.std.squeeze(-1) + self.mean.squeeze(-1)
            else:
                reconstructed = reconstructed * self.std + self.mean
        
        # Update state.
        if timestamp is not None:
            self.state.current_time = timestamp
        
        # Update metrics and coefficients from the last slice.
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
        
        # Update proxy model if available.
        if self.proxy_model is not None:
            observed_state = self.create_reservoir_state(
                pressure=last_reconstructed,
                time=self.state.current_time
            )
            try:
                self.proxy_model.update_from_observations(observed_state, self.sensor_mask)
            except Exception as e:
                logger.warning(f"Proxy update failed: {e}")
        
        # Determine status
        status = 'normal'
        if self.monitor:
             try:
                 check = self.monitor.check(reconstructed)
                 if isinstance(check, dict):
                     status = check.get('status', 'normal')
                 else:
                     status = str(check)
             except Exception:
                 pass

        return {
            "reconstructed_field": reconstructed,
            "alert_status": status,
            "sensor_errors": sensor_errors
        }
    
    def evaluate_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        n_steps: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple development scenarios.
        
        Args:
            scenarios: Scenario list with parameters. Each scenario should contain:
                - 'name': scenario name
                - 'initial_state': optional initial state
            n_steps: Forecast length.
            
        Returns:
            Dictionary mapping scenario names to metrics.
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin is not trained")
        
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', f"scenario_{len(results)}")
            
            # Get the initial state for the scenario.
            if 'initial_state' in scenario:
                initial_state = scenario['initial_state']
            elif self.state.modal_coefficients is not None:
                # Reconstruct the current state from modal coefficients.
                initial_state = self._reconstruct_from_modal(
                    self.state.modal_coefficients[:, 0]
                )
            else:
                # No data available for forecasting.
                logger.warning(f"No initial state for scenario {scenario_name}, skipping")
                continue
            
            try:
                # Run forecast.
                forecast = self.predict(
                    current_state=initial_state,
                    n_steps=n_steps,
                    return_full_field=True
                )
                
                # Compute metrics.
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
        """Detect anomalies in sensor data.
        
        Uses the TBMD CS workflow for each time step.
        
        Args:
            sensor_data: Sensor data with shape (spatial_shape, n_timesteps)
                and nonzero values only at sensor positions.
            threshold: Detection threshold in standard deviations.
            
        Returns:
            List of detected anomalies with timestamp, residual, and severity.
        """
        if not self.state.is_calibrated:
            raise ValueError("Digital Twin is not trained")
        
        anomalies = []
        # Convert data to full shape (spatial_shape[, time]).
        sensor_tensor = self._prepare_sensor_measurements(
            sensor_data.to(device=self.device, dtype=self.dtype)
        )
        spatial_ndim = len(self._spatial_shape) if self._spatial_shape is not None else sensor_tensor.ndim
        if sensor_tensor.ndim == spatial_ndim:
            sensor_tensor = sensor_tensor.unsqueeze(-1)
        else:
            sensor_tensor = sensor_tensor.reshape(self._spatial_shape + (-1,))
        
        # Prepare CS configs once for efficiency.
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
            collect_history=False  # Do not collect history for speed.
        )
        
        A_tensor = self.spatial_modes
        P = self.sensor_mask
        
        # Reconstruct each time step.
        n_timesteps = sensor_tensor.shape[-1]
        
        for t in range(n_timesteps):
            try:
                # Extract measurements for the current step.
                Y = sensor_tensor[..., t]
                
                # Create solver.
                solver = TensorCompressiveSensing(
                    A_tensor,
                    P,
                    Y,
                    cs_config,
                    ext_config
                )
                
                # Solve.
                x_hat, metrics = solver.solve()
                
                # Compute reconstruction error.
                X_reconstructed = reconstruct_tensor(
                    A_tensor=A_tensor,
                    x_hat=x_hat,
                    zero_threshold=1e-6,
                    decimals=4
                )
                
                if X_reconstructed is not None:
                    # Compute residual.
                    residual = torch.norm((X_reconstructed - Y)[self.sensor_mask]).item()
                    
                    # Determine anomaly threshold.
                    if self.std is not None:
                        threshold_value = threshold * self.std.mean().item()
                    else:
                        threshold_value = threshold
                    
                    # Check for anomaly.
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
                logger.warning(f"Reconstruction error at step {t}: {e}")
                anomalies.append({
                    'timestamp': t,
                    'error': str(e),
                    'severity': 'error'
                })
        
        return anomalies
    
    def get_sensor_locations(self) -> np.ndarray:
        """Return placed sensor indices."""
        if self.sensor_indices is None:
            raise ValueError("Sensors have not been placed")
        return self.sensor_indices.cpu().numpy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return digital twin runtime statistics."""
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
    # PROXY MODEL METHODS
    # ==========================================================================
    
    def _init_proxy_model(self):
        """Initialize the proxy model for scenario analysis with well controls.
        
        Creates a LinearDynamicsProxyModel, NeuralProxyModel, or
        PhysicsInformedProxyModel from the modal basis.
        """
        if self.spatial_modes is None:
            raise ValueError("Modal basis has not been computed. Run decomposition first.")
        
        # Flatten modal basis if needed.
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
                logger.info("LinearDynamicsProxyModel initialized")
                
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
                logger.info(f"NeuralProxyModel initialized (hidden={hidden_layers})")
                
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
                logger.info("PhysicsInformedProxyModel initialized")
    
    def _build_proxy_training_sets(
        self,
        data_dict: Dict[str, torch.Tensor]
    ) -> Tuple[List[ReservoirState], List[List[WellControl]]]:
        """Prepare historical states and simple well controls for proxy calibration.

        Uses the first subject in data_dict.
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
        """Calibrate the proxy model on historical data.
        
        Trains the proxy model to predict reservoir dynamics with well controls.
        
        Args:
            historical_states: Historical reservoir states.
            historical_controls: Well controls for each time step.
            **kwargs: Additional calibration parameters:
                - regularization: float for LinearDynamicsProxyModel
                - epochs, learning_rate, batch_size: for NeuralProxyModel
            
        Returns:
            Calibration metrics such as mse and relative_error.
        """
        if self.proxy_model is None:
            raise ValueError("Proxy model is not initialized. Set proxy_model_type in config.")
        
        if self.config.verbose:
            logger.info(f"Calibrating {self.proxy_model_type.value} proxy model...")
        
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
            raise ValueError(f"Unknown proxy model type: {type(self.proxy_model)}")
        
        if self.config.verbose:
            logger.info(f"Proxy model calibrated: {metrics}")
        
        return metrics
    
    def predict_with_controls(
        self,
        current_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float = 1.0,
        time_steps: int = 10
    ) -> List[ReservoirState]:
        """Forecast reservoir state with well controls.
        
        Uses the proxy model for fast scenario analysis.
        
        Args:
            current_state: Current reservoir state.
            well_controls: Well controls.
            time_horizon: Forecast horizon.
            time_steps: Number of time steps.
            
        Returns:
            List of forecasted states.
        """
        if self.proxy_model is None:
            raise ValueError("Proxy model is not initialized or calibrated.")
        
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
        """Evaluate multiple well-control scenarios.
        
        Provides fast what-if analysis through the proxy model.
        
        Args:
            initial_state: Initial reservoir state.
            scenarios: Dictionary mapping scenario names to well controls.
            time_horizon: Forecast horizon.
            time_steps: Number of steps.
            
        Returns:
            Dictionary mapping scenario names to forecasted states and KPIs.
        """
        if self.proxy_model is None:
            raise ValueError("Proxy model is not initialized.")
        
        results = {}
        
        for scenario_name, well_controls in scenarios.items():
            if self.config.verbose:
                logger.info(f"Evaluating scenario: {scenario_name}")
            
            try:
                # Forecast
                forecasted_states = self.predict_with_controls(
                    current_state=initial_state,
                    well_controls=well_controls,
                    time_horizon=time_horizon,
                    time_steps=time_steps
                )
                
                # Compute KPIs.
                kpis = self._compute_scenario_kpis(forecasted_states, well_controls)
                
                results[scenario_name] = {
                    'forecasted_states': forecasted_states,
                    'kpis': kpis,
                    'well_controls': well_controls
                }
                
            except Exception as e:
                logger.error(f"Error in scenario {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results
    
    def _compute_scenario_kpis(
        self,
        forecasted_states: List[ReservoirState],
        well_controls: List[WellControl]
    ) -> Dict[str, float]:
        """Compute key performance indicators for a scenario."""
        kpis = {}
        
        # Pressure statistics
        pressures = [state.pressure for state in forecasted_states]
        avg_pressures = [torch.mean(p).item() for p in pressures]
        
        kpis['avg_pressure'] = float(np.mean(avg_pressures))
        kpis['min_pressure'] = float(np.min(avg_pressures))
        kpis['max_pressure'] = float(np.max(avg_pressures))
        kpis['pressure_std'] = float(np.std(avg_pressures))
        
        # Production and injection
        production_wells = [ctrl for ctrl in well_controls if ctrl.value < 0]
        injection_wells = [ctrl for ctrl in well_controls if ctrl.value > 0]
        
        kpis['total_production'] = abs(sum(ctrl.value for ctrl in production_wells)) * len(forecasted_states)
        kpis['total_injection'] = sum(ctrl.value for ctrl in injection_wells) * len(forecasted_states)
        kpis['net_production'] = kpis['total_production'] - kpis['total_injection']
        
        # Number of active wells
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
        """Create a ReservoirState from input data.
        
        Convenience helper for creating states.
        
        Args:
            pressure: Pressure field.
            saturation: Optional saturation field.
            time: Time.
            well_rates: Well rates.
            
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
        """Create a WellControl instance.
        
        Convenience helper for creating well controls.
        
        Args:
            well_name: Well name.
            control_type: Control type ('rate', 'pressure', 'bhp').
            value: Value where positive means injection and negative means production.
            location: Well coordinates.
            
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
        """Update the proxy model from new observations (data assimilation).
        
        Performs online model updates.
        
        Args:
            observed_state: Observed state.
            sensor_locations: Sensor positions.
        """
        if self.proxy_model is not None:
            self.proxy_model.update_from_observations(observed_state, sensor_locations)
            if self.config.verbose:
                logger.info("Proxy model updated from observations")


# Backward compatibility alias
DigitalTwinTBMD = DigitalTwin
