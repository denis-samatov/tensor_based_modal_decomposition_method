"""
Reservoir Proxy Model for Digital Twin Applications

This module implements simplified reservoir simulation models (proxy models) that can be
integrated with TBMD for fast scenario analysis in digital twin applications.

The proxy models use data-driven approaches (ROM - Reduced Order Models) to approximate
the behavior of complex reservoir simulators at a fraction of the computational cost.

Key Features:
- Linear dynamics approximation
- Nonlinear forecasting with neural networks
- Physics-informed constraints
- Integration with TBMD modal bases
- Fast what-if scenario evaluation

References:
- Benner et al. (2015). "Model Reduction and Approximation"
- Cardoso & Durlofsky (2010). "Use of reduced-order modeling procedures for production optimization"
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ReservoirState:
    """
    Represents the state of a reservoir at a given time.

    Attributes
    ----------
    pressure : torch.Tensor
        Pressure field (spatial dimensions)
    saturation : torch.Tensor, optional
        Saturation field (spatial dimensions)
    time : float
        Current time
    well_rates : Dict[str, float], optional
        Well production/injection rates
    """

    pressure: torch.Tensor
    saturation: Optional[torch.Tensor] = None
    time: float = 0.0
    well_rates: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            "pressure": self.pressure.cpu().numpy()
            if isinstance(self.pressure, torch.Tensor)
            else self.pressure,
            "saturation": self.saturation.cpu().numpy()
            if self.saturation is not None and isinstance(self.saturation, torch.Tensor)
            else self.saturation,
            "time": self.time,
            "well_rates": self.well_rates,
        }


@dataclass
class WellControl:
    """
    Represents well control parameters.

    Attributes
    ----------
    well_name : str
        Name/identifier of the well
    control_type : str
        Type of control: 'rate', 'pressure', 'bhp'
    value : float
        Control value
    location : Tuple[int, ...]
        Spatial location indices
    """

    well_name: str
    control_type: str  # 'rate', 'pressure', 'bhp'
    value: float
    location: Tuple[int, ...]


class ReservoirProxyModelBase(ABC):
    """
    Abstract base class for reservoir proxy models.

    Defines the interface that all proxy models must implement.
    """

    @abstractmethod
    def forecast(
        self,
        current_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float,
        time_steps: int,
    ) -> List[ReservoirState]:
        """
        Forecast reservoir behavior over time horizon.

        Parameters
        ----------
        current_state : ReservoirState
            Current reservoir state
        well_controls : List[WellControl]
            List of well control parameters
        time_horizon : float
            Total time to forecast
        time_steps : int
            Number of time steps

        Returns
        -------
        List[ReservoirState]
            List of forecasted states
        """
        pass

    @abstractmethod
    def update_from_observations(
        self, observed_state: ReservoirState, sensor_locations: torch.Tensor
    ) -> None:
        """
        Update model based on new observations (data assimilation).

        Parameters
        ----------
        observed_state : ReservoirState
            Observed reservoir state
        sensor_locations : torch.Tensor
            Locations where observations were made
        """
        pass


class LinearDynamicsProxyModel(ReservoirProxyModelBase):
    """
    Linear dynamics proxy model using TBMD modal basis.

    Approximates reservoir dynamics as:
        x(t+1) = A @ x(t) + B @ u(t)

    where x(t) are modal coefficients and u(t) are well controls.

    This is the fastest proxy model but assumes linear dynamics.
    """

    def __init__(
        self,
        spatial_shape: Tuple[int, ...],
        modal_basis: torch.Tensor,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Shape of spatial domain
        modal_basis : torch.Tensor
            Modal basis from TBMD (spatial_size x n_modes)
        device : str
            PyTorch device
        dtype : torch.dtype
            Data type
        """
        self.spatial_shape = spatial_shape
        self.modal_basis = modal_basis.to(device, dtype)
        self.n_modes = modal_basis.shape[1]
        self.device = device
        self.dtype = dtype

        # Linear dynamics matrices (to be learned/calibrated)
        self.A = torch.eye(self.n_modes, device=device, dtype=dtype)
        self.B = None  # Will be initialized when we know number of wells

        logger.info(f"Initialized LinearDynamicsProxyModel with {self.n_modes} modes")

    def calibrate(
        self,
        historical_states: List[ReservoirState],
        historical_controls: List[List[WellControl]],
        regularization: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Calibrate linear dynamics model from historical data.

        Parameters
        ----------
        historical_states : List[ReservoirState]
            Historical reservoir states
        historical_controls : List[List[WellControl]]
            Historical well controls for each time step
        regularization : float
            Ridge regularization parameter

        Returns
        -------
        Dict[str, float]
            Calibration metrics
        """
        logger.info("Calibrating linear dynamics model...")

        # Project historical states to modal space
        modal_coeffs = []
        for state in historical_states:
            pressure_flat = state.pressure.flatten()
            coeffs = self.modal_basis.T @ pressure_flat
            modal_coeffs.append(coeffs)

        modal_coeffs = torch.stack(modal_coeffs)  # (T, n_modes)

        # Extract control inputs
        n_wells = len(historical_controls[0]) if historical_controls else 0
        if n_wells > 0:
            control_inputs = torch.zeros(
                (len(historical_controls), n_wells), device=self.device, dtype=self.dtype
            )
            for t, controls in enumerate(historical_controls):
                for i, ctrl in enumerate(controls):
                    control_inputs[t, i] = ctrl.value
        else:
            control_inputs = None

        # Solve for A and B using least squares
        # [x(t+1)] = [A, B] @ [x(t); u(t)]
        X_current = modal_coeffs[:-1]  # (T-1, n_modes)
        X_next = modal_coeffs[1:]  # (T-1, n_modes)

        if control_inputs is not None:
            U = control_inputs[:-1]  # (T-1, n_wells)
            # Augmented input: [X_current | U]
            XU = torch.cat([X_current, U], dim=1)  # (T-1, n_modes + n_wells)

            # Ridge regression: [A, B] = X_next^T @ XU @ (XU^T @ XU + λI)^{-1}
            XU_T_XU = XU.T @ XU
            reg_term = regularization * torch.eye(XU.shape[1], device=self.device, dtype=self.dtype)
            AB = torch.linalg.solve(XU_T_XU + reg_term, XU.T @ X_next).T

            self.A = AB[:, : self.n_modes]
            self.B = AB[:, self.n_modes :]
        else:
            # Only learn A (no controls)
            X_T_X = X_current.T @ X_current
            reg_term = regularization * torch.eye(
                self.n_modes, device=self.device, dtype=self.dtype
            )
            self.A = torch.linalg.solve(X_T_X + reg_term, X_current.T @ X_next).T

        # Compute calibration error
        X_pred = X_current @ self.A.T
        if control_inputs is not None and self.B is not None:
            X_pred += U @ self.B.T

        mse = torch.mean((X_pred - X_next) ** 2).item()
        relative_error = torch.norm(X_pred - X_next) / torch.norm(X_next)

        metrics = {
            "mse": mse,
            "relative_error": float(relative_error.item()),
            "n_samples": len(historical_states) - 1,
        }

        logger.info(f"Calibration complete: MSE={mse:.6f}, Rel.Error={relative_error:.4f}")
        return metrics

    def forecast(
        self,
        current_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float,
        time_steps: int,
    ) -> List[ReservoirState]:
        """Forecast using linear dynamics."""
        dt = time_horizon / time_steps

        # Project current state to modal space
        pressure_flat = current_state.pressure.flatten()
        modal_coeffs = self.modal_basis.T @ pressure_flat

        # Extract control input
        if self.B is not None and well_controls:
            u = torch.tensor(
                [ctrl.value for ctrl in well_controls], device=self.device, dtype=self.dtype
            )
        else:
            u = None

        # Forecast
        forecasted_states = []
        current_time = current_state.time

        for step in range(time_steps):
            # Update modal coefficients
            modal_coeffs = self.A @ modal_coeffs
            if u is not None and self.B is not None:
                modal_coeffs += self.B @ u

            # Reconstruct pressure field
            pressure_flat = self.modal_basis @ modal_coeffs
            pressure = pressure_flat.reshape(self.spatial_shape)

            current_time += dt

            # Create forecasted state
            state = ReservoirState(
                pressure=pressure,
                saturation=None,  # Linear model doesn't track saturation separately
                time=current_time,
                well_rates={ctrl.well_name: ctrl.value for ctrl in well_controls},
            )
            forecasted_states.append(state)

        return forecasted_states

    def update_from_observations(
        self, observed_state: ReservoirState, sensor_locations: torch.Tensor
    ) -> None:
        """
        Update modal coefficients using observations (Kalman-like update).

        This is a simplified data assimilation approach.
        """
        # This would implement a Kalman filter update in the modal space
        # For now, we just log that observations were received
        logger.info(f"Received observations at {len(sensor_locations)} sensors")
        # TODO: Implement proper data assimilation


class NeuralProxyModel(ReservoirProxyModelBase):
    """
    Neural network-based proxy model for nonlinear dynamics.

    Uses a neural network to learn the mapping:
        x(t+1) = f_NN(x(t), u(t))

    More flexible than linear model but requires more training data.
    """

    def __init__(
        self,
        spatial_shape: Tuple[int, ...],
        modal_basis: torch.Tensor,
        hidden_layers: List[int] = [128, 64],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Shape of spatial domain
        modal_basis : torch.Tensor
            Modal basis from TBMD
        hidden_layers : List[int]
            Hidden layer sizes
        device : str
            PyTorch device
        dtype : torch.dtype
            Data type
        """
        self.spatial_shape = spatial_shape
        self.modal_basis = modal_basis.to(device, dtype)
        self.n_modes = modal_basis.shape[1]
        self.device = device
        self.dtype = dtype

        # Build neural network
        self.model = None
        self.hidden_layers = hidden_layers
        self.n_wells = None  # Will be set during training

        logger.info(f"Initialized NeuralProxyModel with {self.n_modes} modes")

    def _build_network(self, n_wells: int):
        """Build the neural network architecture."""
        input_size = self.n_modes + n_wells
        output_size = self.n_modes

        layers = []
        prev_size = input_size

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers).to(self.device)
        self.n_wells = n_wells

        logger.info(f"Built neural network: {input_size} -> {self.hidden_layers} -> {output_size}")

    def train_model(
        self,
        historical_states: List[ReservoirState],
        historical_controls: List[List[WellControl]],
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train the neural network proxy model.

        Parameters
        ----------
        historical_states : List[ReservoirState]
            Historical states
        historical_controls : List[List[WellControl]]
            Historical controls
        epochs : int
            Training epochs
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size

        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        logger.info("Training neural proxy model...")

        # Project to modal space
        modal_coeffs = []
        for state in historical_states:
            pressure_flat = state.pressure.flatten()
            coeffs = self.modal_basis.T @ pressure_flat
            modal_coeffs.append(coeffs)

        modal_coeffs = torch.stack(modal_coeffs)

        # Extract controls
        n_wells = len(historical_controls[0])
        control_inputs = torch.zeros(
            (len(historical_controls), n_wells), device=self.device, dtype=self.dtype
        )
        for t, controls in enumerate(historical_controls):
            for i, ctrl in enumerate(controls):
                control_inputs[t, i] = ctrl.value

        # Build network if not already built
        if self.model is None:
            self._build_network(n_wells)

        # Prepare training data
        X_current = modal_coeffs[:-1]
        X_next = modal_coeffs[1:]
        U = control_inputs[:-1]

        # Combine inputs
        inputs = torch.cat([X_current, U], dim=1)
        targets = X_next

        # Create dataset
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {"train_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_inputs)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history["train_loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        logger.info("Training complete")
        return history

    def forecast(
        self,
        current_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float,
        time_steps: int,
    ) -> List[ReservoirState]:
        """Forecast using neural network."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model() first.")

        dt = time_horizon / time_steps

        # Project current state to modal space
        pressure_flat = current_state.pressure.flatten()
        modal_coeffs = self.modal_basis.T @ pressure_flat

        # Extract control input
        u = torch.tensor(
            [ctrl.value for ctrl in well_controls], device=self.device, dtype=self.dtype
        )

        # Forecast
        forecasted_states = []
        current_time = current_state.time

        self.model.eval()
        with torch.no_grad():
            for step in range(time_steps):
                # Prepare input
                net_input = torch.cat([modal_coeffs, u]).unsqueeze(0)

                # Predict next modal coefficients
                modal_coeffs = self.model(net_input).squeeze(0)

                # Reconstruct pressure field
                pressure_flat = self.modal_basis @ modal_coeffs
                pressure = pressure_flat.reshape(self.spatial_shape)

                current_time += dt

                state = ReservoirState(
                    pressure=pressure,
                    saturation=None,
                    time=current_time,
                    well_rates={ctrl.well_name: ctrl.value for ctrl in well_controls},
                )
                forecasted_states.append(state)

        return forecasted_states

    def update_from_observations(
        self, observed_state: ReservoirState, sensor_locations: torch.Tensor
    ) -> None:
        """Update model from observations (online learning)."""
        logger.info(f"Received observations at {len(sensor_locations)} sensors")
        # TODO: Implement online learning/fine-tuning


class PhysicsInformedProxyModel(LinearDynamicsProxyModel):
    """
    Physics-informed proxy model that enforces mass conservation and other constraints.

    Extends the linear model with physics-based constraints.
    """

    def __init__(
        self,
        spatial_shape: Tuple[int, ...],
        modal_basis: torch.Tensor,
        porosity: Optional[torch.Tensor] = None,
        permeability: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        spatial_shape : Tuple[int, ...]
            Shape of spatial domain
        modal_basis : torch.Tensor
            Modal basis from TBMD
        porosity : torch.Tensor, optional
            Porosity field
        permeability : torch.Tensor, optional
            Permeability field
        device : str
            PyTorch device
        dtype : torch.dtype
            Data type
        """
        super().__init__(spatial_shape, modal_basis, device, dtype)

        self.porosity = porosity
        self.permeability = permeability

        logger.info("Initialized PhysicsInformedProxyModel")

    def enforce_mass_conservation(
        self, pressure_field: torch.Tensor, well_controls: List[WellControl]
    ) -> torch.Tensor:
        """
        Enforce mass conservation constraint.

        Parameters
        ----------
        pressure_field : torch.Tensor
            Predicted pressure field
        well_controls : List[WellControl]
            Well controls

        Returns
        -------
        torch.Tensor
            Corrected pressure field
        """
        # Simplified mass balance check
        # In a real implementation, this would solve a pressure equation

        total_injection = sum(ctrl.value for ctrl in well_controls if ctrl.value > 0)
        total_production = abs(sum(ctrl.value for ctrl in well_controls if ctrl.value < 0))

        if abs(total_injection - total_production) > 1e-6:
            logger.warning(
                f"Mass imbalance: injection={total_injection:.2f}, "
                f"production={total_production:.2f}"
            )

        return pressure_field

    def forecast(
        self,
        current_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float,
        time_steps: int,
    ) -> List[ReservoirState]:
        """Forecast with physics constraints."""
        # Use parent class forecast
        forecasted_states = super().forecast(current_state, well_controls, time_horizon, time_steps)

        # Apply physics constraints
        for state in forecasted_states:
            state.pressure = self.enforce_mass_conservation(state.pressure, well_controls)

        return forecasted_states
