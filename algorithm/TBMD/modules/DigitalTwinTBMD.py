"""
Digital Twin for Reservoir Monitoring and Forecasting using TBMD

This module implements a comprehensive digital twin system that combines:
1. TBMD for efficient data representation (Reduced Order Model)
2. Optimal sensor placement via tensor QR
3. Real-time field reconstruction from sparse sensor data
4. Fast scenario analysis with proxy models
5. Online model updating from observations

The digital twin enables:
- Real-time comparison of predicted vs actual reservoir behavior
- Fast what-if scenario evaluation without running full simulators
- Optimal sensor network design
- Data assimilation and model updating

Key Components:
--------------
- DigitalTwinTBMD: Main digital twin orchestrator
- RealtimeMonitor: Real-time monitoring and alerting
- ScenarioAnalyzer: What-if scenario evaluation
- ModelCalibrator: Online model calibration

References:
-----------
- Glaessgen & Stargel (2012). "The Digital Twin Paradigm for Future NASA and U.S. Air Force Vehicles"
- Brunton et al. (2016). "Compressive sensing and low-rank libraries for classification of bifurcation regimes"
- Cardoso & Durlofsky (2010). "Use of reduced-order modeling for production optimization"
"""

import numpy as np
import torch
import tensorly as tl
from typing import Optional, Tuple, List, Dict, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

# TBMD components
from TBMD.modules.TensorHOSVD import TuckerDecomposer
from TBMD.modules.GeometryAwareTensorHOSVD import GeometryAwareTuckerDecomposer, GeometryAwareConfig
from TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import TensorTubeQRDecomposition
from TBMD.modules.TensorBasedCompressiveSensing import TensorCSReconstructor, TensorCSConfig
from TBMD.modules.GeometryAwareTensorCS import GeometryAwareTensorCS
from TBMD.models.ReservoirProxyModel import (
    ReservoirProxyModelBase, 
    LinearDynamicsProxyModel,
    NeuralProxyModel,
    PhysicsInformedProxyModel,
    ReservoirState,
    WellControl
)
from TBMD.geometry import MeshGeometry, MeshGraphBuilder
from TBMD.utils.tbmd_utils import to_torch_tensor, get_torch_device

logger = logging.getLogger(__name__)


@dataclass
class DigitalTwinConfig:
    """
    Configuration for Digital Twin system.
    
    Attributes
    ----------
    n_spatial_modes : int
        Number of spatial modes to retain in TBMD
    n_temporal_modes : int
        Number of temporal modes to retain
    n_sensors : int
        Number of sensors to deploy
    proxy_model_type : str
        Type of proxy model: 'linear', 'neural', 'physics_informed'
    use_geometry_aware : bool
        Whether to use geometry-aware TBMD
    reconstruction_method : str
        Reconstruction method: 'least_squares', 'admm', 'ista'
    update_frequency : int
        How often to update model from observations (in time steps)
    alert_threshold : float
        Threshold for anomaly detection (relative error)
    """
    n_spatial_modes: int = 50
    n_temporal_modes: int = 20
    n_sensors: int = 30
    proxy_model_type: str = 'linear'
    use_geometry_aware: bool = False
    reconstruction_method: str = 'admm'
    update_frequency: int = 10
    alert_threshold: float = 0.15
    device: str = 'cpu'
    dtype: str = 'float32'
    
    # Geometry-aware settings
    geo_config: Optional[GeometryAwareConfig] = None
    
    def __post_init__(self):
        if self.proxy_model_type not in ['linear', 'neural', 'physics_informed']:
            raise ValueError(f"Unknown proxy_model_type: {self.proxy_model_type}")
        
        if self.reconstruction_method not in ['least_squares', 'admm', 'ista']:
            raise ValueError(f"Unknown reconstruction_method: {self.reconstruction_method}")


@dataclass
class DigitalTwinState:
    """
    Current state of the digital twin.
    
    Tracks the current operational state and history.
    """
    current_time: float = 0.0
    reservoir_state: Optional[ReservoirState] = None
    sensor_readings: Optional[torch.Tensor] = None
    prediction_error: float = 0.0
    is_calibrated: bool = False
    last_update_time: float = 0.0
    alert_status: str = 'normal'  # 'normal', 'warning', 'critical'
    history: Dict[str, List] = field(default_factory=lambda: {
        'times': [],
        'errors': [],
        'alerts': []
    })


class RealtimeMonitor:
    """
    Real-time monitoring component for digital twin.
    
    Monitors predictions vs observations and triggers alerts.
    """
    
    def __init__(self, alert_threshold: float = 0.15):
        """
        Parameters
        ----------
        alert_threshold : float
            Threshold for triggering alerts (relative error)
        """
        self.alert_threshold = alert_threshold
        self.monitoring_history = []
        
    def compare_prediction_observation(self,
                                      predicted: torch.Tensor,
                                      observed: torch.Tensor,
                                      sensor_locations: torch.Tensor) -> Dict[str, float]:
        """
        Compare predicted and observed fields at sensor locations.
        
        Parameters
        ----------
        predicted : torch.Tensor
            Predicted field
        observed : torch.Tensor
            Observed field
        sensor_locations : torch.Tensor
            Binary mask of sensor locations
            
        Returns
        -------
        Dict[str, float]
            Comparison metrics
        """
        # Extract values at sensor locations
        sensor_mask = sensor_locations.bool()
        pred_at_sensors = predicted[sensor_mask]
        obs_at_sensors = observed[sensor_mask]
        
        # Compute metrics
        absolute_error = torch.abs(pred_at_sensors - obs_at_sensors)
        relative_error = torch.norm(absolute_error) / torch.norm(obs_at_sensors)
        max_error = torch.max(absolute_error)
        mean_error = torch.mean(absolute_error)
        
        metrics = {
            'relative_error': float(relative_error.item()),
            'max_error': float(max_error.item()),
            'mean_error': float(mean_error.item()),
            'rmse': float(torch.sqrt(torch.mean(absolute_error ** 2)).item())
        }
        
        return metrics
    
    def check_alert_status(self, metrics: Dict[str, float]) -> str:
        """
        Determine alert status based on metrics.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Comparison metrics
            
        Returns
        -------
        str
            Alert status: 'normal', 'warning', 'critical'
        """
        rel_error = metrics['relative_error']
        
        if rel_error > 2 * self.alert_threshold:
            return 'critical'
        elif rel_error > self.alert_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def log_monitoring_event(self, 
                            time: float, 
                            metrics: Dict[str, float],
                            alert_status: str) -> None:
        """Log monitoring event."""
        event = {
            'time': time,
            'metrics': metrics,
            'alert_status': alert_status,
            'timestamp': datetime.now().isoformat()
        }
        self.monitoring_history.append(event)
        
        if alert_status != 'normal':
            logger.warning(f"[{alert_status.upper()}] at t={time:.2f}: "
                         f"rel_error={metrics['relative_error']:.4f}")


class ScenarioAnalyzer:
    """
    Scenario analysis component for what-if studies.
    
    Evaluates different well control scenarios using the proxy model.
    """
    
    def __init__(self, proxy_model: ReservoirProxyModelBase):
        """
        Parameters
        ----------
        proxy_model : ReservoirProxyModelBase
            Proxy model for fast forecasting
        """
        self.proxy_model = proxy_model
        self.scenario_results = {}
        
    def evaluate_scenario(self,
                         scenario_name: str,
                         initial_state: ReservoirState,
                         well_controls: List[WellControl],
                         time_horizon: float,
                         time_steps: int) -> Dict[str, any]:
        """
        Evaluate a single scenario.
        
        Parameters
        ----------
        scenario_name : str
            Name/ID of scenario
        initial_state : ReservoirState
            Initial reservoir state
        well_controls : List[WellControl]
            Well control parameters for this scenario
        time_horizon : float
            Forecast horizon
        time_steps : int
            Number of time steps
            
        Returns
        -------
        Dict[str, any]
            Scenario results including forecasted states and KPIs
        """
        logger.info(f"Evaluating scenario: {scenario_name}")
        
        # Run forecast
        forecasted_states = self.proxy_model.forecast(
            initial_state, well_controls, time_horizon, time_steps
        )
        
        # Compute KPIs
        kpis = self._compute_kpis(forecasted_states, well_controls)
        
        result = {
            'scenario_name': scenario_name,
            'forecasted_states': forecasted_states,
            'kpis': kpis,
            'well_controls': well_controls
        }
        
        self.scenario_results[scenario_name] = result
        return result
    
    def compare_scenarios(self, 
                         scenario_names: List[str],
                         kpi_name: str = 'total_production') -> Dict[str, float]:
        """
        Compare multiple scenarios on a given KPI.
        
        Parameters
        ----------
        scenario_names : List[str]
            Names of scenarios to compare
        kpi_name : str
            KPI to compare
            
        Returns
        -------
        Dict[str, float]
            KPI values for each scenario
        """
        comparison = {}
        for name in scenario_names:
            if name in self.scenario_results:
                kpi_value = self.scenario_results[name]['kpis'].get(kpi_name, None)
                comparison[name] = kpi_value
        
        return comparison
    
    def _compute_kpis(self,
                     forecasted_states: List[ReservoirState],
                     well_controls: List[WellControl]) -> Dict[str, float]:
        """Compute Key Performance Indicators from forecasted states."""
        kpis = {}
        
        # Average pressure
        avg_pressures = [torch.mean(state.pressure).item() for state in forecasted_states]
        kpis['avg_pressure'] = float(np.mean(avg_pressures))
        kpis['min_pressure'] = float(np.min(avg_pressures))
        kpis['max_pressure'] = float(np.max(avg_pressures))
        
        # Total production (simplified)
        production_wells = [ctrl for ctrl in well_controls if ctrl.value < 0]
        kpis['total_production'] = abs(sum(ctrl.value for ctrl in production_wells)) * len(forecasted_states)
        
        # Total injection
        injection_wells = [ctrl for ctrl in well_controls if ctrl.value > 0]
        kpis['total_injection'] = sum(ctrl.value for ctrl in injection_wells) * len(forecasted_states)
        
        return kpis


class ModelCalibrator:
    """
    Model calibration component for online updating.
    
    Handles calibration of proxy model from historical data and online updates.
    """
    
    def __init__(self, proxy_model: ReservoirProxyModelBase):
        """
        Parameters
        ----------
        proxy_model : ReservoirProxyModelBase
            Proxy model to calibrate
        """
        self.proxy_model = proxy_model
        self.calibration_history = []
        
    def calibrate_from_historical_data(self,
                                       historical_states: List[ReservoirState],
                                       historical_controls: List[List[WellControl]]) -> Dict[str, float]:
        """
        Calibrate proxy model from historical data.
        
        Parameters
        ----------
        historical_states : List[ReservoirState]
            Historical reservoir states
        historical_controls : List[List[WellControl]]
            Historical well controls
            
        Returns
        -------
        Dict[str, float]
            Calibration metrics
        """
        logger.info("Calibrating proxy model from historical data...")
        
        if isinstance(self.proxy_model, LinearDynamicsProxyModel):
            metrics = self.proxy_model.calibrate(historical_states, historical_controls)
        elif isinstance(self.proxy_model, NeuralProxyModel):
            metrics = self.proxy_model.train_model(historical_states, historical_controls)
        else:
            metrics = {'status': 'calibration not implemented for this model type'}
        
        self.calibration_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def online_update(self,
                     observed_state: ReservoirState,
                     sensor_locations: torch.Tensor) -> None:
        """
        Perform online update from new observations.
        
        Parameters
        ----------
        observed_state : ReservoirState
            Newly observed state
        sensor_locations : torch.Tensor
            Sensor locations
        """
        logger.info("Performing online model update...")
        self.proxy_model.update_from_observations(observed_state, sensor_locations)


class DigitalTwinTBMD:
    """
    Digital Twin system for reservoir monitoring and forecasting.
    
    Integrates all components into a cohesive digital twin system:
    - TBMD for efficient representation
    - Optimal sensor placement
    - Real-time reconstruction
    - Fast scenario analysis
    - Online model updating
    
    Usage Example
    -------------
    >>> # Initialize digital twin
    >>> config = DigitalTwinConfig(
    ...     n_spatial_modes=50,
    ...     n_sensors=30,
    ...     proxy_model_type='linear'
    ... )
    >>> twin = DigitalTwinTBMD(config)
    >>> 
    >>> # Train from historical data
    >>> twin.train(historical_data, well_controls)
    >>> 
    >>> # Real-time monitoring
    >>> predicted_state = twin.predict_next_state(current_well_controls)
    >>> twin.update_from_sensors(sensor_readings, sensor_locations)
    >>> 
    >>> # Scenario analysis
    >>> scenarios = {
    ...     'baseline': baseline_controls,
    ...     'increased_injection': increased_controls
    ... }
    >>> results = twin.evaluate_scenarios(scenarios)
    """
    
    def __init__(self, config: DigitalTwinConfig):
        """
        Initialize Digital Twin system.
        
        Parameters
        ----------
        config : DigitalTwinConfig
            Configuration for digital twin
        """
        self.config = config
        self.state = DigitalTwinState()
        
        # Setup device
        self.device = get_torch_device(config.device)
        self.dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # Core components (to be initialized during training)
        self.decomposer = None
        self.qr_decomposer = None
        self.reconstructor = None
        self.proxy_model = None
        self.mesh = None
        
        # Management components
        self.monitor = RealtimeMonitor(config.alert_threshold)
        self.scenario_analyzer = None  # Initialized after proxy model
        self.calibrator = None  # Initialized after proxy model
        
        # Storage
        self.modal_basis = None
        self.sensor_locations = None
        self.spatial_shape = None
        
        logger.info(f"Initialized DigitalTwinTBMD with config: {config}")
    
    def train(self,
              historical_data: torch.Tensor,
              historical_controls: Optional[List[List[WellControl]]] = None,
              mesh: Optional[MeshGeometry] = None,
              rejection_domain: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Train the digital twin from historical data.
        
        This performs:
        1. TBMD decomposition to extract modal basis
        2. Sensor placement optimization
        3. Proxy model calibration
        
        Parameters
        ----------
        historical_data : torch.Tensor
            Historical field data (spatial_dims x time)
        historical_controls : List[List[WellControl]], optional
            Historical well controls
        mesh : MeshGeometry, optional
            Mesh geometry for geometry-aware TBMD
        rejection_domain : torch.Tensor, optional
            Domain where sensors cannot be placed
            
        Returns
        -------
        Dict[str, any]
            Training metrics and summary
        """
        logger.info("=" * 60)
        logger.info("Training Digital Twin")
        logger.info("=" * 60)
        
        # Convert to torch
        historical_data = to_torch_tensor(historical_data, device=self.device, dtype=self.dtype)
        self.spatial_shape = historical_data.shape[:-1]
        
        # Step 1: TBMD Decomposition
        logger.info("\n[1/3] Performing TBMD decomposition...")
        decomposition_result = self._perform_decomposition(historical_data, mesh)
        
        # Step 2: Sensor Placement
        logger.info("\n[2/3] Optimizing sensor placement...")
        sensor_result = self._optimize_sensor_placement(historical_data, rejection_domain)
        
        # Step 3: Calibrate Proxy Model
        logger.info("\n[3/3] Calibrating proxy model...")
        calibration_result = self._calibrate_proxy_model(
            historical_data, historical_controls
        )
        
        # Mark as calibrated
        self.state.is_calibrated = True
        
        # Compile training summary
        training_summary = {
            'decomposition': decomposition_result,
            'sensor_placement': sensor_result,
            'calibration': calibration_result,
            'status': 'trained'
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("Digital Twin Training Complete")
        logger.info("=" * 60)
        logger.info(f"Spatial modes: {self.config.n_spatial_modes}")
        logger.info(f"Sensors placed: {torch.sum(self.sensor_locations).item()}")
        logger.info(f"Proxy model: {self.config.proxy_model_type}")
        
        return training_summary
    
    def _perform_decomposition(self,
                               data: torch.Tensor,
                               mesh: Optional[MeshGeometry] = None) -> Dict[str, any]:
        """Perform TBMD decomposition."""
        if self.config.use_geometry_aware and mesh is not None:
            # Geometry-aware TBMD
            self.mesh = mesh
            geo_config = self.config.geo_config or GeometryAwareConfig()
            
            self.decomposer = GeometryAwareTuckerDecomposer(
                tensor=data,
                mesh=mesh,
                geo_config=geo_config,
                ranks=[self.config.n_spatial_modes, self.config.n_temporal_modes],
                device=str(self.device),
                dtype=self.dtype
            )
        else:
            # Standard TBMD
            self.decomposer = TuckerDecomposer(
                tensor=data,
                ranks=[self.config.n_spatial_modes, self.config.n_temporal_modes],
                device=str(self.device),
                dtype=self.dtype
            )
        
        self.decomposer.decompose()
        
        # Extract spatial modal basis
        factors = self.decomposer.factors
        spatial_factor = factors[0]  # Assume first mode is spatial
        
        # Reshape to (spatial_size, n_modes)
        self.modal_basis = spatial_factor
        
        # Compute reconstruction error
        reconstructed = self.decomposer.reconstruct()
        error = torch.norm(data - reconstructed) / torch.norm(data)
        
        return {
            'reconstruction_error': float(error.item()),
            'n_modes_spatial': self.config.n_spatial_modes,
            'n_modes_temporal': self.config.n_temporal_modes
        }
    
    def _optimize_sensor_placement(self,
                                   data: torch.Tensor,
                                   rejection_domain: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """Optimize sensor placement using tensor QR."""
        self.qr_decomposer = TensorTubeQRDecomposition(
            tensor=data,
            N=self.config.n_sensors,
            rejection_domain=rejection_domain,
            device=str(self.device),
            dtype=self.dtype,
            uniform_distribution=True
        )
        
        P, Q, R = self.qr_decomposer.factorize()
        self.sensor_locations = P
        
        actual_sensors = torch.sum(P).item()
        
        return {
            'requested_sensors': self.config.n_sensors,
            'actual_sensors': actual_sensors,
            'placement_efficiency': actual_sensors / self.config.n_sensors
        }
    
    def _calibrate_proxy_model(self,
                               data: torch.Tensor,
                               controls: Optional[List[List[WellControl]]] = None) -> Dict[str, any]:
        """Calibrate proxy model."""
        # Create appropriate proxy model
        if self.config.proxy_model_type == 'linear':
            self.proxy_model = LinearDynamicsProxyModel(
                spatial_shape=self.spatial_shape,
                modal_basis=self.modal_basis,
                device=str(self.device),
                dtype=self.dtype
            )
        elif self.config.proxy_model_type == 'neural':
            self.proxy_model = NeuralProxyModel(
                spatial_shape=self.spatial_shape,
                modal_basis=self.modal_basis,
                device=str(self.device),
                dtype=self.dtype
            )
        elif self.config.proxy_model_type == 'physics_informed':
            self.proxy_model = PhysicsInformedProxyModel(
                spatial_shape=self.spatial_shape,
                modal_basis=self.modal_basis,
                device=str(self.device),
                dtype=self.dtype
            )
        
        # Initialize scenario analyzer and calibrator
        self.scenario_analyzer = ScenarioAnalyzer(self.proxy_model)
        self.calibrator = ModelCalibrator(self.proxy_model)
        
        # Convert data to list of states
        historical_states = []
        for t in range(data.shape[-1]):
            pressure_field = data[..., t]
            state = ReservoirState(pressure=pressure_field, time=float(t))
            historical_states.append(state)
        
        # Calibrate
        if controls is None:
            # Create dummy controls if not provided
            controls = [[WellControl('dummy', 'rate', 0.0, (0, 0))] 
                       for _ in range(len(historical_states))]
        
        metrics = self.calibrator.calibrate_from_historical_data(
            historical_states, controls
        )
        
        return metrics
    
    def predict_next_state(self,
                          current_state: ReservoirState,
                          well_controls: List[WellControl],
                          time_horizon: float = 1.0,
                          time_steps: int = 1) -> List[ReservoirState]:
        """
        Predict next reservoir state(s) given well controls.
        
        Parameters
        ----------
        current_state : ReservoirState
            Current reservoir state
        well_controls : List[WellControl]
            Well control parameters
        time_horizon : float
            Forecast horizon
        time_steps : int
            Number of time steps to forecast
            
        Returns
        -------
        List[ReservoirState]
            Predicted states
        """
        if not self.state.is_calibrated:
            raise RuntimeError("Digital twin not trained. Call train() first.")
        
        forecasted_states = self.proxy_model.forecast(
            current_state, well_controls, time_horizon, time_steps
        )
        
        # Update internal state
        if forecasted_states:
            self.state.reservoir_state = forecasted_states[-1]
            self.state.current_time = forecasted_states[-1].time
        
        return forecasted_states
    
    def update_from_sensors(self,
                           sensor_readings: torch.Tensor,
                           sensor_locations: Optional[torch.Tensor] = None,
                           current_time: Optional[float] = None) -> Dict[str, any]:
        """
        Update digital twin from sensor readings.
        
        Performs:
        1. Field reconstruction from sparse sensors
        2. Comparison with prediction
        3. Alert checking
        4. Optional model updating
        
        Parameters
        ----------
        sensor_readings : torch.Tensor
            Readings from sensors
        sensor_locations : torch.Tensor, optional
            Sensor location mask (uses default if None)
        current_time : float, optional
            Current time
            
        Returns
        -------
        Dict[str, any]
            Update results and metrics
        """
        if not self.state.is_calibrated:
            raise RuntimeError("Digital twin not trained. Call train() first.")
        
        # Use default sensor locations if not provided
        if sensor_locations is None:
            sensor_locations = self.sensor_locations
        
        # Reconstruct full field from sensor readings
        reconstructed_field = self._reconstruct_from_sensors(
            sensor_readings, sensor_locations
        )
        
        # Create observed state
        observed_state = ReservoirState(
            pressure=reconstructed_field,
            time=current_time or self.state.current_time
        )
        
        # Compare with prediction if available
        metrics = {}
        if self.state.reservoir_state is not None:
            metrics = self.monitor.compare_prediction_observation(
                predicted=self.state.reservoir_state.pressure,
                observed=reconstructed_field,
                sensor_locations=sensor_locations
            )
            
            alert_status = self.monitor.check_alert_status(metrics)
            self.state.alert_status = alert_status
            self.state.prediction_error = metrics['relative_error']
            
            self.monitor.log_monitoring_event(
                observed_state.time, metrics, alert_status
            )
        
        # Update state
        self.state.reservoir_state = observed_state
        self.state.sensor_readings = sensor_readings
        self.state.current_time = observed_state.time
        self.state.last_update_time = observed_state.time
        
        # Add to history
        self.state.history['times'].append(observed_state.time)
        if 'relative_error' in metrics:
            self.state.history['errors'].append(metrics['relative_error'])
        self.state.history['alerts'].append(self.state.alert_status)
        
        # Online model update if needed
        if (len(self.state.history['times']) % self.config.update_frequency == 0):
            self.calibrator.online_update(observed_state, sensor_locations)
        
        return {
            'reconstructed_field': reconstructed_field,
            'metrics': metrics,
            'alert_status': self.state.alert_status
        }
    
    def _reconstruct_from_sensors(self,
                                 sensor_readings: torch.Tensor,
                                 sensor_locations: torch.Tensor) -> torch.Tensor:
        """Reconstruct full field from sparse sensor readings."""
        # Use compressive sensing reconstruction
        if self.config.use_geometry_aware and self.mesh is not None:
            # Geometry-aware reconstruction
            cs_config = TensorCSConfig(
                max_iterations=50,
                convergence_eps=1e-3,
                reconstruction_method=self.config.reconstruction_method
            )
            
            reconstructor = GeometryAwareTensorCS(
                dictionary=self.modal_basis,
                measurements=sensor_readings,
                measurement_matrix=sensor_locations,
                mesh=self.mesh,
                config=cs_config,
                device=str(self.device),
                dtype=self.dtype
            )
        else:
            # Standard reconstruction
            cs_config = TensorCSConfig(
                max_iterations=50,
                convergence_eps=1e-3,
                reconstruction_method=self.config.reconstruction_method
            )
            
            reconstructor = TensorCSReconstructor(
                dictionary=self.modal_basis,
                measurements=sensor_readings,
                measurement_matrix=sensor_locations,
                config=cs_config,
                device=str(self.device),
                dtype=self.dtype
            )
        
        reconstructed = reconstructor.reconstruct()
        return reconstructed.reshape(self.spatial_shape)
    
    def evaluate_scenarios(self,
                          scenarios: Dict[str, List[WellControl]],
                          time_horizon: float = 10.0,
                          time_steps: int = 10) -> Dict[str, Dict]:
        """
        Evaluate multiple what-if scenarios.
        
        Parameters
        ----------
        scenarios : Dict[str, List[WellControl]]
            Dictionary of scenario_name -> well_controls
        time_horizon : float
            Forecast horizon for each scenario
        time_steps : int
            Number of time steps
            
        Returns
        -------
        Dict[str, Dict]
            Results for each scenario
        """
        if not self.state.is_calibrated:
            raise RuntimeError("Digital twin not trained. Call train() first.")
        
        if self.state.reservoir_state is None:
            raise RuntimeError("No current state available. Run update_from_sensors() first.")
        
        results = {}
        
        for scenario_name, well_controls in scenarios.items():
            result = self.scenario_analyzer.evaluate_scenario(
                scenario_name=scenario_name,
                initial_state=self.state.reservoir_state,
                well_controls=well_controls,
                time_horizon=time_horizon,
                time_steps=time_steps
            )
            results[scenario_name] = result
        
        return results
    
    def get_monitoring_summary(self) -> Dict[str, any]:
        """Get summary of monitoring history."""
        if not self.state.history['times']:
            return {'status': 'no monitoring data'}
        
        errors = self.state.history['errors']
        alerts = self.state.history['alerts']
        
        summary = {
            'total_updates': len(self.state.history['times']),
            'current_time': self.state.current_time,
            'current_error': self.state.prediction_error,
            'current_alert_status': self.state.alert_status,
            'mean_error': float(np.mean(errors)) if errors else 0.0,
            'max_error': float(np.max(errors)) if errors else 0.0,
            'warning_count': alerts.count('warning'),
            'critical_count': alerts.count('critical'),
            'normal_count': alerts.count('normal')
        }
        
        return summary
    
    def save(self, save_dir: Union[str, Path]) -> None:
        """
        Save digital twin state to disk.
        
        Parameters
        ----------
        save_dir : str or Path
            Directory to save to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(save_dir / 'config.json', 'w') as f:
            # Convert config to dict (exclude non-serializable)
            config_dict = {
                'n_spatial_modes': self.config.n_spatial_modes,
                'n_temporal_modes': self.config.n_temporal_modes,
                'n_sensors': self.config.n_sensors,
                'proxy_model_type': self.config.proxy_model_type,
                'use_geometry_aware': self.config.use_geometry_aware,
                'reconstruction_method': self.config.reconstruction_method
            }
            json.dump(config_dict, f, indent=2)
        
        # Save modal basis
        torch.save(self.modal_basis, save_dir / 'modal_basis.pt')
        
        # Save sensor locations
        torch.save(self.sensor_locations, save_dir / 'sensor_locations.pt')
        
        # Save proxy model (if linear)
        if isinstance(self.proxy_model, LinearDynamicsProxyModel):
            torch.save({
                'A': self.proxy_model.A,
                'B': self.proxy_model.B
            }, save_dir / 'proxy_model.pt')
        
        # Save monitoring history
        with open(save_dir / 'monitoring_history.json', 'w') as f:
            json.dump(self.monitor.monitoring_history, f, indent=2)
        
        logger.info(f"Digital twin saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> 'DigitalTwinTBMD':
        """
        Load digital twin from disk.
        
        Parameters
        ----------
        load_dir : str or Path
            Directory to load from
            
        Returns
        -------
        DigitalTwinTBMD
            Loaded digital twin
        """
        load_dir = Path(load_dir)
        
        # Load configuration
        with open(load_dir / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = DigitalTwinConfig(**config_dict)
        twin = cls(config)
        
        # Load modal basis
        twin.modal_basis = torch.load(load_dir / 'modal_basis.pt')
        
        # Load sensor locations
        twin.sensor_locations = torch.load(load_dir / 'sensor_locations.pt')
        
        # Mark as calibrated
        twin.state.is_calibrated = True
        
        logger.info(f"Digital twin loaded from {load_dir}")
        return twin

