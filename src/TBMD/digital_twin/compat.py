"""
Compatibility module for legacy components.

This module provides implementations of helper classes that were previously
bundled in DigitalTwinTBMD.py. They are kept here for backward compatibility
and modular usage.
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from TBMD.core.forecasting.ReservoirProxyModel import (
    LinearDynamicsProxyModel,
    NeuralProxyModel,
    ReservoirProxyModelBase,
    ReservoirState,
    WellControl,
)

logger = logging.getLogger(__name__)


class RealtimeMonitor:
    """
    Real-time monitoring component.
    """

    def __init__(self, alert_threshold: float = 0.15):
        self.alert_threshold = alert_threshold
        self.monitoring_history = []

    def compare_prediction_observation(
        self, predicted: torch.Tensor, observed: torch.Tensor, sensor_locations: torch.Tensor
    ) -> Dict[str, float]:
        """Compare predicted and observed fields at sensor locations."""
        sensor_mask = sensor_locations.bool()
        pred_at_sensors = predicted[sensor_mask]
        obs_at_sensors = observed[sensor_mask]

        # Avoid division by zero
        norm_obs = torch.norm(obs_at_sensors)
        if norm_obs == 0:
            norm_obs = 1.0

        absolute_error = torch.abs(pred_at_sensors - obs_at_sensors)
        relative_error = torch.norm(absolute_error) / norm_obs
        max_error = torch.max(absolute_error)
        mean_error = torch.mean(absolute_error)

        metrics = {
            "relative_error": float(relative_error.item())
            if hasattr(relative_error, "item")
            else float(relative_error),
            "max_error": float(max_error.item())
            if hasattr(max_error, "item")
            else float(max_error),
            "mean_error": float(mean_error.item())
            if hasattr(mean_error, "item")
            else float(mean_error),
            "rmse": float(torch.sqrt(torch.mean(absolute_error**2)).item()),
        }

        return metrics

    def check_alert_status(self, metrics: Dict[str, float]) -> str:
        """Determine alert status."""
        rel_error = metrics.get("relative_error", 0.0)

        if rel_error > 2 * self.alert_threshold:
            return "critical"
        elif rel_error > self.alert_threshold:
            return "warning"
        else:
            return "normal"

    def log_monitoring_event(
        self, time: float, metrics: Dict[str, float], alert_status: str
    ) -> None:
        """Log monitoring event."""
        event = {
            "time": time,
            "metrics": metrics,
            "alert_status": alert_status,
            "timestamp": datetime.now().isoformat(),
        }
        self.monitoring_history.append(event)


class ScenarioAnalyzer:
    """
    Scenario analysis component.
    """

    def __init__(self, proxy_model: ReservoirProxyModelBase):
        self.proxy_model = proxy_model
        self.scenario_results = {}

    def evaluate_scenario(
        self,
        scenario_name: str,
        initial_state: ReservoirState,
        well_controls: List[WellControl],
        time_horizon: float,
        time_steps: int,
    ) -> Dict[str, any]:
        """Evaluate a single scenario."""
        logger.info(f"Evaluating scenario: {scenario_name}")

        forecasted_states = self.proxy_model.forecast(
            initial_state, well_controls, time_horizon, time_steps
        )

        kpis = self._compute_kpis(forecasted_states, well_controls)

        result = {
            "scenario_name": scenario_name,
            "forecasted_states": forecasted_states,
            "kpis": kpis,
            "well_controls": well_controls,
        }

        self.scenario_results[scenario_name] = result
        return result

    def compare_scenarios(
        self, scenario_names: List[str], kpi_name: str = "total_production"
    ) -> Dict[str, float]:
        """Compare multiple scenarios."""
        comparison = {}
        for name in scenario_names:
            if name in self.scenario_results:
                kpi_value = self.scenario_results[name]["kpis"].get(kpi_name, None)
                comparison[name] = kpi_value
        return comparison

    def _compute_kpis(
        self, forecasted_states: List[ReservoirState], well_controls: List[WellControl]
    ) -> Dict[str, float]:
        """Compute KPIs."""
        kpis = {}
        if not forecasted_states:
            return kpis

        # Average pressure
        avg_pressures = [torch.mean(state.pressure).item() for state in forecasted_states]
        kpis["avg_pressure"] = float(np.mean(avg_pressures))
        kpis["min_pressure"] = float(np.min(avg_pressures))
        kpis["max_pressure"] = float(np.max(avg_pressures))

        # Total production (simplified)
        production_wells = [ctrl for ctrl in well_controls if ctrl.value < 0]
        kpis["total_production"] = abs(sum(ctrl.value for ctrl in production_wells)) * len(
            forecasted_states
        )

        # Total injection
        injection_wells = [ctrl for ctrl in well_controls if ctrl.value > 0]
        kpis["total_injection"] = sum(ctrl.value for ctrl in injection_wells) * len(
            forecasted_states
        )

        return kpis


class ModelCalibrator:
    """
    Model calibration component.
    """

    def __init__(self, proxy_model: ReservoirProxyModelBase):
        self.proxy_model = proxy_model
        self.calibration_history = []

    def calibrate_from_historical_data(
        self, historical_states: List[ReservoirState], historical_controls: List[List[WellControl]]
    ) -> Dict[str, float]:
        """Calibrate proxy model."""
        logger.info("Calibrating proxy model from historical data...")

        if isinstance(self.proxy_model, LinearDynamicsProxyModel):
            metrics = self.proxy_model.calibrate(historical_states, historical_controls)
        elif isinstance(self.proxy_model, NeuralProxyModel):
            metrics = self.proxy_model.train_model(historical_states, historical_controls)
        else:
            metrics = {"status": "calibration not implemented for this model type"}

        self.calibration_history.append(
            {"timestamp": datetime.now().isoformat(), "metrics": metrics}
        )

        return metrics

    def online_update(self, observed_state: ReservoirState, sensor_locations: torch.Tensor) -> None:
        """Perform online update."""
        logger.info("Performing online model update...")
        self.proxy_model.update_from_observations(observed_state, sensor_locations)
