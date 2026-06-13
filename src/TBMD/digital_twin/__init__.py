"""
Digital Twin Module

Reservoir digital twin implementation built on TBMD.

Main module: digital_twin.py provides the DigitalTwin class with:
- forecasting models (Linear, MLP, LSTM)
- proxy models for scenario analysis (LinearDynamics, Neural, PhysicsInformed)

The optional compatibility layer exposes monitoring-oriented components.
"""

# Main Digital Twin implementation
# Re-export data classes and proxy models from models
from TBMD.core.forecasting.ReservoirProxyModel import (
    LinearDynamicsProxyModel,
    NeuralProxyModel,
    PhysicsInformedProxyModel,
    ReservoirProxyModelBase,
    ReservoirState,
    WellControl,
)

# Extended system with monitoring components (optional)
# Extended system components from compat (for backward compatibility)
from .compat import (
    ModelCalibrator,
    RealtimeMonitor,
    ScenarioAnalyzer,
)
from .digital_twin import DigitalTwin, DigitalTwinState, ForecasterType, ProxyModelType

__all__ = [
    # Main classes
    "DigitalTwin",
    "DigitalTwinState",
    "ForecasterType",
    "ProxyModelType",
    "DigitalTwinTBMD",
    # System components
    # 'SystemDigitalTwinConfig',
    # 'SystemDigitalTwinState',
    "RealtimeMonitor",
    "ScenarioAnalyzer",
    "ModelCalibrator",
    # Data classes
    "WellControl",
    "ReservoirState",
    # Proxy Models
    "ReservoirProxyModelBase",
    "LinearDynamicsProxyModel",
    "NeuralProxyModel",
    "PhysicsInformedProxyModel",
]
