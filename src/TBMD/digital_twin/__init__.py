"""
Digital Twin Module

Reservoir digital twin implementation built on TBMD.

Main module: digital_twin.py provides the DigitalTwin class with:
- forecasting models (Linear, MLP, LSTM)
- proxy models for scenario analysis (LinearDynamics, Neural, PhysicsInformed)

The optional compatibility layer exposes monitoring-oriented components.
"""

# Main Digital Twin implementation
from .digital_twin import DigitalTwin, DigitalTwinState, ForecasterType, ProxyModelType

# Extended system with monitoring components (optional)
# Extended system components from compat (for backward compatibility)
from .compat import (
    RealtimeMonitor,
    ScenarioAnalyzer,
    ModelCalibrator,
)

# Re-export data classes and proxy models from models
from TBMD.core.forecasting.ReservoirProxyModel import (
    WellControl, 
    ReservoirState,
    ReservoirProxyModelBase,
    LinearDynamicsProxyModel,
    NeuralProxyModel,
    PhysicsInformedProxyModel
)

__all__ = [
    # Main classes
    'DigitalTwin',
    'DigitalTwinState',
    'ForecasterType',
    'ProxyModelType',
    'DigitalTwinTBMD',
    
    # System components
    # 'SystemDigitalTwinConfig',
    # 'SystemDigitalTwinState', 
    'RealtimeMonitor',
    'ScenarioAnalyzer',
    'ModelCalibrator',
    
    # Data classes
    'WellControl',
    'ReservoirState',
    
    # Proxy Models
    'ReservoirProxyModelBase',
    'LinearDynamicsProxyModel',
    'NeuralProxyModel',
    'PhysicsInformedProxyModel',
]

