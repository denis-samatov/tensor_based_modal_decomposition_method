"""
Digital Twin Module

Цифровой двойник месторождения с TBMD

Основной модуль: digital_twin.py - содержит DigitalTwin класс с:
- Forecasting моделями (Linear, MLP, LSTM)
- Proxy моделями для сценарного анализа (LinearDynamics, Neural, PhysicsInformed)

Дополнительный модуль: system.py - содержит расширенную версию с компонентами мониторинга
"""

# Main Digital Twin implementation
from .digital_twin import DigitalTwin, DigitalTwinState, ForecasterType, ProxyModelType

# Extended system with monitoring components (optional)
# Extended system with monitoring components (optional)
# from .system import (
#     DigitalTwinTBMD,
#     DigitalTwinConfig as SystemDigitalTwinConfig,
#     DigitalTwinState as SystemDigitalTwinState,
#     RealtimeMonitor,
#     ScenarioAnalyzer,
#     ModelCalibrator,
# )

# Re-export data classes and proxy models from models
from TBMD.models.ReservoirProxyModel import (
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
    # 'DigitalTwinTBMD',
    
    # System components
    # 'SystemDigitalTwinConfig',
    # 'SystemDigitalTwinState', 
    # 'RealtimeMonitor',
    # 'ScenarioAnalyzer',
    # 'ModelCalibrator',
    
    # Data classes
    'WellControl',
    'ReservoirState',
    
    # Proxy Models
    'ReservoirProxyModelBase',
    'LinearDynamicsProxyModel',
    'NeuralProxyModel',
    'PhysicsInformedProxyModel',
]


