"""
Тесты для Digital Twin
"""
import pytest
import torch
import numpy as np

from TBMD.core.digital_twin import DigitalTwin, DigitalTwinState
from TBMD.config import DigitalTwinConfig



class TestDigitalTwin:
    """Тесты для DigitalTwin"""
    
    def test_initialization(self):
        """Тест инициализации"""
        config = DigitalTwinConfig(n_spatial_modes=20, n_sensors=10)
        twin = DigitalTwin(config)
        
        assert twin.config == config
        assert twin.state.is_calibrated is False
        assert twin.spatial_modes is None
    
    def test_train(self):
        """Тест обучения digital twin"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_temporal_modes=5,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # Создать исторические данные
        historical_data = torch.randn(50, 3, 20)  # (I=50, J=3, T=20)
        
        twin.train(historical_data, normalize=True)
        
        assert twin.state.is_calibrated is True
        assert twin.spatial_modes is not None
        assert twin.spatial_modes.shape == (150, 10)  # (I*J, R)
        assert twin.sensor_indices is not None
        assert len(twin.sensor_indices) == 15
    
    def test_predict(self):
        """Тест прогнозирования"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_temporal_modes=5,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # Обучение
        historical_data = torch.randn(50, 3, 20)
        twin.train(historical_data, normalize=True)
        
        # Прогноз
        current_state = torch.randn(50, 3)
        forecast = twin.predict(current_state, n_steps=5)
        
        assert forecast.shape == (50, 3, 5)
    
    def test_update_from_sensors(self):
        """Тест обновления из сенсоров"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # Обучение
        historical_data = torch.randn(50, 3, 20)
        twin.train(historical_data, normalize=True)
        
        # Измерения с сенсоров
        sensor_readings = torch.randn(15, 3)
        
        reconstructed = twin.update_from_sensors(sensor_readings)
        
        assert reconstructed.shape == (50, 3)
        assert len(twin.state.history['observations']) == 1
    
    def test_evaluate_scenarios(self):
        """Тест сценарного анализа"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # Обучение
        historical_data = torch.randn(50, 3, 20)
        twin.train(historical_data, normalize=True)
        
        # Сценарии
        scenarios = [
            {'name': 'baseline'},
            {'name': 'optimistic'},
            {'name': 'pessimistic'}
        ]
        
        results = twin.evaluate_scenarios(scenarios, n_steps=5)
        
        assert len(results) == 3
        assert 'baseline' in results
        assert 'mean_value' in results['baseline']
    
    def test_detect_anomalies(self):
        """Тест детекции аномалий"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # Обучение
        historical_data = torch.randn(50, 3, 20)
        twin.train(historical_data, normalize=True)
        
        # Данные с сенсоров
        sensor_data = torch.randn(15, 10)
        
        anomalies = twin.detect_anomalies(sensor_data, threshold=3.0)
        
        assert isinstance(anomalies, list)
    
    def test_get_sensor_locations(self):
        """Тест получения расположения сенсоров"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # Обучение
        historical_data = torch.randn(50, 3, 20)
        twin.train(historical_data, normalize=True)
        
        locations = twin.get_sensor_locations()
        
        assert len(locations) == 15
        assert isinstance(locations, np.ndarray)
    
    def test_get_statistics(self):
        """Тест получения статистики"""
        config = DigitalTwinConfig(
            n_spatial_modes=10,
            n_sensors=15,
            verbose=False
        )
        twin = DigitalTwin(config)
        
        # До обучения
        stats_before = twin.get_statistics()
        assert stats_before['is_calibrated'] is False
        
        # После обучения
        historical_data = torch.randn(50, 3, 20)
        twin.train(historical_data)
        
        stats_after = twin.get_statistics()
        assert stats_after['is_calibrated'] is True
        assert stats_after['n_spatial_modes'] == 10
        assert stats_after['n_sensors'] == 15
    
    def test_not_calibrated_error(self):
        """Тест ошибки при использовании до обучения"""
        config = DigitalTwinConfig(n_spatial_modes=10, n_sensors=15)
        twin = DigitalTwin(config)
        
        current_state = torch.randn(50, 3)
        
        with pytest.raises(ValueError, match="не обучен"):
            twin.predict(current_state)


class TestDigitalTwinState:
    """Тесты для DigitalTwinState"""
    
    def test_initialization(self):
        """Тест инициализации состояния"""
        state = DigitalTwinState()
        
        assert state.current_time == 0.0
        assert state.modal_coefficients is None
        assert state.is_calibrated is False
        assert state.alert_status == 'normal'
        assert 'times' in state.history
        assert 'errors' in state.history

