"""
Тесты для модулей размещения сенсоров
"""
import pytest
import torch
import numpy as np

from TBMD.core.sensor_placement import TensorTubeQRDecomposition
from TBMD.config import SensorPlacementConfig


class TestTensorTubeQRDecomposition:
    """Тесты для TensorTubeQRDecomposition"""
    
    def test_initialization(self, sample_tensor_small, sensor_config):
        """Тест инициализации"""
        qr = TensorTubeQRDecomposition(sample_tensor_small, config=sensor_config)
        assert qr.config.n_sensors == sensor_config.n_sensors
        assert qr.P is None
    
    def test_basic_sensor_placement(self, sample_tensor_small, sensor_config):
        """Тест базового размещения сенсоров"""
        qr = TensorTubeQRDecomposition(sample_tensor_small, config=sensor_config)
        P, Q, R = qr.factorize()
        
        # P is a binary mask tensor of shape (I, ...), or (I, J, ...)
        assert P is not None
        assert Q is not None
        assert R is not None
        
        # Check actual number of sensors placed
        n_placed = torch.sum(P).item()
        assert n_placed <= sensor_config.n_sensors
            
        assert P.shape == sample_tensor_small.shape[:-1]

    def test_factorize_method(self, sample_tensor_small, sensor_config):
        """Тест метода factorize"""
        qr = TensorTubeQRDecomposition(sample_tensor_small, config=sensor_config)
        P, Q, R = qr.factorize()
        
        assert isinstance(P, torch.Tensor)
        assert isinstance(Q, torch.Tensor)
        assert isinstance(R, torch.Tensor)
    
    def test_sensor_indices_uniqueness(self, sample_tensor_small, sensor_config):
        """Тест уникальности индексов сенсоров"""
        qr = TensorTubeQRDecomposition(sample_tensor_small, config=sensor_config)
        P, _, _ = qr.factorize()
        
        # P is a binary mask efficiently ensuring uniqueness of locations
        # Check that values are 0 or 1
        unique_vals = torch.unique(P)
        assert torch.all(torch.isin(unique_vals, torch.tensor([0, 1], device=P.device, dtype=torch.int32)))

    def test_too_many_sensors(self, sensor_config):
        """Тест когда сенсоров больше чем точек"""
        small_tensor = torch.randn(3, 3, 5) # 9 spatial points
        
        # Config asks for 20 sensors, but only 9 locations
        config = SensorPlacementConfig(n_sensors=20, verbose=False)
        qr = TensorTubeQRDecomposition(small_tensor, config=config)
        P, _, _ = qr.factorize()
        
        n_placed = torch.sum(P).item()
        assert n_placed <= 9
    
    def test_invalid_input_dimension(self, sensor_config):
        """Тест с невалидной размерностью входных данных"""
        tensor_2d = torch.randn(10, 5) # 2D < 3D
        
        with pytest.raises(ValueError, match="at least 3 dimensions"): 
             TensorTubeQRDecomposition(tensor_2d, config=sensor_config)
    
    def test_check_factorization(self, sample_tensor_small, sensor_config):
        """Тест метода check_factorization"""
        qr = TensorTubeQRDecomposition(sample_tensor_small, config=sensor_config)
        qr.factorize()
        
        is_valid, error, metrics = qr.check_factorization()
        assert isinstance(is_valid, bool)
        assert isinstance(error, float)
        assert isinstance(metrics, dict)
        assert 'sensor_count' in metrics

    def test_get_algorithm_info(self, sample_tensor_small, sensor_config):
        """Тест метода get_algorithm_info"""
        qr = TensorTubeQRDecomposition(sample_tensor_small, config=sensor_config)
        qr.factorize()
        
        info = qr.get_algorithm_info()
        assert isinstance(info, dict)
        assert 'tensor_shape' in info
        assert 'actual_sensors' in info

