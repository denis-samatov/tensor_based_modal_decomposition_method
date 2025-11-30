"""
Тесты для модулей размещения сенсоров
"""
import pytest
import torch
import numpy as np

from TBMD.core.sensor_placement import TensorTubeQRDecomposition, SensorPlacementResult
from TBMD.config import SensorPlacementConfig


class TestTensorTubeQRDecomposition:
    """Тесты для TensorTubeQRDecomposition"""
    
    def test_initialization(self, sensor_config):
        """Тест инициализации"""
        qr = TensorTubeQRDecomposition(sensor_config)
        assert qr.config == sensor_config
        assert qr.result is None
    
    def test_basic_sensor_placement(self, sample_spatial_modes, sensor_config):
        """Тест базового размещения сенсоров"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        assert isinstance(result, SensorPlacementResult)
        assert len(result.sensor_indices) == 30  # from config
        assert result.measurement_matrix.shape == (30, 100)  # n_sensors × I
    
    def test_factorize_method(self, sample_spatial_modes, sensor_config):
        """Тест метода factorize (алиас для place_sensors)"""
        qr = TensorTubeQRDecomposition(sensor_config)
        P, Q, R = qr.factorize(sample_spatial_modes)
        
        assert P is not None
        assert P.shape == (30, 100)
    
    def test_sensor_indices_uniqueness(self, sample_spatial_modes, sensor_config):
        """Тест уникальности индексов сенсоров"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        # Все индексы должны быть уникальными
        assert len(result.sensor_indices) == len(set(result.sensor_indices))
    
    def test_required_indices_constraint(self, sample_spatial_modes):
        """Тест ограничения required_indices"""
        required = [0, 10, 20]
        config = SensorPlacementConfig(
            n_sensors=30,
            required_indices=required,
            verbose=False
        )
        
        qr = TensorTubeQRDecomposition(config)
        result = qr.place_sensors(sample_spatial_modes)
        
        # Все required индексы должны быть включены
        for req_idx in required:
            assert req_idx in result.sensor_indices
    
    def test_forbidden_indices_constraint(self, sample_spatial_modes):
        """Тест ограничения forbidden_indices"""
        forbidden = [0, 1, 2, 3, 4]
        config = SensorPlacementConfig(
            n_sensors=30,
            forbidden_indices=forbidden,
            verbose=False
        )
        
        qr = TensorTubeQRDecomposition(config)
        result = qr.place_sensors(sample_spatial_modes)
        
        # Ни один forbidden индекс не должен быть включен
        for forb_idx in forbidden:
            assert forb_idx not in result.sensor_indices
    
    def test_too_many_sensors(self, sensor_config):
        """Тест когда сенсоров больше чем точек"""
        small_modes = torch.randn(10, 5)  # только 10 точек
        
        config = SensorPlacementConfig(n_sensors=20, verbose=False)  # 20 сенсоров
        qr = TensorTubeQRDecomposition(config)
        result = qr.place_sensors(small_modes)
        
        # Должно скорректироваться до 10
        assert len(result.sensor_indices) <= 10
    
    def test_invalid_input_dimension(self, sensor_config):
        """Тест с невалидной размерностью входных данных"""
        tensor_3d = torch.randn(10, 10, 5)
        qr = TensorTubeQRDecomposition(sensor_config)
        
        with pytest.raises(ValueError, match="должна быть 2D матрицей"):
            qr.place_sensors(tensor_3d)
    
    def test_condition_number_computation(self, sample_spatial_modes, sensor_config):
        """Тест вычисления числа обусловленности"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        assert 'condition_number' in result.metadata
        assert result.metadata['condition_number'] > 0
    
    def test_importance_scores(self, sample_spatial_modes, sensor_config):
        """Тест оценок важности позиций"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        assert result.importance_scores is not None
        # Scores должны быть нормализованы к [0, 1]
        assert result.importance_scores.max() <= 1.0
        assert result.importance_scores.min() >= 0.0
    
    def test_get_sensor_locations_indices_only(self, sample_spatial_modes, sensor_config):
        """Тест получения расположения сенсоров (только индексы)"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        locations = qr.get_sensor_locations()
        assert locations.shape == (30, 1)  # n_sensors × 1 (indices as coords)
    
    def test_get_sensor_locations_with_coordinates(self, sample_spatial_modes, sensor_config):
        """Тест получения расположения с координатами"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        # Создать фиктивные координаты
        coords = np.random.rand(100, 2)  # 100 точек, 2D координаты
        
        locations = qr.get_sensor_locations(coords)
        assert locations.shape == (30, 2)  # n_sensors × 2D
    
    def test_compute_coverage_quality(self, sample_spatial_modes, sensor_config):
        """Тест оценки качества покрытия"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        quality = qr.compute_coverage_quality(sample_spatial_modes)
        
        assert 0 <= quality <= 1
        assert quality > 0  # Should have some coverage
    
    def test_measurement_matrix_structure(self, sample_spatial_modes, sensor_config):
        """Тест структуры measurement matrix"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        P = result.measurement_matrix
        
        # P должна быть разреженной (sparse)
        assert P.is_sparse
        
        # Каждая строка должна иметь ровно один ненулевой элемент
        P_dense = P.to_dense()
        assert torch.all(P_dense.sum(dim=1) == 1.0)
    
    def test_metadata_completeness(self, sample_spatial_modes, sensor_config):
        """Тест полноты метаданных"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        assert 'method' in result.metadata
        assert 'n_sensors' in result.metadata
        assert 'condition_number' in result.metadata
        assert 'pivoting_strategy' in result.metadata
    
    def test_fallback_method(self, sensor_config):
        """Тест fallback метода при ошибке QR"""
        # Создать особую матрицу которая может вызвать проблемы
        problematic_modes = torch.randn(100, 20)
        
        qr = TensorTubeQRDecomposition(sensor_config)
        
        # Должно работать даже если основной метод не работает
        result = qr.place_sensors(problematic_modes)
        
        assert len(result.sensor_indices) == 30
        assert result.importance_scores is not None


class TestSensorPlacementResult:
    """Тесты для SensorPlacementResult"""
    
    def test_result_structure(self, sample_spatial_modes, sensor_config):
        """Тест структуры результата"""
        qr = TensorTubeQRDecomposition(sensor_config)
        result = qr.place_sensors(sample_spatial_modes)
        
        # Проверить все поля
        assert hasattr(result, 'sensor_indices')
        assert hasattr(result, 'measurement_matrix')
        assert hasattr(result, 'Q_matrix')
        assert hasattr(result, 'R_matrix')
        assert hasattr(result, 'importance_scores')
        assert hasattr(result, 'metadata')

