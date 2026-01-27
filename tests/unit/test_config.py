"""
Тесты для конфигурационных модулей
"""
import pytest
import torch
from TBMD.config import (
    BaseConfig,
    DecompositionConfig,
    GeometryAwareDecompositionConfig,
    SensorPlacementConfig,
    GeometricSensorConfig,
    ReconstructionConfig,
    GeometryAwareReconstructionConfig,
    DigitalTwinConfig
)


class TestBaseConfig:
    """Тесты для BaseConfig"""
    
    def test_default_values(self):
        """Тест значений по умолчанию"""
        config = BaseConfig()
        assert config.backend == 'pytorch'
        assert config.dtype == 'float32'
        assert config.seed == 0
        assert config.deterministic is True
    
    def test_auto_device_selection(self):
        """Тест автоматического выбора устройства"""
        config = BaseConfig(device=None)
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert config.device == expected
    
    def test_to_dict(self):
        """Тест преобразования в словарь"""
        config = BaseConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'backend' in config_dict
        assert 'device' in config_dict


class TestDecompositionConfig:
    """Тесты для DecompositionConfig"""
    
    def test_valid_ranks(self):
        """Тест валидных рангов"""
        config = DecompositionConfig(ranks=[50, 20])
        assert config.ranks == [50, 20]
    
    def test_invalid_ranks_negative(self):
        """Тест отрицательных рангов"""
        with pytest.raises(ValueError, match="должны быть положительными"):
            DecompositionConfig(ranks=[-1, 20])
    
    def test_invalid_energy_threshold(self):
        """Тест невалидного порога энергии"""
        with pytest.raises(ValueError, match="energy_threshold"):
            DecompositionConfig(energy_threshold=1.5)


class TestGeometryAwareDecompositionConfig:
    """Тесты для GeometryAwareDecompositionConfig"""
    
    def test_valid_alpha(self):
        """Тест валидного alpha"""
        config = GeometryAwareDecompositionConfig(alpha=0.1)
        assert config.alpha == 0.1
    
    def test_invalid_alpha(self):
        """Тест невалидного alpha"""
        with pytest.raises(ValueError, match="alpha должен быть"):
            GeometryAwareDecompositionConfig(alpha=1.5)
    
    def test_adaptive_alpha(self):
        """Тест адаптивного alpha"""
        config = GeometryAwareDecompositionConfig(
            alpha_adaptive=True,
            alpha_min=0.01,
            alpha_max=0.5
        )
        assert config.alpha_adaptive is True


class TestSensorPlacementConfig:
    """Тесты для SensorPlacementConfig"""
    
    def test_valid_n_sensors(self):
        """Тест валидного количества сенсоров"""
        config = SensorPlacementConfig(n_sensors=100)
        assert config.n_sensors == 100
    
    def test_invalid_n_sensors(self):
        """Тест невалидного количества сенсоров"""
        with pytest.raises(ValueError, match="должен быть положительным"):
            SensorPlacementConfig(n_sensors=-10)


class TestReconstructionConfig:
    """Тесты для ReconstructionConfig"""
    
    def test_valid_solver(self):
        """Тест валидного решателя"""
        config = ReconstructionConfig(solver='admm')
        assert config.solver == 'admm'
    
    def test_invalid_damping_factor(self):
        """Тест невалидного damping factor"""
        with pytest.raises(ValueError, match="damping_factor"):
            ReconstructionConfig(damping_factor=1.5)
    
    def test_convergence_parameters(self):
        """Тест параметров сходимости"""
        config = ReconstructionConfig(
            max_iterations=100,
            convergence_eps=1e-3
        )
        assert config.max_iterations == 100
        assert config.convergence_eps == 1e-3


class TestDigitalTwinConfig:
    """Тесты для DigitalTwinConfig"""
    
    def test_valid_configuration(self):
        """Тест валидной конфигурации"""
        config = DigitalTwinConfig(
            n_spatial_modes=40,
            n_sensors=30,
            forecaster_type='lstm'
        )
        assert config.n_spatial_modes == 40
        assert config.n_sensors == 30
        assert config.forecaster_type == 'lstm'
    
    def test_train_test_split_validation(self):
        """Тест валидации train/test split"""
        with pytest.raises(ValueError, match="train_test_split"):
            DigitalTwinConfig(train_test_split=1.5)
    
    def test_forecaster_config_defaults(self):
        """Тест значений по умолчанию для forecaster"""
        config = DigitalTwinConfig()
        assert 'hidden_size' in config.forecaster_config
        assert 'learning_rate' in config.forecaster_config

