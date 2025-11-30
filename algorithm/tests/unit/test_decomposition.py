"""
Тесты для модулей декомпозиции
"""
import pytest
import torch
import numpy as np

from TBMD.core.decomposition import TuckerDecomposer, DecompositionResult
from TBMD.config import DecompositionConfig


class TestTuckerDecomposer:
    """Тесты для TuckerDecomposer"""
    
    def test_initialization(self, decomposition_config):
        """Тест инициализации"""
        decomposer = TuckerDecomposer(decomposition_config)
        assert decomposer.config == decomposition_config
        assert decomposer.result is None
    
    def test_basic_decomposition(self, sample_tensor_medium, decomposition_config):
        """Тест базовой декомпозиции"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        assert isinstance(result, DecompositionResult)
        assert result.spatial_modes.shape[1] == 20  # rank from config
        assert result.temporal_modes.shape[1] == 10
        assert result.reconstruction_error < 1.0  # Should be reasonable
    
    def test_automatic_rank_selection(self, sample_tensor_medium):
        """Тест автоматического выбора рангов"""
        config = DecompositionConfig(
            ranks=None,  # Auto selection
            energy_threshold=0.95,
            verbose=False
        )
        decomposer = TuckerDecomposer(config)
        result = decomposer.decompose(sample_tensor_medium)
        
        assert result.energy_retained >= 0.95
        assert 'ranks' in result.metadata
        assert result.metadata['ranks'] is not None
    
    def test_reconstruction(self, sample_tensor_medium, decomposition_config):
        """Тест реконструкции"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        reconstructed = decomposer.reconstruct(result)
        
        assert reconstructed.shape == sample_tensor_medium.shape
        
        # Проверить ошибку реконструкции
        error = torch.norm(sample_tensor_medium - reconstructed) / torch.norm(sample_tensor_medium)
        assert error < 0.5  # Should be reasonable
    
    def test_invalid_tensor_dimension(self, decomposition_config):
        """Тест с невалидной размерностью тензора"""
        tensor_2d = torch.randn(10, 10)
        decomposer = TuckerDecomposer(decomposition_config)
        
        with pytest.raises(ValueError, match="Ожидается 3D тензор"):
            decomposer.decompose(tensor_2d)
    
    def test_get_spatial_modes(self, sample_tensor_medium, decomposition_config):
        """Тест получения пространственных мод"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        # Получить все моды
        all_modes = decomposer.get_spatial_modes()
        assert all_modes.shape[1] == 20
        
        # Получить первые 10 мод
        first_10 = decomposer.get_spatial_modes(n_modes=10)
        assert first_10.shape[1] == 10
    
    def test_compute_modal_coefficients(self, sample_tensor_medium, decomposition_config):
        """Тест вычисления модальных коэффициентов"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        # Взять временной срез
        field = sample_tensor_medium[:, 0, 0]  # (I,)
        
        # Вычислить коэффициенты
        coeffs = decomposer.compute_modal_coefficients(field)
        
        assert coeffs.shape[0] == 20  # rank
    
    def test_reconstruct_from_modes(self, sample_tensor_medium, decomposition_config):
        """Тест реконструкции из мод"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        # Получить моды
        spatial_modes = result.spatial_modes
        temporal_coeffs = spatial_modes.T @ sample_tensor_medium[:, :, 0]
        
        # Реконструировать
        reconstructed = decomposer.reconstruct_from_modes(
            spatial_modes,
            temporal_coeffs
        )
        
        assert reconstructed.shape == (sample_tensor_medium.shape[0], sample_tensor_medium.shape[1])
    
    def test_energy_computation(self, sample_tensor_small, decomposition_config):
        """Тест вычисления энергии"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_small)
        
        assert 0 <= result.energy_retained <= 1
        assert result.energy_retained > 0.5  # Should retain reasonable energy
    
    def test_metadata(self, sample_tensor_medium, decomposition_config):
        """Тест метаданных результата"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        assert 'ranks' in result.metadata
        assert 'method' in result.metadata
        assert 'backend' in result.metadata
        assert 'input_shape' in result.metadata
        assert result.metadata['input_shape'] == tuple(sample_tensor_medium.shape)


class TestDecompositionResult:
    """Тесты для DecompositionResult"""
    
    def test_to_dict(self, sample_tensor_medium, decomposition_config):
        """Тест преобразования результата в словарь"""
        decomposer = TuckerDecomposer(decomposition_config)
        result = decomposer.decompose(sample_tensor_medium)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'spatial_modes_shape' in result_dict
        assert 'reconstruction_error' in result_dict
        assert 'energy_retained' in result_dict
        assert 'metadata' in result_dict

