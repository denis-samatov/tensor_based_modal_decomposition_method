"""
Тесты для модулей реконструкции
"""
import pytest
import torch
import numpy as np

from TBMD.core.reconstruction import TensorCompressiveSensing, ReconstructionResult
from TBMD.config import ReconstructionConfig


class TestTensorCompressiveSensing:
    """Тесты для TensorCompressiveSensing"""
    
    def test_initialization(self, reconstruction_config):
        """Тест инициализации"""
        cs = TensorCompressiveSensing(reconstruction_config)
        assert cs.config == reconstruction_config
        assert cs.result is None
    
    def test_basic_reconstruction_admm(self, sample_spatial_modes, sample_measurements):
        """Тест базовой реконструкции с ADMM"""
        # Создать measurement matrix
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(
            solver='admm',
            max_iterations=50,
            convergence_eps=1e-2,
            verbose=False
        )
        
        cs = TensorCompressiveSensing(config)
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        assert isinstance(result, ReconstructionResult)
        assert result.reconstructed_field.shape == (sample_spatial_modes.shape[0], sample_measurements.shape[1])
        assert result.iterations <= 50
    
    def test_reconstruction_with_ground_truth(self, sample_spatial_modes, sample_measurements):
        """Тест реконструкции с ground truth"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        # Создать ground truth
        ground_truth = torch.randn(I, sample_measurements.shape[1])
        
        config = ReconstructionConfig(solver='admm', max_iterations=30, verbose=False)
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P,
            ground_truth=ground_truth
        )
        
        assert result.reconstruction_error is not None
        assert result.reconstruction_error >= 0
    
    def test_solver_least_squares(self, sample_spatial_modes, sample_measurements):
        """Тест решателя least squares"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(solver='least_squares', verbose=False)
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        # Least squares должен сходиться за 1 итерацию
        assert result.iterations == 1
        assert len(result.convergence_history) == 1
    
    def test_solver_ista(self, sample_spatial_modes, sample_measurements):
        """Тест решателя ISTA"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(
            solver='ista',
            max_iterations=50,
            l1_lambda=0.01,
            verbose=False
        )
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        assert result.iterations <= 50
        assert len(result.convergence_history) > 0
    
    def test_solver_fista(self, sample_spatial_modes, sample_measurements):
        """Тест решателя FISTA (Fast ISTA)"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(
            solver='fista',
            max_iterations=50,
            l1_lambda=0.01,
            verbose=False
        )
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        assert result.iterations <= 50
    
    def test_convergence_history(self, sample_spatial_modes, sample_measurements):
        """Тест истории сходимости"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(
            solver='admm',
            max_iterations=20,
            verbose=False
        )
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        # История должна быть монотонно убывающей (в общем случае)
        assert len(result.convergence_history) > 0
        # Residual должен уменьшаться
        if len(result.convergence_history) > 1:
            # В конце residual должен быть меньше чем в начале
            assert result.convergence_history[-1] < result.convergence_history[0]
    
    def test_invalid_dictionary_dimension(self, sample_measurements):
        """Тест с невалидной размерностью словаря"""
        dictionary_3d = torch.randn(10, 10, 5)
        P = torch.eye(10).to_sparse()
        
        config = ReconstructionConfig(verbose=False)
        cs = TensorCompressiveSensing(config)
        
        with pytest.raises(ValueError, match="Dictionary должен быть 2D"):
            cs.reconstruct(dictionary_3d, sample_measurements, P)
    
    def test_incompatible_dimensions(self, sample_spatial_modes, sample_measurements):
        """Тест с несовместимыми размерностями"""
        # Создать P с неправильным размером
        wrong_size = 50  # не совпадает с I = 100
        P = torch.eye(30, wrong_size).to_sparse()
        
        config = ReconstructionConfig(verbose=False)
        cs = TensorCompressiveSensing(config)
        
        with pytest.raises(ValueError, match="не совпадает"):
            cs.reconstruct(sample_spatial_modes, sample_measurements, P)
    
    def test_metadata_completeness(self, sample_spatial_modes, sample_measurements):
        """Тест полноты метаданных"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(solver='admm', verbose=False)
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        assert 'solver' in result.metadata
        assert 'l1_lambda' in result.metadata
        assert 'l2_lambda' in result.metadata
        assert 'damping_factor' in result.metadata
        assert 'converged' in result.metadata
    
    def test_residual_computation(self, sample_spatial_modes, sample_measurements):
        """Тест вычисления residual"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(solver='admm', max_iterations=20, verbose=False)
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        assert result.residual >= 0
        # Residual должен быть разумным
        assert result.residual < 10.0
    
    def test_warm_start(self, sample_spatial_modes, sample_measurements):
        """Тест warm start"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(
            solver='admm',
            warm_start=True,
            initial_guess='least_squares',
            verbose=False
        )
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        # С warm start может сходиться быстрее
        assert result.iterations > 0


class TestReconstructionResult:
    """Тесты для ReconstructionResult"""
    
    def test_result_structure(self, sample_spatial_modes, sample_measurements):
        """Тест структуры результата"""
        n_sensors = 30
        I = sample_spatial_modes.shape[0]
        sensor_indices = np.random.choice(I, n_sensors, replace=False)
        
        P = torch.zeros(n_sensors, I)
        P[range(n_sensors), sensor_indices] = 1.0
        P = P.to_sparse()
        
        config = ReconstructionConfig(solver='admm', max_iterations=10, verbose=False)
        cs = TensorCompressiveSensing(config)
        
        result = cs.reconstruct(
            dictionary=sample_spatial_modes,
            measurements=sample_measurements,
            measurement_matrix=P
        )
        
        # Проверить все поля
        assert hasattr(result, 'reconstructed_field')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'residual')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'convergence_history')
        assert hasattr(result, 'reconstruction_error')
        assert hasattr(result, 'metadata')

