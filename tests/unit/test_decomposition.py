"""Tests for decomposition modules."""
import pytest
import torch
import numpy as np

from TBMD.core.decomposition import TuckerDecomposer, HOSVDDecompositionResult as DecompositionResult
from TBMD.config import DecompositionConfig


class TestTuckerDecomposer:
    """Tests for TuckerDecomposer."""
    
    def test_initialization(self, sample_tensor_medium, decomposition_config):
        """Test initialization."""
        decomposer = TuckerDecomposer(sample_tensor_medium, config=decomposition_config)
        assert decomposer.config == decomposition_config
        # Cores/factors should raise error before decompose
        with pytest.raises(Exception):
            _ = decomposer.cores
    
    def test_basic_decomposition(self, sample_tensor_medium, decomposition_config):
        """Test basic decomposition."""
        decomposer = TuckerDecomposer(sample_tensor_medium, config=decomposition_config)
        decomposer.decompose()
        
        core = decomposer.cores
        factors = decomposer.factors
        
        assert core is not None
        assert isinstance(factors, list)
        assert len(factors) == 3
        
        # Check ranks match config [20, 10, 5]
        # factors[i] shape is (Size_i, Rank_i) usually for decomposition
        # Tensorly tucker factors: [U, V, W] where U is (I, R1).
        assert factors[0].shape[1] == 20
        assert factors[1].shape[1] == 10
        assert factors[2].shape[1] == 5
    
    def test_automatic_rank_selection(self, sample_tensor_medium):
        """Test automatic rank selection."""
        # Note: New implementation requires ranks=None for auto, 
        # but validation usually forces ranks determination if not provided?
        # Let's check if passing None works
        config = DecompositionConfig(
            ranks=None,  # Auto selection
            energy_threshold=0.95,
            verbose=False
        )
        decomposer = TuckerDecomposer(sample_tensor_medium, config=config)
        decomposer.decompose()
        
        # Check that something was produced
        assert decomposer.cores is not None
        # We can't easily assert energy retention directly as it's not exposed as property 
        # except via reconstruction error?
        
    def test_reconstruction(self, sample_tensor_medium, decomposition_config):
        """Test reconstruction."""
        decomposer = TuckerDecomposer(sample_tensor_medium, config=decomposition_config)
        decomposer.decompose()
        decomposer.reconstruct()
        
        reconstructed = decomposer.reconstructed_tensors
        error = decomposer.reconstruction_errors
        
        assert reconstructed.shape == sample_tensor_medium.shape
        assert error < 1.0  # High error expected for random noise with low rank
    
    def test_invalid_tensor_dimension(self, decomposition_config):
        """Test invalid tensor dimensionality."""
        tensor_1d = torch.randn(10)
        # Init might succeed if validation is lazy, but explicit decompose should fail
        # Or init might fail.
        # Implementation calls process_tensors in init, which validates shape.
        
        with pytest.raises(Exception, match="at least"):
             TuckerDecomposer(tensor_1d, config=decomposition_config)


class TestDecompositionResult:
    """Tests for DecompositionResult."""
    
    def test_structure(self):
        """Test result structure."""
        core = torch.randn(5, 5, 5)
        factors = [torch.randn(10, 5) for _ in range(3)]
        
        res = DecompositionResult(core=core, factors=factors)
        assert res.core is core
        assert res.factors is factors
