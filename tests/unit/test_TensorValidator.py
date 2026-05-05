import unittest
import torch
from TBMD.core.decomposition.hosvd import TensorValidator, ValidationError, InvalidRankError

class TestTensorValidator(unittest.TestCase):
    """Test cases for the TensorValidator class."""

    def test_validate_tensor_shape(self):
        """Test validate_tensor_shape method."""
        # Valid 3D tensor
        tensor_3d = torch.randn(2, 3, 4)
        TensorValidator.validate_tensor_shape(tensor_3d, min_dims=2)  # Should not raise

        # Valid 2D tensor with min_dims=2
        tensor_2d = torch.randn(2, 3)
        TensorValidator.validate_tensor_shape(tensor_2d, min_dims=2)  # Should not raise

        # Invalid: too few dimensions
        tensor_1d = torch.randn(5)
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_tensor_shape(tensor_1d, min_dims=2)
        self.assertIn("at least 2 dimensions", str(cm.exception))

        # Invalid: non-positive dimension size (0)
        # torch.randn doesn't allow 0, but we can create one with torch.empty or similar if needed,
        # but let's just check the logic with a mocked-like shape if possible,
        # though validate_tensor_shape expects a torch.Tensor.
        # We can create a tensor with a zero dimension.
        tensor_zero_dim = torch.empty(2, 0, 4)
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_tensor_shape(tensor_zero_dim)
        self.assertIn("must be positive", str(cm.exception))

    def test_validate_ranks(self):
        """Test validate_ranks method."""
        shape = (10, 8, 6)

        # Ranks is None -> automatic selection
        ranks = TensorValidator.validate_ranks(None, shape)
        self.assertEqual(ranks, [6, 6, 6])

        # Ranks is None, empty shape
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_ranks(None, ())
        self.assertIn("empty tensor shape", str(cm.exception))

        # Ranks is int
        ranks = TensorValidator.validate_ranks(4, shape)
        self.assertEqual(ranks, [4, 4, 4])

        # Ranks is int, too small
        with self.assertRaises(InvalidRankError) as cm:
            TensorValidator.validate_ranks(0, shape)
        self.assertIn("must be >= 1", str(cm.exception))

        # Ranks is int, too large
        with self.assertRaises(InvalidRankError) as cm:
            TensorValidator.validate_ranks(11, shape)
        self.assertIn("exceeds minimum tensor dimension", str(cm.exception))

        # Ranks is list
        ranks_list = [5, 4, 3]
        ranks = TensorValidator.validate_ranks(ranks_list, shape)
        self.assertEqual(ranks, ranks_list)

        # Ranks is list, length mismatch
        with self.assertRaises(InvalidRankError) as cm:
            TensorValidator.validate_ranks([5, 4], shape)
        self.assertIn("length 2 must match tensor modes 3", str(cm.exception))

        # Ranks is list, value too small
        with self.assertRaises(InvalidRankError) as cm:
            TensorValidator.validate_ranks([5, 0, 3], shape)
        self.assertIn("at position 1 must be >= 1", str(cm.exception))

        # Ranks is list, value too large
        with self.assertRaises(InvalidRankError) as cm:
            TensorValidator.validate_ranks([5, 4, 10], shape)
        self.assertIn("at position 2 exceeds dimension 6", str(cm.exception))

        # Invalid ranks type
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_ranks("invalid", shape)
        self.assertIn("must be None, int, or list of ints", str(cm.exception))

    def test_validate_epsilon(self):
        """Test validate_epsilon method."""
        # Valid float
        self.assertEqual(TensorValidator.validate_epsilon(1e-2), 0.01)

        # Valid int
        self.assertEqual(TensorValidator.validate_epsilon(1), 1.0)

        # Invalid: non-numeric type
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_epsilon("0.01")
        self.assertIn("must be numeric", str(cm.exception))

        # Invalid: zero
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_epsilon(0)
        self.assertIn("must be positive", str(cm.exception))

        # Invalid: negative
        with self.assertRaises(ValidationError) as cm:
            TensorValidator.validate_epsilon(-0.5)
        self.assertIn("must be positive", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
