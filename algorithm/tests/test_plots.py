import unittest
from unittest.mock import patch

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Try to import the functions to test
try:
    from algorithm.TBMD.utils.plots import visualize_tensor, normalize_for_rgb_display
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

class TestPlots(unittest.TestCase):

    @unittest.skipUnless(HAS_TORCH and HAS_PLOTS and HAS_MATPLOTLIB, "Requires torch, plots and matplotlib")
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_tensor_torch(self, mock_savefig, mock_show):
        tensor = torch.rand((10, 10, 10))
        try:
            visualize_tensor(tensor)
        except Exception as e:
            self.fail(f"visualize_tensor raised an exception: {e}")

    @unittest.skipUnless(HAS_NUMPY and HAS_PLOTS and HAS_MATPLOTLIB, "Requires numpy, plots and matplotlib")
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_tensor_numpy(self, mock_savefig, mock_show):
        tensor = np.random.rand(10, 10, 10)
        try:
            visualize_tensor(tensor)
        except Exception as e:
            self.fail(f"visualize_tensor raised an exception: {e}")

    @unittest.skipUnless(HAS_NUMPY and HAS_PLOTS, "Requires numpy and plots")
    def test_normalize_for_rgb_display_numpy(self):
        # Normal case
        data = np.array([0, 0.5, 1.0])
        normalized = normalize_for_rgb_display(data)
        np.testing.assert_array_almost_equal(normalized, [0, 0.5, 1.0])
        self.assertIsInstance(normalized, np.ndarray)

        # Scaling case
        data = np.array([-1.0, 0.0, 1.0])
        normalized = normalize_for_rgb_display(data)
        np.testing.assert_array_almost_equal(normalized, [0, 0.5, 1.0])

        # Range > 1
        data = np.array([0, 10, 20])
        normalized = normalize_for_rgb_display(data)
        np.testing.assert_array_almost_equal(normalized, [0, 0.5, 1.0])

    @unittest.skipUnless(HAS_TORCH and HAS_NUMPY and HAS_PLOTS, "Requires torch, numpy and plots")
    def test_normalize_for_rgb_display_torch(self):
        # Torch tensor input
        data = torch.tensor([0.0, 5.0, 10.0])
        normalized = normalize_for_rgb_display(data)
        self.assertIsInstance(normalized, np.ndarray)
        np.testing.assert_array_almost_equal(normalized, [0, 0.5, 1.0])

    @unittest.skipUnless(HAS_NUMPY and HAS_PLOTS, "Requires numpy and plots")
    def test_normalize_for_rgb_display_constant(self):
        # Constant values should return zeros (as per implementation)
        data = np.ones((5, 5))
        normalized = normalize_for_rgb_display(data)
        np.testing.assert_array_equal(normalized, np.zeros((5, 5)))

        data = np.zeros((5, 5))
        normalized = normalize_for_rgb_display(data)
        np.testing.assert_array_equal(normalized, np.zeros((5, 5)))

    @unittest.skipUnless(HAS_NUMPY and HAS_PLOTS, "Requires numpy and plots")
    def test_normalize_for_rgb_display_clipping(self):
        # Implementation has np.clip(array, 0, 1) at the end
        data = np.array([0, 0.5, 1.0])
        normalized = normalize_for_rgb_display(data)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))

if __name__ == '__main__':
    unittest.main()
