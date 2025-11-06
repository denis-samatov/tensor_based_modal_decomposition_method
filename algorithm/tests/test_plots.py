import unittest
import numpy as np
import torch
from algorithm.TBMD.utils.plots import visualize_tensor

class TestPlots(unittest.TestCase):

    def test_visualize_tensor_torch(self):
        tensor = torch.rand((10, 10, 10))
        try:
            visualize_tensor(tensor)
        except Exception as e:
            self.fail(f"visualize_tensor raised an exception: {e}")

    def test_visualize_tensor_numpy(self):
        tensor = np.random.rand(10, 10, 10)
        try:
            visualize_tensor(tensor)
        except Exception as e:
            self.fail(f"visualize_tensor raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
