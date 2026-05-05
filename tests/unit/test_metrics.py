import unittest
import torch
import numpy as np
from TBMD.core.metrics.metrics import compute_metrics

class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        # Test with torch tensors
        tensor1 = torch.randn(10, 10)
        tensor2 = tensor1 + torch.randn(10, 10) * 0.1
        error, mse, ssim, psnr = compute_metrics(tensor1, tensor2)
        self.assertIsInstance(error, float)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(ssim, float)
        self.assertIsInstance(psnr, float)

        # Test with numpy arrays
        np_array1 = np.random.rand(10, 10)
        np_array2 = np_array1 + np.random.rand(10, 10) * 0.1
        error, mse, ssim, psnr = compute_metrics(np_array1, np_array2)
        self.assertIsInstance(error, float)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(ssim, float)
        self.assertIsInstance(psnr, float)

        # Test with background value
        tensor1_bg = torch.ones(10, 10)
        tensor2_bg = torch.ones(10, 10)
        tensor1_bg[5, 5] = 10
        tensor2_bg[5, 5] = 10
        error, mse, ssim, psnr = compute_metrics(tensor1_bg, tensor2_bg, background_value=1)
        self.assertEqual(error, 0.0)
        self.assertEqual(mse, 0.0)
        self.assertEqual(ssim, 1.0)
        self.assertTrue(np.isinf(psnr))



    def test_compute_metrics_constant_arrays(self):
        # Test with identical constant arrays (e.g. all tens) to ensure no division-by-zero
        # and PSNR is handled correctly (infinity for 0 MSE).
        tens1 = np.full((5, 5), 10.0)
        tens2 = np.full((5, 5), 10.0)

        # When comparing identical arrays, err_norm should be 0.0, mse 0.0,
        # ssim 1.0, and psnr should be infinity because mse is 0.
        error, mse, ssim, psnr = compute_metrics(tens1, tens2)

        self.assertEqual(error, 0.0)
        self.assertEqual(mse, 0.0)
        self.assertEqual(ssim, 1.0)
        self.assertEqual(psnr, float('inf'))

if __name__ == "__main__":
    unittest.main()