import unittest
import torch
import numpy as np
from TBMD.core.data.processors import (
    process_data,
    calculate_global_minmax_params,
    calculate_global_zscore_params,
    inverse_normalization,
)

class TestProcessData(unittest.TestCase):
    def setUp(self):
        self.train_data = {
            "subject1": torch.randn(10, 10, 5) * 100,
            "subject2": torch.randn(10, 10, 5) * 50,
        }
        self.test_data = {
            "subject3": torch.randn(10, 10, 5) * 75,
        }

    def test_calculate_global_minmax_params(self):
        min_val, max_val = calculate_global_minmax_params(self.train_data)
        self.assertIsInstance(min_val, float)
        self.assertIsInstance(max_val, float)

    def test_calculate_global_zscore_params(self):
        mean, std = calculate_global_zscore_params(self.train_data)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)

    def test_process_data_minmax(self):
        min_val, max_val = calculate_global_minmax_params(self.train_data)
        params = {"min": min_val, "max": max_val}
        processed_data = process_data(
            self.train_data,
            normalization_method="minmax",
            global_params=params,
        )
        for subject, tensor in processed_data.items():
            tensor = torch.from_numpy(tensor)
            self.assertLessEqual(torch.max(tensor), 1.0)
            self.assertGreaterEqual(torch.min(tensor), 0.0)

    def test_process_data_zscore(self):
        mean, std = calculate_global_zscore_params(self.train_data)
        params = {"mean": mean, "std": std}
        processed_data = process_data(
            self.train_data,
            normalization_method="zscore",
            global_params=params,
        )
        for subject, tensor in processed_data.items():
            # Z-score normalization doesn't guarantee a specific range,
            # but we can check if it ran without errors.
            self.assertEqual(tensor.shape, self.train_data[subject].shape)

    def test_inverse_normalization_minmax(self):
        min_val, max_val = calculate_global_minmax_params(self.train_data)
        params = {"min": min_val, "max": max_val}
        processed_data = process_data(
            self.train_data,
            normalization_method="minmax",
            global_params=params,
        )
        inverted_data = inverse_normalization(
            processed_data["subject1"],
            normalization_method="minmax",
            global_params=params,
        )
        inverted_data = torch.from_numpy(inverted_data)
        self.assertTrue(
            torch.allclose(inverted_data, self.train_data["subject1"], atol=1e-4)
        )

    def test_inverse_normalization_zscore(self):
        mean, std = calculate_global_zscore_params(self.train_data)
        params = {"mean": mean, "std": std}
        processed_data = process_data(
            self.train_data,
            normalization_method="zscore",
            global_params=params,
        )
        inverted_data = inverse_normalization(
            processed_data["subject1"],
            normalization_method="zscore",
            global_params=params,
        )
        inverted_data = torch.from_numpy(inverted_data)
        self.assertTrue(
            torch.allclose(inverted_data, self.train_data["subject1"], atol=1e-5)
        )

if __name__ == "__main__":
    unittest.main()