import unittest
import torch
from algorithm.TBMD.utils.split_data import (
    split_data_in_memory_ordered,
    split_data_in_memory,
)

class TestSplitData(unittest.TestCase):
    def setUp(self):
        self.data = {
            "subject1": torch.randn(10, 10, 20),
            "subject2": torch.randn(10, 10, 10),
        }

    def test_split_data_in_memory_ordered(self):
        train_data, test_data = split_data_in_memory_ordered(
            self.data, train_ratio=0.8
        )
        self.assertIn("subject1", train_data)
        self.assertIn("subject1", test_data)
        self.assertEqual(train_data["subject1"].shape[-1], 16)
        self.assertEqual(test_data["subject1"].shape[-1], 4)
        self.assertEqual(train_data["subject2"].shape[-1], 8)
        self.assertEqual(test_data["subject2"].shape[-1], 2)

    def test_split_data_in_memory(self):
        experiments_data = split_data_in_memory(
            self.data, num_experiments=2, train_ratio=0.8
        )
        self.assertIn(1, experiments_data)
        self.assertIn(2, experiments_data)
        self.assertIn("train", experiments_data[1])
        self.assertIn("test", experiments_data[1])
        self.assertIn("subject1", experiments_data[1]["train"])
        self.assertIn("subject1", experiments_data[1]["test"])

if __name__ == "__main__":
    unittest.main()