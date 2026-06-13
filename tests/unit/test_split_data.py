import unittest

try:
    import numpy as np
    import torch

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

if DEPS_AVAILABLE:
    from TBMD.core.data.splitters import (
        split_data_in_memory,
        split_data_in_memory_ordered,
    )


@unittest.skipUnless(DEPS_AVAILABLE, "Dependencies (numpy, torch) are required for this test")
class TestSplitData(unittest.TestCase):
    def setUp(self):
        self.data = {
            "subject1": torch.randn(10, 10, 20),
            "subject2": torch.randn(10, 10, 10),
        }
        self.data_with_empty = {
            "subject1": torch.randn(10, 10, 20),
            "subject_empty": torch.empty(10, 10, 0),
            "subject_none": None,
        }

    def test_split_data_in_memory_ordered(self):
        train_data, test_data = split_data_in_memory_ordered(self.data, train_ratio=0.8)
        self.assertIn("subject1", train_data)
        self.assertIn("subject1", test_data)
        self.assertEqual(train_data["subject1"].shape[-1], 16)
        self.assertEqual(test_data["subject1"].shape[-1], 4)
        self.assertEqual(train_data["subject2"].shape[-1], 8)
        self.assertEqual(test_data["subject2"].shape[-1], 2)

    def test_split_data_in_memory(self):
        experiments_data = split_data_in_memory(self.data, num_experiments=2, train_ratio=0.8)
        self.assertIn(1, experiments_data)
        self.assertIn(2, experiments_data)
        self.assertIn("train", experiments_data[1])
        self.assertIn("test", experiments_data[1])
        self.assertIn("subject1", experiments_data[1]["train"])
        self.assertIn("subject1", experiments_data[1]["test"])

    def test_split_data_in_memory_empty_subject(self):
        experiments_data = split_data_in_memory(
            self.data_with_empty, num_experiments=1, train_ratio=0.8
        )
        self.assertIn(1, experiments_data)
        self.assertIn("train", experiments_data[1])
        self.assertIn("test", experiments_data[1])

        # subject1 should be processed normally
        self.assertIn("subject1", experiments_data[1]["train"])

        # Empty subjects and None subjects should be gracefully skipped
        self.assertNotIn("subject_empty", experiments_data[1]["train"])
        self.assertNotIn("subject_none", experiments_data[1]["train"])
        self.assertNotIn("subject_empty", experiments_data[1]["test"])
        self.assertNotIn("subject_none", experiments_data[1]["test"])

    def test_split_data_in_memory_ordered_empty_subject(self):
        train_data, test_data = split_data_in_memory_ordered(self.data_with_empty, train_ratio=0.8)

        # subject1 should be processed normally
        self.assertIn("subject1", train_data)

        # Empty subjects and None subjects should be gracefully skipped
        # The function default returns None for absent keys if it's a defaultdict,
        # but we should just verify the skipped keys aren't in the resulting dict explicitly set
        self.assertIsNone(train_data.get("subject_empty"))
        self.assertIsNone(train_data.get("subject_none"))


if __name__ == "__main__":
    unittest.main()
