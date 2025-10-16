import unittest
import torch
import numpy as np
from algorithm.TBMD.modules.TensorHOSVD import TuckerDecomposer

class TestTensorHOSVD(unittest.TestCase):
    def setUp(self):
        self.tensors = {
            "subject1": torch.randn(10, 10, 5, 10),
            "subject2": torch.randn(10, 10, 5, 10),
        }
        self.decomposer = TuckerDecomposer(
            tensors=self.tensors,
            ranks=[5, 5, 3, 5],
        )

    def test_decompose(self):
        self.decomposer.decompose()
        self.assertIsNotNone(self.decomposer.cores)
        self.assertIsNotNone(self.decomposer.factors)
        self.assertIn("subject1", self.decomposer.cores)
        self.assertIn("subject1", self.decomposer.factors)
        self.assertEqual(
            self.decomposer.cores["subject1"].shape,
            (5, 5, 3, 5)
        )
        self.assertEqual(len(self.decomposer.factors["subject1"]), 4)

    def test_reconstruct(self):
        self.decomposer.decompose()
        self.decomposer.reconstruct()
        self.assertIsNotNone(self.decomposer.reconstructed_tensors)
        self.assertIn("subject1", self.decomposer.reconstructed_tensors)
        self.assertEqual(
            self.decomposer.reconstructed_tensors["subject1"].shape,
            self.tensors["subject1"].shape,
        )
        self.assertIn("subject1", self.decomposer.reconstruction_errors)

if __name__ == "__main__":
    unittest.main()