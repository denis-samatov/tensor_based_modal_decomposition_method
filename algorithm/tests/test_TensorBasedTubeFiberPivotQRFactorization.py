import unittest
import torch
from algorithm.TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import (
    TensorTubeQRDecomposition,
)

class TestTensorBasedTubeFiberPivotQRFactorization(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.randn(10, 10, 5, 10)
        self.qr_decomposer = TensorTubeQRDecomposition(
            tensor=self.tensor,
            N=5,
        )

    def test_factorize(self):
        P, Q, R = self.qr_decomposer.factorize()
        self.assertIsNotNone(P)
        self.assertIsNotNone(Q)
        self.assertIsNotNone(R)
        self.assertEqual(P.shape, (10, 10, 5))
        self.assertEqual(Q.shape, (10, 10))
        self.assertEqual(R.shape, (10, 10, 5, 10))

    def test_check_factorization(self):
        self.qr_decomposer.factorize()
        is_valid, error, metrics = self.qr_decomposer.check_factorization()
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(error, float)
        self.assertIn("orthogonality_deviation", metrics)

if __name__ == "__main__":
    unittest.main()