import unittest
import torch
from TBMD.core.reconstruction.tensor_compressive_sensing import (
    TensorCompressiveSensing,
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig,
)

class TestTensorBasedCompressiveSensing(unittest.TestCase):
    def setUp(self):
        self.A_tensor = torch.randn(10, 10, 5)
        self.P = torch.zeros(10, 10, dtype=torch.bool)
        self.P[2:4, 2:4] = True
        self.Y = torch.randn(10, 10)
        self.config = CompressiveSensingConfig()
        self.ext_config = ExtensionCompressiveSensingConfig()
        self.solver = TensorCompressiveSensing(
            self.A_tensor, self.P, self.Y, self.config, self.ext_config
        )

    def test_solve(self):
        x_hat, met = self.solver.solve()
        self.assertIsNotNone(x_hat)
        self.assertIsNotNone(met)
        self.assertEqual(x_hat.shape, (5,))
        self.assertIsInstance(met.converged, bool)
        self.assertIsInstance(met.iterations, int)

    def test_reconstruction_error(self):
        x_hat, _ = self.solver.solve()
        error = self.solver.reconstruction_error(x_hat)
        self.assertIsInstance(error, float)

if __name__ == "__main__":
    unittest.main()