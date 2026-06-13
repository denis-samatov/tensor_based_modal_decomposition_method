import unittest

import numpy as np
import torch

from TBMD.core.utils.misc import (
    build_wells_matrix,
    build_Y_matrices,
    get_torch_device,
    reconstruct_tensor,
    to_torch_tensor,
)


class TestUtils(unittest.TestCase):
    def test_to_torch_tensor(self):
        # Test with numpy array
        np_array = np.random.rand(3, 3)
        torch_tensor = to_torch_tensor(np_array, device="cpu")
        self.assertTrue(torch.is_tensor(torch_tensor))
        self.assertEqual(torch_tensor.shape, (3, 3))

        # Test with torch tensor
        torch_tensor_in = torch.randn(2, 2)
        torch_tensor_out = to_torch_tensor(torch_tensor_in, device="cpu")
        self.assertTrue(torch.equal(torch_tensor_in, torch_tensor_out))

        # Test with list
        list_in = [[1, 2], [3, 4]]
        torch_tensor = to_torch_tensor(list_in, device="cpu")
        self.assertTrue(torch.is_tensor(torch_tensor))
        self.assertEqual(torch_tensor.shape, (2, 2))

    def test_get_torch_device(self):
        # Test with 'cpu'
        device = get_torch_device("cpu")
        self.assertEqual(device.type, "cpu")

        # Test with 'cuda' if available
        if torch.cuda.is_available():
            device = get_torch_device("cuda")
            self.assertEqual(device.type, "cuda")

    def test_reconstruct_tensor(self):
        A_tensor = torch.randn(10, 5, 3)
        x_hat = torch.randn(3)
        reconstructed = reconstruct_tensor(A_tensor, x_hat)
        self.assertEqual(reconstructed.shape, (10, 5))

    def test_build_Y_matrices(self):
        test_tensors = {"subject1": torch.randn(10, 10, 5)}
        P = torch.zeros(10, 10, dtype=torch.bool)
        P[2:4, 2:4] = True
        Y_mats = build_Y_matrices(test_tensors, P)
        self.assertIn("subject1", Y_mats)
        self.assertEqual(Y_mats["subject1"].shape, (10, 10, 5))
        self.assertTrue(torch.all(Y_mats["subject1"][~P] == 0))
        self.assertTrue(torch.all(Y_mats["subject1"][P] != 0))

    def test_build_wells_matrix(self):
        wells = {"subject1": [[1, 1], [2, 2]]}
        grid_shape = (5, 5, 2)
        wells_matrix = build_wells_matrix(wells, grid_shape)
        self.assertIn("subject1", wells_matrix)
        self.assertEqual(wells_matrix["subject1"].shape, (5, 5))
        self.assertEqual(wells_matrix["subject1"][1, 1], 1)
        self.assertEqual(wells_matrix["subject1"][2, 2], 1)
        self.assertEqual(torch.sum(wells_matrix["subject1"]), 2)


if __name__ == "__main__":
    unittest.main()
