import unittest
import torch
import numpy as np
from algorithm.TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import (
    TensorTubeQRDecomposition,
    UniformDistributionManager,
    TensorQRConfig
)

class TestTensorBasedTubeFiberPivotQRFactorization(unittest.TestCase):
    def setUp(self):
        # Create a tensor with repeating patterns
        # 10x10 spatial, 5 temporal, 10 feature/tube dimension
        self.tensor = torch.randn(10, 10, 5, 10)

        # Create a structured tensor for testing uniform distribution
        self.structured_tensor = torch.zeros(10, 10, 5, 10)
        # Region 1: first 5x5
        self.structured_tensor[0:5, 0:5, :, :] = 1.0
        # Region 2: rest
        self.structured_tensor[5:, 5:, :, :] = 2.0

        self.qr_decomposer = TensorTubeQRDecomposition(
            tensor=self.tensor,
            N=5,
        )

        self.config = TensorQRConfig()
        self.device = torch.device('cpu')

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

    def test_uniform_distribution_manager(self):
        """Test the UniformDistributionManager optimizations."""
        spatial_shape = (10, 10, 5)
        manager = UniformDistributionManager(self.config, spatial_shape, self.device)
        manager.identify_similar_regions(self.structured_tensor)

        # Check if region_ids is populated
        self.assertIsNotNone(manager.region_ids)
        self.assertEqual(manager.region_ids.shape, (10, 10))

        # Check number of unique regions (should be > 1)
        self.assertGreater(len(manager.similar_regions), 1)

        # Verify coordinates in similar_regions are correct
        # One region should have coordinates from (0,0) to (4,4)
        found_region_1 = False
        for pid, coords in manager.similar_regions.items():
            # Check if (0,0) is in this region
            mask = (coords[:, 0] == 0) & (coords[:, 1] == 0)
            if mask.any():
                found_region_1 = True
                # Verify other points: (0, 1) should also be there
                mask2 = (coords[:, 0] == 0) & (coords[:, 1] == 1)
                self.assertTrue(mask2.any())

        self.assertTrue(found_region_1)

        # Test blocking logic
        available = torch.ones(spatial_shape, dtype=torch.bool)
        pivot = (0, 0, 0)
        manager.mark_similar_regions_unavailable(pivot, available)

        # (0, 0, 0) should remain available (logic excludes self from blocking)
        self.assertTrue(available[0, 0, 0])

        # Check if neighbors are blocked (limit depends on region size)
        # Count blocked neighbors in slice 0
        blocked_count = 0
        region_coords = manager.similar_regions[manager.region_ids[0, 0].item()]
        for row in region_coords:
            px, py = row[0].item(), row[1].item()
            # Skip self
            if px == 0 and py == 0:
                continue
            if not available[px, py, 0]:
                blocked_count += 1

        # Should block some but not all (max 25%)
        # Region size is 25. 25/4 = 6. So at most 6 blocked.
        self.assertGreaterEqual(blocked_count, 0)
        self.assertLessEqual(blocked_count, max(1, len(region_coords)//4))

    def test_full_decomposition_with_uniform_distribution(self):
        # Initialize with uniform_distribution=True
        decomposer = TensorTubeQRDecomposition(
            tensor=self.structured_tensor,
            N=5,
            uniform_distribution=True
        )
        P, Q, R = decomposer.factorize()
        self.assertIsNotNone(P)

if __name__ == "__main__":
    unittest.main()
