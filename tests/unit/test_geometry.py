import unittest

import numpy as np
import scipy.sparse as sp

from TBMD.core.geometry.graph import GeometricWeightComputer, MeshGeometry, MeshGraphBuilder


class TestGeometry(unittest.TestCase):
    def test_compute_edge_distances(self):
        # Create a simple graph
        # 0 -- 1
        # |  /
        # 2
        coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

        row = [0, 1, 0, 2, 1, 2]
        col = [1, 0, 2, 0, 2, 1]
        data = [1, 1, 1, 1, 1, 1]
        A = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        builder = MeshGraphBuilder()
        distances = builder._compute_edge_distances(A, coordinates)

        # Check values
        d01 = np.linalg.norm(coordinates[0] - coordinates[1])  # 1.0
        d02 = np.linalg.norm(coordinates[0] - coordinates[2])  # 1.0
        d12 = np.linalg.norm(coordinates[1] - coordinates[2])  # sqrt(2)

        dist_dense = distances.toarray()
        self.assertAlmostEqual(dist_dense[0, 1], d01)
        self.assertAlmostEqual(dist_dense[1, 0], d01)
        self.assertAlmostEqual(dist_dense[0, 2], d02)
        self.assertAlmostEqual(dist_dense[2, 0], d02)
        self.assertAlmostEqual(dist_dense[1, 2], d12)
        self.assertAlmostEqual(dist_dense[2, 1], d12)

    def test_compute_distances_from_adjacency(self):
        coordinates = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)

        row = [0, 1]
        col = [1, 0]
        data = [1, 1]
        A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

        # Mock mesh
        mesh = MeshGeometry(
            adjacency_matrix=A,
            laplacian_matrix=A,  # Dummy
            normalized_laplacian=A,  # Dummy
            coordinates=coordinates,
        )

        computer = GeometricWeightComputer(mesh)
        distances = computer._compute_distances_from_adjacency(A)

        dist_val = 5.0  # sqrt(3^2 + 4^2)
        dist_dense = distances.toarray()
        self.assertAlmostEqual(dist_dense[0, 1], dist_val)
        self.assertAlmostEqual(dist_dense[1, 0], dist_val)


class TestKNNGraph(unittest.TestCase):
    def test_build_knn_graph(self):
        # Create a simple set of points
        # 0: (0, 0)
        # 1: (1, 0)
        # 2: (0, 1)
        # 3: (1, 1)
        coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)

        # k=1: each point should connect to its nearest neighbor
        # 0 -> 1 or 2 (dist 1)
        # 1 -> 0 or 3 (dist 1)
        # 2 -> 0 or 3 (dist 1)
        # 3 -> 1 or 2 (dist 1)

        builder = MeshGraphBuilder()
        k = 1
        A, D = builder._build_knn_graph(coordinates, k)

        # Check shapes
        self.assertEqual(A.shape, (4, 4))
        self.assertEqual(D.shape, (4, 4))

        # Check symmetry
        self.assertTrue((A - A.T).nnz == 0)
        self.assertTrue((D - D.T).nnz == 0)

        # Check that diagonal is zero (no self-loops in adjacency)
        self.assertEqual(A.diagonal().sum(), 0)

        # Check that we have connections
        self.assertTrue(A.nnz > 0)

        # Check distances
        # Distance between connected nodes should be > 0
        self.assertTrue(np.all(D.data > 0))

        # Check specific connections for k=3 (should connect to all others)
        k = 3
        A, D = builder._build_knn_graph(coordinates, k)

        # Should be fully connected (except self-loops)
        # 4 nodes, each connected to 3 others = 12 edges
        self.assertEqual(A.nnz, 12)

        # Check distance between 0 and 3 is sqrt(2)
        D_dense = D.toarray()
        self.assertAlmostEqual(D_dense[0, 3], np.sqrt(2))
        self.assertAlmostEqual(D_dense[3, 0], np.sqrt(2))


if __name__ == "__main__":
    unittest.main()


class TestGeometricWeightComputer(unittest.TestCase):
    def test_compute_proximity_penalty_caching(self):
        coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)

        # We only need coordinates for the mock mesh
        mesh = MeshGeometry(
            adjacency_matrix=sp.csr_matrix((4, 4)),
            laplacian_matrix=sp.csr_matrix((4, 4)),
            normalized_laplacian=sp.csr_matrix((4, 4)),
            coordinates=coordinates,
        )
        computer = GeometricWeightComputer(mesh)

        sensors1 = np.array([0, 1])

        # 1. Initial compute
        penalty1 = computer.compute_proximity_penalty(sensors1, 0.1)

        # 2. Same positions, same distance (cache exact match)
        penalty2 = computer.compute_proximity_penalty(sensors1, 0.1)
        np.testing.assert_array_almost_equal(penalty1, penalty2)

        # 3. Same positions, different distance (cache distance, recompute penalty)
        penalty3 = computer.compute_proximity_penalty(sensors1, 0.5)
        self.assertFalse(np.array_equal(penalty1, penalty3))

        # The true distance should be 0 for sensors 0, 1
        # distance to 2 is 1 (from 0)
        # distance to 3 is 1 (from 1)
        # penalty for distance 0 = exp(0) = 1.0
        # penalty for distance 1 with min_dist 0.5 = exp(-1.0 / 0.5) = exp(-2.0)
        expected_penalty3 = np.array([1.0, 1.0, np.exp(-1.0 / 0.5), np.exp(-1.0 / 0.5)])
        np.testing.assert_array_almost_equal(penalty3, expected_penalty3)

        # 4. Incremental addition
        sensors2 = np.array([0, 1, 2])
        penalty4 = computer.compute_proximity_penalty(sensors2, 0.5)
        expected_penalty4 = np.array([1.0, 1.0, 1.0, np.exp(-1.0 / 0.5)])
        np.testing.assert_array_almost_equal(penalty4, expected_penalty4)
