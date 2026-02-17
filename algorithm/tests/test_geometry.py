import unittest
import numpy as np
import scipy.sparse as sp
from algorithm.TBMD.utils.geometry import MeshGraphBuilder, GeometricWeightComputer, MeshGeometry

class TestGeometry(unittest.TestCase):
    def test_compute_edge_distances(self):
        # Create a simple graph
        # 0 -- 1
        # |  /
        # 2
        coordinates = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        row = [0, 1, 0, 2, 1, 2]
        col = [1, 0, 2, 0, 2, 1]
        data = [1, 1, 1, 1, 1, 1]
        A = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        builder = MeshGraphBuilder()
        distances = builder._compute_edge_distances(A, coordinates)

        # Check values
        d01 = np.linalg.norm(coordinates[0] - coordinates[1]) # 1.0
        d02 = np.linalg.norm(coordinates[0] - coordinates[2]) # 1.0
        d12 = np.linalg.norm(coordinates[1] - coordinates[2]) # sqrt(2)

        dist_dense = distances.toarray()
        self.assertAlmostEqual(dist_dense[0, 1], d01)
        self.assertAlmostEqual(dist_dense[1, 0], d01)
        self.assertAlmostEqual(dist_dense[0, 2], d02)
        self.assertAlmostEqual(dist_dense[2, 0], d02)
        self.assertAlmostEqual(dist_dense[1, 2], d12)
        self.assertAlmostEqual(dist_dense[2, 1], d12)

    def test_compute_distances_from_adjacency(self):
        coordinates = np.array([
            [0, 0, 0],
            [3, 4, 0]
        ], dtype=float)

        row = [0, 1]
        col = [1, 0]
        data = [1, 1]
        A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

        # Mock mesh
        mesh = MeshGeometry(
            adjacency_matrix=A,
            laplacian_matrix=A, # Dummy
            normalized_laplacian=A, # Dummy
            coordinates=coordinates
        )

        computer = GeometricWeightComputer(mesh)
        distances = computer._compute_distances_from_adjacency(A)

        dist_val = 5.0 # sqrt(3^2 + 4^2)
        dist_dense = distances.toarray()
        self.assertAlmostEqual(dist_dense[0, 1], dist_val)
        self.assertAlmostEqual(dist_dense[1, 0], dist_val)

if __name__ == '__main__':
    unittest.main()
