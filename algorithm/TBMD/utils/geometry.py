"""
Geometry-aware utilities for TBMD on unstructured meshes.

This module provides:
1. Cell adjacency graph construction for unstructured meshes
2. Laplacian matrix computation (graph Laplacian for spatial smoothness)
3. Geometric distance and gradient estimation
4. Utilities for geometry-aware sensor placement

Key concepts:
- Graph Laplacian L = D - A, where D is degree matrix and A is adjacency matrix
- Normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
- Used to impose smoothness regularization on spatial modes in HOSVD
"""

import numpy as np
import torch
import scipy.sparse as sp
from typing import Union, Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.spatial import KDTree, Delaunay
import warnings

from ..utils.utils import to_torch_tensor, get_torch_device


@dataclass
class MeshGeometry:
    """A container for mesh geometry information.

    Attributes:
        adjacency_matrix (sp.spmatrix): A sparse adjacency matrix representing
            cell connectivity.
        laplacian_matrix (sp.spmatrix): The graph Laplacian matrix L = D - A.
        normalized_laplacian (sp.spmatrix): The normalized Laplacian L_sym = I
            - D^{-1/2} A D^{-1/2}.
        coordinates (np.ndarray): The cell center coordinates.
        distances (Optional[sp.spmatrix]): The pairwise distances between
            adjacent cells.
        gradient_weights (Optional[sp.spmatrix]): The edge weights based on
            gradient magnitude estimation.
    """
    adjacency_matrix: sp.spmatrix
    laplacian_matrix: sp.spmatrix
    normalized_laplacian: sp.spmatrix
    coordinates: np.ndarray
    distances: Optional[sp.spmatrix] = None
    gradient_weights: Optional[sp.spmatrix] = None
    
    def to_torch(self, device: str = 'cpu', dtype: torch.dtype = torch.float32) -> 'TorchMeshGeometry':
        """Converts sparse matrices to PyTorch sparse tensors.

        Args:
            device (str): The device to move the tensors to. Defaults to 'cpu'.
            dtype (torch.dtype): The desired data type of the tensors. Defaults
                to torch.float32.

        Returns:
            TorchMeshGeometry: A new object with PyTorch sparse tensors.
        """
        dev = get_torch_device(device)
        
        def sparse_to_torch(mat):
            """Convert scipy sparse matrix to torch sparse tensor."""
            if mat is None:
                return None
            coo = mat.tocoo()
            indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
            values = torch.FloatTensor(coo.data).to(dtype)
            shape = coo.shape
            return torch.sparse_coo_tensor(indices, values, shape, device=dev)
        
        return TorchMeshGeometry(
            adjacency_matrix=sparse_to_torch(self.adjacency_matrix),
            laplacian_matrix=sparse_to_torch(self.laplacian_matrix),
            normalized_laplacian=sparse_to_torch(self.normalized_laplacian),
            coordinates=torch.from_numpy(self.coordinates).to(device=dev, dtype=dtype),
            distances=sparse_to_torch(self.distances),
            gradient_weights=sparse_to_torch(self.gradient_weights)
        )


@dataclass
class TorchMeshGeometry:
    """A PyTorch version of MeshGeometry with sparse tensors.

    Attributes:
        adjacency_matrix (torch.Tensor): A sparse adjacency matrix.
        laplacian_matrix (torch.Tensor): The graph Laplacian matrix.
        normalized_laplacian (torch.Tensor): The normalized Laplacian.
        coordinates (torch.Tensor): The cell center coordinates.
        distances (Optional[torch.Tensor]): The pairwise distances between
            adjacent cells.
        gradient_weights (Optional[torch.Tensor]): The edge weights based on
            gradient magnitude estimation.
    """
    adjacency_matrix: torch.Tensor
    laplacian_matrix: torch.Tensor
    normalized_laplacian: torch.Tensor
    coordinates: torch.Tensor
    distances: Optional[torch.Tensor] = None
    gradient_weights: Optional[torch.Tensor] = None


class MeshGraphBuilder:
    """Builds cell adjacency graphs for structured and unstructured meshes.

    This class supports multiple connectivity strategies:
    - 'grid': Regular grid with 4-connectivity (2D) or 6-connectivity (3D).
    - 'knn': K-nearest neighbors based on cell centers.
    - 'radius': Connects cells within a distance threshold.
    - 'delaunay': Delaunay triangulation-based connectivity (2D/3D).
    """
    
    def __init__(self, connectivity_type: str = 'knn', **kwargs):
        """Initializes the MeshGraphBuilder.

        Args:
            connectivity_type (str): The type of connectivity to build. Can be
                'grid', 'knn', 'radius', or 'delaunay'. Defaults to 'knn'.
            **kwargs: Additional parameters for the connectivity type (e.g.,
                `k` for 'knn', `radius` for 'radius').
        """
        self.connectivity_type = connectivity_type
        self.params = kwargs
        
    def build_from_shape(self, spatial_shape: Tuple[int, ...]) -> MeshGeometry:
        """Builds a graph for a regular grid mesh.

        Args:
            spatial_shape (Tuple[int, ...]): The shape of the grid (H, W) or
                (H, W, D).

        Returns:
            MeshGeometry: The constructed mesh geometry.
        """
        if len(spatial_shape) == 2:
            return self._build_2d_grid(spatial_shape)
        elif len(spatial_shape) == 3:
            return self._build_3d_grid(spatial_shape)
        else:
            raise ValueError(f"Unsupported spatial dimension: {len(spatial_shape)}")
    
    def build_from_coordinates(self, coordinates: np.ndarray) -> MeshGeometry:
        """Builds a graph from cell center coordinates (unstructured mesh).

        Args:
            coordinates (np.ndarray): The cell center coordinates, with shape
                (N_cells, spatial_dim).

        Returns:
            MeshGeometry: The constructed mesh geometry.
        """
        N = len(coordinates)
        
        if self.connectivity_type == 'knn':
            k = self.params.get('k', 6)
            A, distances = self._build_knn_graph(coordinates, k)
        elif self.connectivity_type == 'radius':
            radius = self.params.get('radius')
            if radius is None:
                raise ValueError("'radius' parameter required for radius connectivity")
            A, distances = self._build_radius_graph(coordinates, radius)
        elif self.connectivity_type == 'delaunay':
            A, distances = self._build_delaunay_graph(coordinates)
        else:
            raise ValueError(f"Unknown connectivity type for unstructured mesh: {self.connectivity_type}")
        
        # Compute Laplacian matrices
        L = self._compute_laplacian(A)
        L_norm = self._compute_normalized_laplacian(A)
        
        return MeshGeometry(
            adjacency_matrix=A,
            laplacian_matrix=L,
            normalized_laplacian=L_norm,
            coordinates=coordinates,
            distances=distances
        )
    
    def _build_2d_grid(self, shape: Tuple[int, int]) -> MeshGeometry:
        """Build 4-connected grid graph for 2D."""
        H, W = shape
        N = H * W
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        coordinates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
        
        # Build adjacency (4-connectivity)
        row, col, data = [], [], []
        
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                # Right neighbor
                if j < W - 1:
                    neighbor = i * W + (j + 1)
                    row.extend([idx, neighbor])
                    col.extend([neighbor, idx])
                    data.extend([1.0, 1.0])
                # Bottom neighbor
                if i < H - 1:
                    neighbor = (i + 1) * W + j
                    row.extend([idx, neighbor])
                    col.extend([neighbor, idx])
                    data.extend([1.0, 1.0])
        
        A = sp.csr_matrix((data, (row, col)), shape=(N, N))
        
        # Compute distances
        distances = self._compute_edge_distances(A, coordinates)
        
        # Laplacians
        L = self._compute_laplacian(A)
        L_norm = self._compute_normalized_laplacian(A)
        
        return MeshGeometry(
            adjacency_matrix=A,
            laplacian_matrix=L,
            normalized_laplacian=L_norm,
            coordinates=coordinates,
            distances=distances
        )
    
    def _build_3d_grid(self, shape: Tuple[int, int, int]) -> MeshGeometry:
        """Build 6-connected grid graph for 3D."""
        H, W, D = shape
        N = H * W * D
        
        # Create coordinate grid
        z_coords, y_coords, x_coords = np.mgrid[0:H, 0:W, 0:D]
        coordinates = np.stack([
            x_coords.ravel(),
            y_coords.ravel(),
            z_coords.ravel()
        ], axis=1)
        
        # Build adjacency (6-connectivity)
        row, col, data = [], [], []
        
        for i in range(H):
            for j in range(W):
                for k in range(D):
                    idx = i * W * D + j * D + k
                    
                    # Right neighbor (x+1)
                    if j < W - 1:
                        neighbor = i * W * D + (j + 1) * D + k
                        row.extend([idx, neighbor])
                        col.extend([neighbor, idx])
                        data.extend([1.0, 1.0])
                    
                    # Bottom neighbor (y+1)
                    if i < H - 1:
                        neighbor = (i + 1) * W * D + j * D + k
                        row.extend([idx, neighbor])
                        col.extend([neighbor, idx])
                        data.extend([1.0, 1.0])
                    
                    # Forward neighbor (z+1)
                    if k < D - 1:
                        neighbor = i * W * D + j * D + (k + 1)
                        row.extend([idx, neighbor])
                        col.extend([neighbor, idx])
                        data.extend([1.0, 1.0])
        
        A = sp.csr_matrix((data, (row, col)), shape=(N, N))
        
        # Compute distances
        distances = self._compute_edge_distances(A, coordinates)
        
        # Laplacians
        L = self._compute_laplacian(A)
        L_norm = self._compute_normalized_laplacian(A)
        
        return MeshGeometry(
            adjacency_matrix=A,
            laplacian_matrix=L,
            normalized_laplacian=L_norm,
            coordinates=coordinates,
            distances=distances
        )
    
    def _build_knn_graph(self, coordinates: np.ndarray, k: int) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Build k-nearest neighbors graph."""
        N = len(coordinates)
        tree = KDTree(coordinates)
        
        # Query k+1 neighbors (including self)
        distances, indices = tree.query(coordinates, k=min(k+1, N))
        
        row, col, data, dist_data = [], [], [], []
        
        for i in range(N):
            for j, neighbor_idx in enumerate(indices[i]):
                if neighbor_idx != i:  # Exclude self-loops
                    row.append(i)
                    col.append(neighbor_idx)
                    data.append(1.0)
                    dist_data.append(distances[i, j])
        
        A = sp.csr_matrix((data, (row, col)), shape=(N, N))
        D = sp.csr_matrix((dist_data, (row, col)), shape=(N, N))
        
        # Symmetrize (undirected graph)
        A = (A + A.T) / 2
        A.data = np.ones_like(A.data)  # Binary adjacency
        
        D = (D + D.T) / 2
        
        return A, D
    
    def _build_radius_graph(self, coordinates: np.ndarray, radius: float) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Build radius-based graph."""
        N = len(coordinates)
        tree = KDTree(coordinates)
        
        # Query all neighbors within radius
        pairs = tree.query_pairs(radius, output_type='ndarray')
        
        if len(pairs) == 0:
            row = np.array([], dtype=int)
            col = np.array([], dtype=int)
            data = np.array([], dtype=float)
            dist_data = np.array([], dtype=float)
        else:
            # Vectorized implementation for speed
            i_indices = pairs[:, 0]
            j_indices = pairs[:, 1]

            diff = coordinates[i_indices] - coordinates[j_indices]
            dists = np.linalg.norm(diff, axis=1)

            # We need both directions: (i, j) and (j, i)
            row = np.concatenate([i_indices, j_indices])
            col = np.concatenate([j_indices, i_indices])
            data = np.ones(len(row))
            dist_data = np.concatenate([dists, dists])
        
        A = sp.csr_matrix((data, (row, col)), shape=(N, N))
        D = sp.csr_matrix((dist_data, (row, col)), shape=(N, N))
        
        return A, D
    
    def _build_delaunay_graph(self, coordinates: np.ndarray) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Build graph from Delaunay triangulation."""
        N = len(coordinates)
        
        try:
            tri = Delaunay(coordinates)
        except Exception as e:
            warnings.warn(f"Delaunay triangulation failed: {e}. Falling back to KNN.")
            return self._build_knn_graph(coordinates, k=6)
        
        # Extract edges from simplices
        edges = set()
        for simplex in tri.simplices:
            n_vertices = len(simplex)
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        row, col, data, dist_data = [], [], [], []
        
        for i, j in edges:
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            # Add both directions
            row.extend([i, j])
            col.extend([j, i])
            data.extend([1.0, 1.0])
            dist_data.extend([dist, dist])
        
        A = sp.csr_matrix((data, (row, col)), shape=(N, N))
        D = sp.csr_matrix((dist_data, (row, col)), shape=(N, N))
        
        return A, D
    
    def _compute_edge_distances(self, A: sp.spmatrix, coordinates: np.ndarray) -> sp.spmatrix:
        """Compute Euclidean distances for all edges in adjacency matrix."""
        A_coo = A.tocoo()
        
        # Vectorized computation
        row_coords = coordinates[A_coo.row]
        col_coords = coordinates[A_coo.col]
        diff = row_coords - col_coords
        distances = np.linalg.norm(diff, axis=1)
        
        return sp.csr_matrix((distances, (A_coo.row, A_coo.col)), shape=A.shape)
    
    @staticmethod
    def _compute_laplacian(A: sp.spmatrix) -> sp.spmatrix:
        """Compute graph Laplacian L = D - A."""
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - A
        return L.tocsr()
    
    @staticmethod
    def _compute_normalized_laplacian(A: sp.spmatrix) -> sp.spmatrix:
        """Compute normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}."""
        degrees = np.array(A.sum(axis=1)).flatten()
        
        # Avoid division by zero for isolated nodes
        degrees = np.where(degrees > 0, degrees, 1.0)
        
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        I = sp.eye(A.shape[0])
        L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
        
        return L_norm.tocsr()


class GeometricWeightComputer:
    """Computes geometric weights for sensor placement.

    These weights help prioritize areas with:
    - High spatial gradients (sharp fronts)
    - Geometric significance (corners, boundaries)
    - Flow features (vortices, stagnation points)
    """
    
    def __init__(self, mesh: MeshGeometry):
        """Initializes the GeometricWeightComputer.

        Args:
            mesh (MeshGeometry): The mesh geometry information.
        """
        self.mesh = mesh
    
    def compute_gradient_weights(self, field: np.ndarray, method: str = 'fd') -> np.ndarray:
        """Computes the spatial gradient magnitude for each cell.

        Args:
            field (np.ndarray): A scalar field of values at each cell, with
                shape (N_cells, N_time) or (N_cells,).
            method (str): The method to use for gradient computation. Can be
                'fd' (finite difference) or 'graph' (graph-based). Defaults to
                'fd'.

        Returns:
            np.ndarray: The gradient magnitude at each cell, averaged over
            time if the field is 2D.
        """
        if field.ndim == 2:
            # Average over time
            field_mean = field.mean(axis=1)
        else:
            field_mean = field
        
        if method == 'fd':
            return self._compute_fd_gradient(field_mean)
        elif method == 'graph':
            return self._compute_graph_gradient(field_mean)
        else:
            raise ValueError(f"Unknown gradient method: {method}")
    
    def _compute_fd_gradient(self, field: np.ndarray) -> np.ndarray:
        """Finite difference gradient using vectorized sparse matrix operations."""
        A = self.mesh.adjacency_matrix
        D_mat = self.mesh.distances

        if D_mat is None:
            D_mat = self._compute_distances_from_adjacency(A)

        N = len(field)
        
        # Convert to COO for efficient access
        if not sp.isspmatrix_coo(D_mat):
            D_coo = D_mat.tocoo()
        else:
            D_coo = D_mat

        # Create sparse matrix W2 where W2[i, j] = 1/d_ij^2 for d_ij > 0
        mask = D_coo.data > 0
        if not np.any(mask):
            return np.zeros(N)

        row = D_coo.row[mask]
        col = D_coo.col[mask]
        data_inv_sq = 1.0 / (D_coo.data[mask] ** 2)
        
        W2 = sp.csr_matrix((data_inv_sq, (row, col)), shape=(N, N))

        # Count of valid neighbors per row (k)
        # Using a binary matrix of valid connections
        ones_data = np.ones_like(data_inv_sq)
        W2_binary = sp.csr_matrix((ones_data, (row, col)), shape=(N, N))
        k = np.array(W2_binary.sum(axis=1)).flatten()

        # Compute gradient terms:
        # S_i = sum_j ( (f_j - f_i)^2 / d_ij^2 )
        #     = sum_j ( f_j^2/d_ij^2 - 2*f_i*f_j/d_ij^2 + f_i^2/d_ij^2 )
        #     = (W2 @ f^2)_i - 2*f_i * (W2 @ f)_i + f_i^2 * (W2 @ 1)_i

        f_sq = field ** 2
        term1 = W2.dot(f_sq)
        term2 = -2 * field * W2.dot(field)

        # W2 @ 1 is just the row sums of W2
        w2_row_sums = np.array(W2.sum(axis=1)).flatten()
        term3 = f_sq * w2_row_sums
        
        S = term1 + term2 + term3
        
        # Handle floating point inaccuracies
        S = np.maximum(S, 0)

        # Compute root mean square
        # gradient = sqrt( S / k )

        # Avoid division by zero
        k_safe = np.where(k > 0, k, 1.0)
        gradient_sq = S / k_safe
        gradient = np.sqrt(gradient_sq)

        # Zero out gradients where k=0
        gradient[k == 0] = 0.0
        
        return gradient
    
    def _compute_graph_gradient(self, field: np.ndarray) -> np.ndarray:
        """Graph-based gradient using Laplacian."""
        L = self.mesh.laplacian_matrix
        
        # Graph gradient: |L * f|
        grad_field = L @ field
        gradient = np.abs(grad_field)
        
        return gradient
    
    def _compute_distances_from_adjacency(self, A: sp.spmatrix) -> sp.spmatrix:
        """Compute distances using coordinates if not available."""
        coords = self.mesh.coordinates
        A_coo = A.tocoo()
        
        # Vectorized computation
        row_coords = coords[A_coo.row]
        col_coords = coords[A_coo.col]
        diff = row_coords - col_coords
        distances = np.linalg.norm(diff, axis=1)
        
        return sp.csr_matrix((distances, (A_coo.row, A_coo.col)), shape=A.shape)
    
    def update_proximity_penalty(self, new_sensor_idx: int, current_min_dists: np.ndarray, min_distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """Updates the proximity penalty incrementally given a new sensor.

        This method avoids recomputing the KDTree by updating the minimum
        distances directly, which is O(N) instead of O(N log K).

        Args:
            new_sensor_idx (int): The index of the newly added sensor.
            current_min_dists (np.ndarray): The current minimum distance from
                each cell to any existing sensor. Shape (N,).
            min_distance (float): The minimum allowed distance.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - penalty (np.ndarray): The new penalty values.
                - new_min_dists (np.ndarray): The updated minimum distances.
        """
        sensor_coord = self.mesh.coordinates[new_sensor_idx]

        # Calculate distance from new sensor to all points
        diff = self.mesh.coordinates - sensor_coord
        dists = np.linalg.norm(diff, axis=1)

        # Update minimum distances
        new_min_dists = np.minimum(current_min_dists, dists)

        # Calculate penalty: exponential decay
        penalty = np.exp(-new_min_dists / (min_distance + 1e-10))

        return penalty, new_min_dists

    def compute_proximity_penalty(self, sensor_positions: np.ndarray,
                                   min_distance: float) -> np.ndarray:
        """Computes a penalty for placing sensors too close to existing ones.

        Args:
            sensor_positions (np.ndarray): The indices of currently placed
                sensors, with shape (N_sensors,).
            min_distance (float): The minimum allowed distance between sensors,
                in coordinate units.

        Returns:
            np.ndarray: The penalty values for each cell, with shape
            (N_cells,). Higher values are less desirable.
        """
        N = len(self.mesh.coordinates)
        penalty = np.zeros(N)
        
        if len(sensor_positions) == 0:
            return penalty
        
        # Get coordinates of existing sensors
        sensor_coords = self.mesh.coordinates[sensor_positions]
        
        # Compute distance from each cell to nearest sensor
        tree = KDTree(sensor_coords)
        distances, _ = tree.query(self.mesh.coordinates)
        
        # Apply penalty: exponential decay with distance
        # penalty = exp(-distances / min_distance)
        penalty = np.exp(-distances / (min_distance + 1e-10))
        
        return penalty


def estimate_characteristic_length(mesh: MeshGeometry) -> float:
    """Estimates the characteristic length scale of the mesh.

    This function uses the average edge length from the adjacency matrix and
    distances to estimate the characteristic length.

    Args:
        mesh (MeshGeometry): The mesh geometry.

    Returns:
        float: The characteristic length, which is the mean edge length.
    """
    if mesh.distances is not None:
        D_coo = mesh.distances.tocoo()
        return float(np.mean(D_coo.data))
    else:
        # Fallback: estimate from coordinate extent
        coords = mesh.coordinates
        extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        N = len(coords)
        # Rough estimate assuming uniform distribution
        return float(np.mean(extent) / np.sqrt(N))
