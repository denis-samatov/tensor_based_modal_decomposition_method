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

from TBMD.utils.utils import to_torch_tensor, get_torch_device


@dataclass
class MeshGeometry:
    """
    Container for mesh geometry information.
    
    Attributes
    ----------
    adjacency_matrix : scipy.sparse matrix (N_cells × N_cells)
        Sparse adjacency matrix representing cell connectivity.
    laplacian_matrix : scipy.sparse matrix (N_cells × N_cells)
        Graph Laplacian matrix L = D - A.
    normalized_laplacian : scipy.sparse matrix (N_cells × N_cells)
        Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
    coordinates : np.ndarray (N_cells, spatial_dim)
        Cell center coordinates (x, y) or (x, y, z).
    distances : scipy.sparse matrix (N_cells × N_cells), optional
        Pairwise distances between adjacent cells.
    gradient_weights : scipy.sparse matrix (N_cells × N_cells), optional
        Edge weights based on gradient magnitude estimation.
    """
    adjacency_matrix: sp.spmatrix
    laplacian_matrix: sp.spmatrix
    normalized_laplacian: sp.spmatrix
    coordinates: np.ndarray
    distances: Optional[sp.spmatrix] = None
    gradient_weights: Optional[sp.spmatrix] = None
    
    def to_torch(self, device: str = 'cpu', dtype: torch.dtype = torch.float32) -> 'TorchMeshGeometry':
        """Convert sparse matrices to PyTorch sparse tensors."""
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
    """PyTorch version of MeshGeometry with sparse tensors."""
    adjacency_matrix: torch.Tensor
    laplacian_matrix: torch.Tensor
    normalized_laplacian: torch.Tensor
    coordinates: torch.Tensor
    distances: Optional[torch.Tensor] = None
    gradient_weights: Optional[torch.Tensor] = None


class MeshGraphBuilder:
    """
    Build cell adjacency graphs for structured and unstructured meshes.
    
    Supports multiple connectivity strategies:
    - 'grid': Regular grid with 4-connectivity (2D) or 6-connectivity (3D)
    - 'knn': K-nearest neighbors based on cell centers
    - 'radius': Connect cells within a distance threshold
    - 'delaunay': Delaunay triangulation-based connectivity (2D/3D)
    """
    
    def __init__(self, connectivity_type: str = 'knn', **kwargs):
        """
        Parameters
        ----------
        connectivity_type : {'grid', 'knn', 'radius', 'delaunay'}
            Type of connectivity to build.
        **kwargs : dict
            Additional parameters:
            - k : int (for 'knn'), default 6
            - radius : float (for 'radius')
            - periodic : bool (for 'grid'), default False
        """
        self.connectivity_type = connectivity_type
        self.params = kwargs
        
    def build_from_shape(self, spatial_shape: Tuple[int, ...]) -> MeshGeometry:
        """
        Build graph for a regular grid mesh.
        
        Parameters
        ----------
        spatial_shape : tuple of int
            Shape of the grid (H, W) or (H, W, D).
        
        Returns
        -------
        MeshGeometry
            Constructed mesh geometry.
        """
        if len(spatial_shape) == 2:
            return self._build_2d_grid(spatial_shape)
        elif len(spatial_shape) == 3:
            return self._build_3d_grid(spatial_shape)
        else:
            raise ValueError(f"Unsupported spatial dimension: {len(spatial_shape)}")
    
    def build_from_coordinates(self, coordinates: np.ndarray) -> MeshGeometry:
        """
        Build graph from cell center coordinates (unstructured mesh).
        
        Parameters
        ----------
        coordinates : np.ndarray (N_cells, spatial_dim)
            Cell center coordinates.
        
        Returns
        -------
        MeshGeometry
            Constructed mesh geometry.
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
        
        row, col, data, dist_data = [], [], [], []
        
        for i, j in pairs:
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            # Add both directions
            row.extend([i, j])
            col.extend([j, i])
            data.extend([1.0, 1.0])
            dist_data.extend([dist, dist])
        
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
        distances = []
        
        for i, j in zip(A_coo.row, A_coo.col):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances.append(dist)
        
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
    """
    Compute geometric weights for sensor placement.
    
    These weights help prioritize areas with:
    - High spatial gradients (sharp fronts)
    - Geometric significance (corners, boundaries)
    - Flow features (vortices, stagnation points)
    """
    
    def __init__(self, mesh: MeshGeometry):
        """
        Parameters
        ----------
        mesh : MeshGeometry
            Mesh geometry information.
        """
        self.mesh = mesh
    
    def compute_gradient_weights(self, field: np.ndarray, method: str = 'fd') -> np.ndarray:
        """
        Compute spatial gradient magnitude for each cell.
        
        Parameters
        ----------
        field : np.ndarray (N_cells, N_time) or (N_cells,)
            Scalar field values at each cell.
        method : {'fd', 'graph'}
            'fd': Finite difference approximation using neighbors
            'graph': Graph-based gradient using Laplacian
        
        Returns
        -------
        np.ndarray (N_cells,)
            Gradient magnitude at each cell (averaged over time if field is 2D).
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
        """Finite difference gradient using neighbor information."""
        A = self.mesh.adjacency_matrix
        D_mat = self.mesh.distances
        
        if D_mat is None:
            # Fallback: compute distances on the fly
            D_mat = self._compute_distances_from_adjacency(A)
        
        # Approximate gradient magnitude
        gradient = np.zeros(len(field))
        
        A_coo = A.tocoo()
        D_coo = D_mat.tocoo()
        
        # Build a dict for quick lookup
        dist_dict = {(i, j): d for i, j, d in zip(D_coo.row, D_coo.col, D_coo.data)}
        
        for idx in range(len(field)):
            neighbors = A.indices[A.indptr[idx]:A.indptr[idx+1]]
            if len(neighbors) == 0:
                continue
            
            grad_components = []
            for neighbor in neighbors:
                key = (idx, neighbor)
                dist = dist_dict.get(key, 1.0)
                if dist > 0:
                    grad_components.append((field[neighbor] - field[idx]) / dist)
            
            if grad_components:
                gradient[idx] = np.sqrt(np.mean(np.array(grad_components)**2))
        
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
        
        distances = []
        for i, j in zip(A_coo.row, A_coo.col):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances.append(dist)
        
        return sp.csr_matrix((distances, (A_coo.row, A_coo.col)), shape=A.shape)
    
    def compute_proximity_penalty(self, sensor_positions: np.ndarray,
                                   min_distance: float) -> np.ndarray:
        """
        Compute penalty for placing sensors too close to existing ones.
        
        Parameters
        ----------
        sensor_positions : np.ndarray (N_sensors,)
            Indices of currently placed sensors.
        min_distance : float
            Minimum allowed distance between sensors (in coordinate units).
        
        Returns
        -------
        np.ndarray (N_cells,)
            Penalty values for each cell (higher = less desirable).
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
    """
    Estimate characteristic length scale of the mesh.
    
    Uses average edge length from adjacency matrix and distances.
    
    Parameters
    ----------
    mesh : MeshGeometry
        Mesh geometry.
    
    Returns
    -------
    float
        Characteristic length (mean edge length).
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

