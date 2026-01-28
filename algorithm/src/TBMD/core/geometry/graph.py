"""
Mesh graph construction utilities.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
from scipy.spatial import KDTree, Delaunay
import warnings

from .mesh import MeshGeometry

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
        """
        Build 6-connected grid graph for 3D.
        
        Shape is interpreted as (H, W, D) corresponding to axes (z, y, x).
        """
        H, W, D = shape
        N = H * W * D
        
        # Create coordinate grid
        # np.mgrid[0:H, 0:W, 0:D] returns grids where:
        # z_coords varies along axis 0 (H)
        # y_coords varies along axis 1 (W)
        # x_coords varies along axis 2 (D)
        z_coords, y_coords, x_coords = np.mgrid[0:H, 0:W, 0:D]
        
        coordinates = np.stack([
            x_coords.ravel(),
            y_coords.ravel(),
            z_coords.ravel()
        ], axis=1)
        
        # Build adjacency (6-connectivity)
        row, col, data = [], [], []
        
        for i in range(H):      # z-axis
            for j in range(W):  # y-axis
                for k in range(D): # x-axis
                    idx = i * W * D + j * D + k
                    
                    # Neighbor in y-axis (j+1)
                    if j < W - 1:
                        neighbor = i * W * D + (j + 1) * D + k
                        row.extend([idx, neighbor])
                        col.extend([neighbor, idx])
                        data.extend([1.0, 1.0])
                    
                    # Neighbor in z-axis (i+1)
                    if i < H - 1:
                        neighbor = (i + 1) * W * D + j * D + k
                        row.extend([idx, neighbor])
                        col.extend([neighbor, idx])
                        data.extend([1.0, 1.0])
                    
                    # Neighbor in x-axis (k+1)
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
