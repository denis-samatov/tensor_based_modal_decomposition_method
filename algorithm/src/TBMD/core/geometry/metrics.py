"""
Geometric metrics and weight computation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree

from .mesh import MeshGeometry

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
        # Handle 3D input (Nx, Ny, Nz, T) or (Nx, Ny, Nz)
        if field.ndim > 2:
            # Flatten spatial dimensions
            if field.ndim == 3: # (Nx, Ny, Nz) -> (N_cells,)
                field = field.flatten()
            else: # (Nx, Ny, Nz, T) -> (N_cells, T)
                field = field.reshape(-1, field.shape[-1])
        
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
        if D_coo.nnz > 0:
            return float(np.mean(D_coo.data))
    
    # Fallback: estimate from coordinate extent
    coords = mesh.coordinates
    extent = np.max(coords, axis=0) - np.min(coords, axis=0)
    N = len(coords)
    dim = coords.shape[1]
    
    # Estimate based on volume/area per cell
    # V_total ≈ prod(extent)
    # V_cell ≈ V_total / N
    # L_char ≈ V_cell^(1/dim)
    
    # Use geometric mean of extents for robustness against flat dimensions
    # Or simply prod(extent) if all dims are significant
    valid_extents = extent[extent > 1e-6]
    if len(valid_extents) == 0:
        return 1.0
        
    volume = np.prod(valid_extents)
    effective_dim = len(valid_extents)
    
    return float((volume / N) ** (1.0 / effective_dim))
