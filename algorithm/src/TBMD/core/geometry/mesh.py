"""
Mesh geometry data structures.
"""

import numpy as np
import torch
import scipy.sparse as sp
from typing import Optional
from dataclasses import dataclass
from TBMD.core.utils.misc import get_torch_device

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
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes (cells) in the mesh."""
        return self.adjacency_matrix.shape[0]
    
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
