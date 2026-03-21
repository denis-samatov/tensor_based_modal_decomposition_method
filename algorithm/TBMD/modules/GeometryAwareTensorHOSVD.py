"""
Geometry-Aware Tensor HOSVD with Laplacian Regularization.

This module extends the standard Tucker/HOSVD decomposition by incorporating
geometric information from unstructured meshes. Key features:

1. **Laplacian Regularization**: During spatial factor extraction, we add a
   smoothness penalty based on the mesh graph Laplacian to encourage spatially
   smooth modes that respect the mesh geometry.

2. **Regularized HOOI**: Instead of standard HOOI (Higher-Order Orthogonal Iteration),
   we solve:
       min_{U_1} ||X_{(1)} - U_1 * G_{(1)} * (U_3 ⊗ U_2)^T||_F^2 + α * ||L * U_1||_F^2
   
   where L is the (normalized) graph Laplacian and α is the regularization weight.

3. **Flexible Regularization**: Can apply Laplacian penalty to spatial modes only,
   or to multiple modes if desired.

Mathematical Background
-----------------------
Standard Tucker decomposition:
    X ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃

With Laplacian regularization on mode 1 (spatial):
    min ||X - G ×₁ U₁ ×₂ U₂ ×₃ U₃||²_F + α ||L U₁||²_F

The solution involves solving a generalized eigenvalue problem or using
iterative methods with Tikhonov regularization.

References
----------
- Kolda & Bader (2009). "Tensor Decompositions and Applications"
- Jiang et al. (2017). "Smooth Tucker decomposition for brain connectivity"
"""

import numpy as np
import torch
import tensorly as tl
import logging
from typing import Union, List, Optional, Tuple, Dict
from dataclasses import dataclass
from scipy import sparse as sp

from .TensorHOSVD import (
    TensorProcessor,
    DecomposerState,
    ValidationError,
    TensorDecompositionError
)
from ..utils.geometry import MeshGeometry, MeshGraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class GeometryAwareConfig:
    """Configuration for geometry-aware HOSVD.

    Attributes:
        alpha (float): Laplacian regularization strength. Higher values lead
            to smoother modes. Defaults to 0.01.
        spatial_modes (List[int]): The modes to regularize (0-indexed).
            Typically, mode 0 is used for spatial regularization. Defaults to
            [0].
        laplacian_type (str): The type of Laplacian to use ('standard' or
            'normalized'). Defaults to 'normalized'.
        connectivity_type (str): The method for building the mesh graph
            ('grid', 'knn', 'radius', or 'delaunay'). Defaults to 'grid'.
        connectivity_params (dict): Parameters for graph construction (e.g.,
            `{'k': 6}` for 'knn').
        use_generalized_eig (bool): If `True`, solves a generalized eigenvalue
            problem; otherwise, uses Tikhonov regularization. Defaults to
            `False`.
    """
    alpha: float = 0.01
    spatial_modes: List[int] = None
    laplacian_type: str = 'normalized'
    connectivity_type: str = 'grid'
    connectivity_params: Dict = None
    use_generalized_eig: bool = False

    def __post_init__(self):
        """Validates the configuration after initialization."""
        if self.spatial_modes is None:
            self.spatial_modes = [0]
        if self.connectivity_params is None:
            self.connectivity_params = {}
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.laplacian_type not in ['standard', 'normalized']:
            raise ValueError(
                "laplacian_type must be 'standard' or 'normalized'")


class GeometryAwareTuckerCore:
    """A core implementation of geometry-aware Tucker decomposition.

    This class uses alternating least squares (ALS) with Laplacian
    regularization on specified modes.

    Args:
        mesh (MeshGeometry): The mesh geometry, including Laplacian matrices.
        geo_config (GeometryAwareConfig): The geometry-aware configuration.
        ranks (Optional[Union[int, List[int]]]): The Tucker ranks.
        epsilon (float): The convergence tolerance. Defaults to 1e-2.
        max_iter (int): The maximum number of ALS iterations. Defaults to 50.
        random_state (Optional[int]): The random seed for reproducibility.
    """

    def __init__(self,
                 mesh: MeshGeometry,
                 geo_config: GeometryAwareConfig,
                 ranks: Optional[Union[int, List[int]]] = None,
                 epsilon: float = 1e-2,
                 max_iter: int = 50,
                 random_state: Optional[int] = None):
        self.mesh = mesh
        self.geo_config = geo_config
        self.ranks = ranks
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self._eye_cache = {}

        # Select Laplacian type
        if geo_config.laplacian_type == 'normalized':
            self.laplacian = mesh.normalized_laplacian
        else:
            self.laplacian = mesh.laplacian_matrix

        logger.info(f"GeometryAwareTuckerCore initialized with α={geo_config.alpha}, "
                    f"regularizing modes {geo_config.spatial_modes}")

    def _prepare_laplacian(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Precomputes the Laplacian regularization term L^T L."""
        # Convert Laplacian to torch (if sparse)
        if sp.issparse(self.laplacian):
            L_torch = self._sparse_scipy_to_torch(
                self.laplacian, device, dtype)
        else:
            L_torch = torch.from_numpy(self.laplacian).to(
                device=device, dtype=dtype)

        # Compute L^T L (Laplacian regularization term)
        if L_torch.is_sparse:
            # Sparse matrix multiplication
            LTL = torch.sparse.mm(L_torch.t(), L_torch).to_dense()
        else:
            LTL = L_torch.T @ L_torch

        return LTL

    def decompose(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs geometry-aware Tucker decomposition.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing the
            core tensor and a list of factor matrices.
        """
        # Validate and normalize ranks
        if self.ranks is None:
            ranks = [min(tensor.shape)] * len(tensor.shape)
        elif isinstance(self.ranks, int):
            ranks = [self.ranks] * len(tensor.shape)
        else:
            ranks = self.ranks

        if len(ranks) != len(tensor.shape):
            raise ValidationError(
                f"Ranks length {len(ranks)} must match tensor ndim {len(tensor.shape)}")

        # Precompute LTL
        LTL = self._prepare_laplacian(tensor.device, tensor.dtype)

        # Initialize factors using standard SVD (or HOSVD)
        factors = self._initialize_factors(tensor, ranks)

        # ALS iterations with Laplacian regularization
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # Update each mode
            for mode in range(len(tensor.shape)):
                if mode in self.geo_config.spatial_modes:
                    # Regularized update
                    factors[mode] = self._update_factor_regularized(
                        tensor, factors, mode, ranks[mode], LTL
                    )
                else:
                    # Standard update
                    factors[mode] = self._update_factor_standard(
                        tensor, factors, mode, ranks[mode]
                    )

            # Compute core tensor
            core = self._compute_core(tensor, factors)

            # Check convergence
            reconstruction = tl.tucker_to_tensor((core, factors))
            error = float(tl.norm(tensor - reconstruction) / tl.norm(tensor))

            if abs(prev_error - error) < self.epsilon:
                logger.info(
                    f"Converged at iteration {iteration+1}, error={error:.6f}")
                break

            prev_error = error

        return core, factors

    def _initialize_factors(self, tensor: torch.Tensor, ranks: List[int]) -> List[torch.Tensor]:
        """Initialize factors using truncated SVD on each unfolding."""
        factors = []

        for mode in range(len(tensor.shape)):
            unfolding = tl.unfold(tensor, mode)

            if unfolding.shape[0] <= unfolding.shape[1]:
                # Tall matrix: compute left singular vectors
                try:
                    U, _, _ = torch.svd(unfolding)
                    factor = U[:, :ranks[mode]]
                except:
                    # Fallback to random initialization
                    factor = torch.randn(unfolding.shape[0], ranks[mode],
                                         device=tensor.device, dtype=tensor.dtype)
                    factor, _ = torch.qr(factor)
            else:
                # Wide matrix: compute via covariance
                try:
                    C = unfolding @ unfolding.T
                    eigenvalues, eigenvectors = torch.linalg.eigh(C)
                    # Take largest eigenvalues
                    idx = torch.argsort(eigenvalues, descending=True)[
                        :ranks[mode]]
                    factor = eigenvectors[:, idx]
                except:
                    factor = torch.randn(unfolding.shape[0], ranks[mode],
                                         device=tensor.device, dtype=tensor.dtype)
                    factor, _ = torch.qr(factor)

            factors.append(factor)

        return factors

    def _update_factor_standard(self, tensor: torch.Tensor, factors: List[torch.Tensor],
                                mode: int, rank: int) -> torch.Tensor:
        """Standard factor update (no regularization)."""
        # Compute unfolding
        unfolding = tl.unfold(tensor, mode)

        # Compute product of other factors (Khatri-Rao product)
        G = self._compute_gramian(factors, mode)

        # Normal equations: U_mode = X_(mode) * G * (G^T G)^{-1}
        rhs = unfolding @ G
        lhs = G.T @ G

        try:
            # Regularize for numerical stability
            lhs_reg = lhs + 1e-8 * \
                self._get_identity(lhs.shape[0], lhs.device, lhs.dtype)
            U_new = torch.linalg.solve(lhs_reg.T, rhs.T).T
        except:
            # Fallback to pseudo-inverse
            U_new = rhs @ torch.linalg.pinv(lhs)

        # Orthogonalize (optional but recommended)
        U_new, _ = torch.linalg.qr(U_new)

        return U_new

    def _update_factor_regularized(self, tensor: torch.Tensor, factors: List[torch.Tensor],
                                   mode: int, rank: int, LTL: torch.Tensor) -> torch.Tensor:
        """
        Regularized factor update with Laplacian penalty.

        Solves: min ||X_(mode) - U * G^T||²_F + α ||L * U||²_F

        Normal equations:
            (X_(mode) * G * G^T + α * L^T L) * U = X_(mode) * G
        """
        # Compute unfolding
        unfolding = tl.unfold(tensor, mode)

        # Compute product of other factors
        G = self._compute_gramian(factors, mode)

        # Data term: X_(mode) * G
        rhs = unfolding @ G

        # LHS: G^T G (data term) + α L^T L (regularization)
        GTG = G.T @ G

        # Check dimension compatibility
        if LTL.shape[0] != unfolding.shape[0]:
            raise ValidationError(
                f"Laplacian size {LTL.shape[0]} doesn't match mode size {unfolding.shape[0]}"
            )

        # Build augmented system
        # (Data term)     (Regularization)
        # rhs^T G^T G  +  α U^T L^T L  →  we want U
        #
        # More efficiently: solve (G^T G + α L^T L) U = rhs^T

        # We need to solve for each column of U separately, or use a block solver
        # For simplicity, solve: U * (GTG) = rhs - we add penalty differently

        # Alternative formulation: solve (GTG + α * L^T L ⊗ I_r) vec(U) = vec(rhs)
        # But this is expensive. Instead, use iterative refinement or direct solve.

        # Direct approach: for each column u_i of U:
        #   (G^T G + α L^T L) u_i = rhs[:, i]

        # Better: solve all at once using Kronecker structure, but for now:
        # We solve column-by-column

        U_new = torch.zeros(
            unfolding.shape[0], rank, device=tensor.device, dtype=tensor.dtype)

        # Precompute regularized LHS (same for all columns)
        # Note: we need (sum_j u_j GTG_jj + alpha sum_i (L u)_i^2)
        # Standard formulation: minimize ||X - U G^T||^2 + alpha ||L U||^2
        # Taking derivative w.r.t. U:
        #   -2 (X G - U G^T G) + 2 alpha L^T L U = 0
        #   →  (G G^T + alpha L^T L) U = X G

        GGT = GTG  # Note: this is actually G^T G from the Khatri-Rao product

        # Build LHS for normal equations
        # We have: X_(mode) @ G  and (G^T @ G)
        # Regularized: (G^T G) U^T = (X_(mode) G)^T  →  U (G^T G) = X_(mode) G
        # With regularization: U (G^T G + α I ⊗ L^T L) = X_(mode) G

        # Simpler: work in unfolding space
        # min_U ||X_(mode) - U K||^2 + α ||L U||^2,  where K = (other factors Khatri-Rao)^T
        # Solution: U = (X_(mode) K^T) (K K^T + α L^T L)^{-1}

        KKT = G @ G.T  # This is the Gram matrix in the mode space

        # Regularized system
        lhs = KKT + self.geo_config.alpha * LTL + 1e-8 * \
            self._get_identity(KKT.shape[0], tensor.device, tensor.dtype)

        # Solve for U^T (transpose of factor)
        rhs_T = rhs  # This is actually X_(mode) @ K^T

        try:
            U_new = torch.linalg.solve(lhs, rhs_T)
        except:
            # Fallback to pseudo-inverse
            U_new = torch.linalg.pinv(lhs) @ rhs_T

        # Orthogonalize (optional)
        U_new, _ = torch.linalg.qr(U_new)

        return U_new

    def _get_identity(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create a cached identity matrix of the specified size."""
        key = (size, str(device), dtype)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(size, device=device, dtype=dtype)
        return self._eye_cache[key]

    def _compute_gramian(self, factors: List[torch.Tensor], skip_mode: int) -> torch.Tensor:
        """
        Compute Khatri-Rao product of all factors except skip_mode.

        Returns G such that unfolding ≈ U_{skip_mode} @ G.T
        """
        # Start with the last mode (excluding skip_mode)
        modes = [i for i in range(len(factors)) if i != skip_mode]

        if len(modes) == 0:
            return torch.ones(1, 1, device=factors[0].device, dtype=factors[0].dtype)

        # Khatri-Rao product (column-wise Kronecker)
        # For Tucker, we need the Kronecker product in reverse order
        G = factors[modes[-1]]

        for mode in reversed(modes[:-1]):
            G = self._khatri_rao(factors[mode], G)

        return G

    @staticmethod
    def _khatri_rao(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute Khatri-Rao product (column-wise Kronecker product).

        Parameters
        ----------
        A : Tensor (I, K)
        B : Tensor (J, K)

        Returns
        -------
        Tensor (I*J, K)
        """
        if A.shape[1] != B.shape[1]:
            raise ValueError(
                f"Number of columns must match: {A.shape[1]} != {B.shape[1]}")

        I, K = A.shape
        J = B.shape[0]

        # Efficient implementation using einsum
        # Result[i*J + j, k] = A[i, k] * B[j, k]
        result = (A.unsqueeze(1) * B.unsqueeze(0)).reshape(I * J, K)

        return result

    def _compute_core(self, tensor: torch.Tensor, factors: List[torch.Tensor]) -> torch.Tensor:
        """Compute core tensor given factors."""
        # G = X ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ
        core = tensor.clone()

        for mode, factor in enumerate(factors):
            core = tl.tenalg.mode_dot(core, factor.T, mode=mode)

        return core

    @staticmethod
    def _sparse_scipy_to_torch(scipy_sparse: sp.spmatrix,
                               device: torch.device,
                               dtype: torch.dtype) -> torch.Tensor:
        """Convert scipy sparse matrix to torch sparse tensor."""
        coo = scipy_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data).to(dtype)
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, shape, device=device)


class GeometryAwareTuckerDecomposer:
    """A high-level interface for geometry-aware Tucker decomposition.

    Args:
        tensor (Union[torch.Tensor, np.ndarray]): The input tensor to decompose.
        mesh (Union[MeshGeometry, Tuple[int, ...]]): A `MeshGeometry` object
            or a tuple representing the spatial shape of the mesh.
        geo_config (Optional[GeometryAwareConfig]): Configuration for the
            geometry-aware decomposition.
        ranks (Optional[Union[int, List[int]]]): The Tucker ranks.
        epsilon (float): The convergence tolerance. Defaults to 1e-2.
        max_iter (int): The maximum number of ALS iterations. Defaults to 50.
        random_state (Optional[int]): The random seed for reproducibility.
        device (str): The PyTorch device to use. Defaults to 'cpu'.
        dtype (torch.dtype): The data type for the tensor. Defaults to
            `torch.float32`.
    """

    def __init__(self,
                 tensor: Union[torch.Tensor, np.ndarray],
                 mesh: Union[MeshGeometry, Tuple[int, ...]],
                 geo_config: Optional[GeometryAwareConfig] = None,
                 ranks: Optional[Union[int, List[int]]] = None,
                 epsilon: float = 1e-2,
                 max_iter: int = 50,
                 random_state: Optional[int] = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.processor = TensorProcessor(device, dtype)
        self.tensor = self.processor.process_tensors(tensor)

        # Build or validate mesh
        if isinstance(mesh, tuple):
            # Build grid mesh from shape
            geo_config = geo_config or GeometryAwareConfig()
            builder = MeshGraphBuilder(
                connectivity_type=geo_config.connectivity_type,
                **geo_config.connectivity_params
            )
            self.mesh = builder.build_from_shape(mesh)
            logger.info(
                f"Built {geo_config.connectivity_type} mesh for shape {mesh}")
        else:
            self.mesh = mesh

        # Validate mesh vs tensor shape
        spatial_shape = self.tensor.shape[:-1] if len(
            self.tensor.shape) == 3 else self.tensor.shape
        expected_cells = int(np.prod(spatial_shape))

        if self.mesh.adjacency_matrix.shape[0] != expected_cells:
            raise ValidationError(
                f"Mesh has {self.mesh.adjacency_matrix.shape[0]} cells but "
                f"tensor spatial size is {expected_cells}"
            )

        self.geo_config = geo_config or GeometryAwareConfig()

        # Initialize core decomposer
        self.decomposer_core = GeometryAwareTuckerCore(
            mesh=self.mesh,
            geo_config=self.geo_config,
            ranks=ranks,
            epsilon=epsilon,
            max_iter=max_iter,
            random_state=random_state
        )

        # State
        self.state = DecomposerState.INITIALIZED
        self._core = None
        self._factors = None

    def decompose(self) -> None:
        """Performs the geometry-aware decomposition."""
        if self.state != DecomposerState.INITIALIZED:
            logger.warning(f"Re-decomposing from state {self.state}")

        try:
            self._core, self._factors = self.decomposer_core.decompose(
                self.tensor)
            self.state = DecomposerState.DECOMPOSED
            logger.info("Geometry-aware decomposition completed")
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            raise TensorDecompositionError(
                f"Geometry-aware decomposition failed: {e}")

    @property
    def cores(self) -> torch.Tensor:
        """Returns the core tensor of the decomposition.

        Returns:
            torch.Tensor: The core tensor.
        """
        if self.state == DecomposerState.INITIALIZED:
            raise ValueError("Call decompose() first")
        return self._core

    @property
    def factors(self) -> List[torch.Tensor]:
        """Returns the factor matrices of the decomposition.

        Returns:
            List[torch.Tensor]: A list of factor matrices.
        """
        if self.state == DecomposerState.INITIALIZED:
            raise ValueError("Call decompose() first")
        return self._factors

    def reconstruct(self) -> torch.Tensor:
        """Reconstructs the tensor from its factors.

        Returns:
            torch.Tensor: The reconstructed tensor.
        """
        if self.state != DecomposerState.DECOMPOSED:
            raise ValueError("Call decompose() first")

        return tl.tucker_to_tensor((self._core, self._factors))

    def get_spatial_modes(self) -> torch.Tensor:
        """Returns the spatial factor matrix (mode 0 by default).

        Returns:
            torch.Tensor: The spatial factor matrix.
        """
        if self.state != DecomposerState.DECOMPOSED:
            raise ValueError("Call decompose() first")

        # Assume mode 0 is spatial
        return self._factors[0]
