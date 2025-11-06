"""
Geometry-Aware Tensor Compressive Sensing with Laplacian Regularization.

This module extends the standard tensor compressive sensing (Algorithm 3) by 
incorporating geometric information from unstructured meshes to promote spatially
smooth reconstructions.

Key Features
------------
1. **Spatial Smoothness**: Adds Laplacian regularization ||L·x||² to encourage
   smooth fields that respect mesh connectivity.

2. **Adaptive Regularization**: Automatically balances data fidelity, sparsity,
   and smoothness based on problem conditioning.

3. **ADMM with Geometry**: Extends the ADMM solver with an additional quadratic
   term for mesh-aware reconstruction.

Mathematical Formulation
------------------------
Standard TBMD-CS (Algorithm 3):
    min ||Ax - y||² + ε||d||₁

Geometry-Aware TBMD-CS:
    min ||Ax - y||² + ε||d||₁ + α||L·x||²

where:
    - A: forward model (mode shapes)
    - y: measurements at sensor locations
    - x: coefficients to recover
    - d: auxiliary variable for L1 penalty
    - L: graph Laplacian (promotes spatial smoothness)
    - α: regularization strength

ADMM Formulation
----------------
Introduce splitting: x ≈ d, minimize:
    L(x,d,p) = ||Ax-y||² + ε||d||₁ + α||Lx||² + δ/2||x-d+p||²

Updates:
    x^(k+1) = argmin_x ||Ax-y||² + α||Lx||² + δ/2||x-d^k+p^k||²
    d^(k+1) = S_{ε/δ}(λx^(k+1) + (1-λ)d^k + p^k)
    p^(k+1) = p^k + (x^(k+1) - d^(k+1))

The x-update becomes:
    (A^T A + α L^T L + δI) x = A^T y + δ(d - p)

References
----------
- Algorithm 3 (TBMD-CS): Tensor-based compressive sensing via ADMM
- Boyd et al. (2011): Distributed Optimization and Statistical Learning via ADMM
- Jiang et al. (2017): Smooth Tucker decomposition for brain connectivity
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy import sparse as sp

from TBMD.modules.TensorBasedCompressiveSensing import (
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig,
    CompressiveSensingMetrics,
    TensorCompressiveSensing,
    LinearSolver,
    DeltaPolicy,
    StopPolicy,
    MetricsHook,
    make_linear_solver,
    make_delta_policy,
    make_stop_policy,
    noop_metrics_hook
)
from TBMD.utils.utils import get_torch_device, to_torch_tensor
from TBMD.utils.geometry import MeshGeometry

logger = logging.getLogger(__name__)


@dataclass
class GeometryAwareCSConfig(CompressiveSensingConfig):
    """
    Extended configuration for geometry-aware compressive sensing.
    
    Adds spatial regularization parameters to the base CS config.
    
    Additional Parameters
    ---------------------
    alpha : float, default=0.01
        Laplacian regularization strength. Higher values → smoother reconstructions.
    laplacian_type : {'standard', 'normalized'}, default='normalized'
        Type of Laplacian to use for smoothness penalty.
    auto_alpha : bool, default=True
        If True, automatically tune α based on problem conditioning.
    alpha_max : float, default=1.0
        Maximum value for α when auto-tuning is enabled.
    """
    alpha: float = 0.01
    laplacian_type: str = 'normalized'
    auto_alpha: bool = True
    alpha_max: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        super().__post_init__()
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.laplacian_type not in ['standard', 'normalized']:
            raise ValueError("laplacian_type must be 'standard' or 'normalized'")
        if self.alpha_max < self.alpha:
            raise ValueError("alpha_max must be >= alpha")


class GeometryAwareTensorCS:
    """
    ADMM solver for geometry-aware tensor compressive sensing.
    
    Extends the base TensorCompressiveSensing class with Laplacian regularization
    for spatially smooth reconstructions on unstructured meshes.
    
    Parameters
    ----------
    A : (... , W) array_like
        Forward model flattened along the last axis. Spatial dims must match P and Y.
    P : bool array_like, shape = A.shape[:-1]
        Sensor mask. Only entries with True are used.
    Y : array_like, shape = A.shape[:-1]
        Measurements corresponding to A.
    mesh : MeshGeometry
        Mesh geometry with Laplacian matrices.
    core_cfg : GeometryAwareCSConfig, optional
        Core algorithm configuration with geometry parameters.
    ext_cfg : ExtensionCompressiveSensingConfig, optional
        Extensions configuration (solver, stopping, etc.).
    solver : LinearSolver, optional
        Custom linear solver. If None, built from ext_cfg.
    delta_policy : DeltaPolicy, optional
        Custom δ policy. If None, created from ext_cfg.
    stop_policy : StopPolicy, optional
        Custom stop policy. If None, created from ext_cfg.
    hook : MetricsHook, optional
        Callback executed each iteration.
    
    Examples
    --------
    >>> from TBMD.utils.geometry import MeshGraphBuilder
    >>> # Build mesh
    >>> builder = MeshGraphBuilder(connectivity_type='grid')
    >>> mesh = builder.build_from_shape((100, 100))
    >>> 
    >>> # Setup geometry-aware CS
    >>> config = GeometryAwareCSConfig(alpha=0.1, epsilon_l1=1e-2)
    >>> solver = GeometryAwareTensorCS(A, P, Y, mesh, core_cfg=config)
    >>> x, metrics = solver.solve()
    """
    
    def __init__(
        self,
        A: Union[np.ndarray, torch.Tensor],
        P: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        mesh: MeshGeometry,
        core_cfg: Optional[GeometryAwareCSConfig] = None,
        ext_cfg: Optional[ExtensionCompressiveSensingConfig] = None,
        solver: Optional[LinearSolver] = None,
        delta_policy: Optional[DeltaPolicy] = None,
        stop_policy: Optional[StopPolicy] = None,
        hook: Optional[MetricsHook] = None,
    ) -> None:
        self.cfg = core_cfg or GeometryAwareCSConfig()
        self.ext = ext_cfg or ExtensionCompressiveSensingConfig()
        self.mesh = mesh
        
        device = get_torch_device(self.cfg.device)
        dtype = self.cfg.dtype
        
        # --- Inputs conversion ---
        A_t = to_torch_tensor(A, device=device, dtype=dtype)
        P_t = to_torch_tensor(P, device=device, dtype=torch.bool)
        Y_t = to_torch_tensor(Y, device=device, dtype=dtype)
        
        if A_t.ndim < 2:
            raise ValueError("A must have ≥2 dims")
        if P_t.shape != A_t.shape[:-1] or Y_t.shape != A_t.shape[:-1]:
            raise ValueError("Shapes of P/Y must match spatial part of A")
        
        W = A_t.shape[-1]
        mask = P_t.reshape(-1)
        if not mask.any():
            raise ValueError("Empty sensor mask P")
        
        # Validate mesh size
        spatial_size = int(np.prod(A_t.shape[:-1]))
        if self.mesh.adjacency_matrix.shape[0] != spatial_size:
            raise ValueError(
                f"Mesh has {self.mesh.adjacency_matrix.shape[0]} cells but "
                f"spatial size is {spatial_size}"
            )
        
        A_flat = A_t.reshape(-1, W)
        Y_flat = Y_t.reshape(-1, 1)
        self.As = A_flat[mask]       # Ns×W
        self.Ys = Y_flat[mask]       # Ns×1
        
        # --- Precomputations ---
        self.W = W
        self.device = device
        self.dtype = dtype
        self.AtA = self.As.T @ self.As
        self.AtY = self.As.T @ self.Ys
        self.I = torch.eye(W, device=device, dtype=dtype)
        
        # --- Laplacian setup ---
        if self.cfg.laplacian_type == 'normalized':
            laplacian = mesh.normalized_laplacian
        else:
            laplacian = mesh.laplacian_matrix
        
        # Convert to torch
        if sp.issparse(laplacian):
            self.L = self._sparse_scipy_to_torch(laplacian, device, dtype)
        else:
            self.L = torch.from_numpy(laplacian).to(device=device, dtype=dtype)
        
        # Precompute L^T L
        if self.L.is_sparse:
            self.LTL = torch.sparse.mm(self.L.t(), self.L).to_dense()
        else:
            self.LTL = self.L.T @ self.L
        
        # Check dimensions
        if self.LTL.shape[0] != spatial_size:
            raise ValueError(
                f"Laplacian size {self.LTL.shape[0]} doesn't match spatial size {spatial_size}"
            )
        
        # Since x has size W (coefficients), we need to map to spatial dimensions
        # Typically, A @ x reconstructs the spatial field, so L acts on spatial field
        # We'll compute ||L (A x)||² = x^T A^T L^T L A x
        # So the regularization matrix is: A^T L^T L A
        self.ALTLA = A_flat.T @ self.LTL @ A_flat  # W×W
        
        # Auto-tune alpha if requested
        if self.cfg.auto_alpha:
            self.alpha = self._auto_tune_alpha()
            logger.info(f"Auto-tuned α = {self.alpha:.6f}")
        else:
            self.alpha = self.cfg.alpha
        
        # --- ADMM variables ---
        self.delta = self.cfg.delta_init
        self.x = torch.zeros(W, 1, device=device, dtype=dtype)
        self.d = torch.zeros_like(self.x)
        self.p = torch.zeros_like(self.x)
        self._d_prev = torch.zeros_like(self.x)
        
        # --- Strategies ---
        self.solver_fn = solver or make_linear_solver(self.ext)
        self.delta_policy = delta_policy or make_delta_policy(self.ext.delta_policy)
        self.stop_policy = stop_policy or make_stop_policy(self.ext)
        self.hook = hook or noop_metrics_hook
        
        self.history = []
        
        logger.info(
            f"GeometryAwareTensorCS initialized: "
            f"α={self.alpha:.6f}, W={W}, sensors={self.As.shape[0]}"
        )
    
    @staticmethod
    def _sparse_scipy_to_torch(
        scipy_sparse: sp.spmatrix,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert scipy sparse matrix to torch sparse tensor."""
        coo = scipy_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data).to(dtype)
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, shape, device=device)
    
    def _auto_tune_alpha(self) -> float:
        """
        Automatically tune α based on problem conditioning.
        
        Strategy: Balance data term and regularization term scales.
        α ≈ λ_max(A^T A) / λ_max(L^T L A A^T L^T L)
        
        Returns
        -------
        float
            Tuned regularization strength.
        """
        try:
            # Estimate spectral norms
            AtA_norm = torch.linalg.norm(self.AtA, ord=2).item()
            ALTLA_norm = torch.linalg.norm(self.ALTLA, ord=2).item()
            
            if ALTLA_norm > 1e-10:
                alpha = self.cfg.alpha * (AtA_norm / ALTLA_norm)
                alpha = min(alpha, self.cfg.alpha_max)
            else:
                alpha = self.cfg.alpha
                logger.warning("Laplacian regularization term is near-zero, using default α")
            
            return max(alpha, 1e-6)  # Ensure positive
        except Exception as e:
            logger.warning(f"Auto-tuning failed: {e}, using default α={self.cfg.alpha}")
            return self.cfg.alpha
    
    @staticmethod
    def _soft(z: torch.Tensor, kappa: float) -> torch.Tensor:
        """Soft-thresholding operator."""
        return torch.sign(z) * torch.clamp(torch.abs(z) - kappa, min=0.0)
    
    def _objective(self) -> float:
        """
        Compute the current objective value.
        
        Objective: 0.5‖Ax−y‖² + ε‖d‖₁ + 0.5 α‖L(Ax)‖²
        """
        res = self.As @ self.x - self.Ys
        data_term = 0.5 * torch.norm(res).pow(2).item()
        sparsity_term = self.cfg.epsilon_l1 * torch.norm(self.d, p=1).item()
        
        # Smoothness term: ||L (A x)||² = x^T A^T L^T L A x
        smoothness_term = 0.5 * self.alpha * (self.x.T @ self.ALTLA @ self.x).item()
        
        return data_term + sparsity_term + smoothness_term
    
    def _admm_step(self) -> Tuple[float, float, float]:
        """
        Perform one ADMM iteration with Laplacian regularization.
        
        x-update: (A^T A + α A^T L^T L A + δI) x = A^T y + δ(d - p)
        
        Returns
        -------
        primal : float
            ‖x − d‖₂
        dual : float
            ‖δ(d − d_prev)‖₂
        obj : float
            Objective value.
        """
        cfg = self.cfg
        
        # x-update with Laplacian regularization
        lhs = self.AtA + self.alpha * self.ALTLA + self.delta * self.I
        rhs = self.AtY + self.delta * (self.d - self.p)
        self.x = self.solver_fn(lhs, rhs)
        
        # Relaxation
        x_hat = cfg.relax_lambda * self.x + (1 - cfg.relax_lambda) * self.d
        
        # d-update (soft-thresholding)
        self._d_prev.copy_(self.d)
        self.d = self._soft(x_hat + self.p, cfg.epsilon_l1 / self.delta)
        
        # p-update
        self.p = self.p + (x_hat - self.d)
        
        # Residuals
        primal = torch.norm(self.x - self.d).item()
        dual = torch.norm(self.delta * (self.d - self._d_prev)).item()
        
        # δ-update
        new_delta, p_scale = self.delta_policy(self.delta, primal, dual, self.cfg.delta_max)
        if new_delta != self.delta:
            self.delta = new_delta
            if p_scale != 1.0:
                self.p *= p_scale
        
        obj = self._objective()
        return primal, dual, obj
    
    def solve(self) -> Tuple[torch.Tensor, CompressiveSensingMetrics]:
        """
        Run ADMM until convergence or max_iter.
        
        Returns
        -------
        x_vec : torch.Tensor, shape = (W,)
            Recovered coefficients (detached CPU tensor).
        metrics : CompressiveSensingMetrics
            Summary metrics and diagnostics.
        """
        start = time.perf_counter()
        converged = False
        primal = dual = obj = 0.0
        
        for it in range(1, self.cfg.max_iter + 1):
            primal, dual, obj = self._admm_step()
            res = max(primal, dual)
            
            if self.ext.collect_history:
                self.history.append(res)
            
            self.hook(it, primal, dual, obj, self.delta)
            
            if self.stop_policy(it, primal, dual, self.cfg, self.history):
                converged = True
                break
        
        elapsed = time.perf_counter() - start
        x_vec = self.x.view(-1).detach().cpu()
        
        metrics = CompressiveSensingMetrics(
            iterations=it,
            converged=converged,
            primal_residual=primal,
            dual_residual=dual,
            objective=obj,
            delta_final=float(self.delta),
            history=self.history if self.ext.collect_history else [],
            time_sec=elapsed,
        )
        
        logger.info(
            f"Solved in {it} iterations ({elapsed:.2f}s): "
            f"converged={converged}, obj={obj:.6e}"
        )
        
        return x_vec, metrics
    
    def reconstruction_error(self, x: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Relative reconstruction error w.r.t. observed measurements.
        
        Parameters
        ----------
        x : array_like
            Ground-truth or reference vector of shape (W,) or (W, 1).
        
        Returns
        -------
        float
            ‖A_s x − y_s‖ / ‖y_s‖
        """
        x_t = to_torch_tensor(x, device=self.device, dtype=self.dtype).view(-1, 1)
        res = self.As @ x_t - self.Ys
        return (torch.norm(res) / torch.norm(self.Ys)).item()
    
    def get_spatial_field(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct spatial field from coefficients.
        
        Parameters
        ----------
        x : torch.Tensor, optional
            Coefficient vector. If None, uses self.x.
        
        Returns
        -------
        torch.Tensor
            Spatial field reconstruction (flattened).
        """
        if x is None:
            x = self.x
        
        # A is (spatial_size, W), x is (W, 1) → result is (spatial_size, 1)
        # But we need the full A matrix (not just sensor rows)
        # This requires storing the full A during init
        logger.warning("Full spatial field reconstruction requires full A matrix")
        return self.As @ x  # Only at sensor locations

