"""
TBMD‑CS (Algorithm 3) — Core + Extensions
=========================================

Idea
----
The **core** strictly follows formulas (32–36); everything else is pushed into pluggable
strategies: linear solver choice, δ update policy, stopping policy, and optional logging/metrics.

Module Layout
-------------
- ``CoreCompressiveSensingConfig``  — minimal hyperparameters for the algorithm core.
- ``ExtensionCompressiveSensingConfig`` — convenient toggles for "extensions" (not part of the strict algorithm).
- ``LinearSolver`` API  — abstraction over solving (Aᵀ A + δI) x = rhs.
- ``DeltaPolicy`` API   — strategy for updating δ.
- ``StopPolicy`` API    — strategy for termination (tol, relative drop, etc.).
- ``MetricsHook``       — optional metric collection / logging callback.
- ``TensorCompressiveSensingCore`` — class implementing ADMM based only on the provided strategies.

Dependencies: ``torch``, ``numpy``, ``TBMD.utils.tbmd_utils`` (``get_torch_device``, ``to_torch_tensor``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union, Protocol
import time
import torch
import numpy as np

from TBMD.utils.tbmd_utils import get_torch_device, to_torch_tensor


# ------------------------------------------------------------------
# 1. Configs
# ------------------------------------------------------------------

from TBMD.config.reconstruction_config import CompressiveSensingConfig, ExtensionCompressiveSensingConfig


# ------------------------------------------------------------------
# 2. Strategy Protocols (interfaces)
# ------------------------------------------------------------------

class LinearSolver(Protocol):
    """Callable signature for linear solvers.

    Should solve (lhs) x = rhs and return x.
    """

    def __call__(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor: ...


class DeltaPolicy(Protocol):
    """Callable signature for δ update strategies.

    Returns
    -------
    new_delta : float
        Updated δ.
    p_scale_factor : float
        Scaling factor applied to the dual variable p when δ changes.
    """

    def __call__(self, delta: float, primal: float, dual: float, delta_max: float) -> Tuple[float, float]: ...


class StopPolicy(Protocol):
    """Callable signature for stopping policies."""

    def __call__(self, it: int, primal: float, dual: float,
                 cfg: CompressiveSensingConfig,
                 history: List[float]) -> bool: ...


class MetricsHook(Protocol):
    """User‑provided callback to log metrics each iteration.

    Parameters
    ----------
    it : int
        Iteration number (1‑based).
    primal : float
        Primal residual norm.
    dual : float
        Dual residual norm.
    obj : float
        Objective value at this iteration.
    delta : float
        Current δ value.
    """

    def __call__(self, it: int, primal: float, dual: float, obj: float, delta: float) -> None: ...


# ------------------------------------------------------------------
# 3. Default strategy implementations
# ------------------------------------------------------------------

def make_linear_solver(cfg: ExtensionCompressiveSensingConfig) -> LinearSolver:
    """Factory for a ``LinearSolver`` based on ``cfg.solver``.

    Returns a callable that solves (lhs)x = rhs. Falls back to SVD if the chosen
    solver fails with a ``LinAlgError``.
    """
    reg = cfg.reg

    def cholesky(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Cholesky factorization with diagonal regularization and SVD fallback."""
        lhs_reg = lhs + reg * torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)
        try:
            L = torch.linalg.cholesky(lhs_reg)
            return torch.cholesky_solve(rhs, L, upper=False)
        except torch.linalg.LinAlgError:
            return svd(lhs_reg, rhs)

    def direct(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Direct linear solve (LU) with regularization and SVD fallback."""
        lhs_reg = lhs + reg * torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)
        try:
            return torch.linalg.solve(lhs_reg, rhs)
        except torch.linalg.LinAlgError:
            return svd(lhs_reg, rhs)

    def svd(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Pseudo‑inverse via SVD with thresholding of small singular values."""
        U, S, Vh = torch.linalg.svd(lhs, full_matrices=False)
        eps = torch.finfo(S.dtype).eps
        thresh = eps * max(lhs.shape) * S.max()
        S_inv = torch.where(S > thresh, S.reciprocal(), torch.zeros_like(S))
        return Vh.T @ (S_inv.unsqueeze(1) * (U.T @ rhs))

    return {"cholesky": cholesky, "direct": direct, "svd": svd}[cfg.solver]


def make_delta_policy(name: str) -> DeltaPolicy:
    """Factory for δ update rules.

    Available
    ---------
    "boyd"
        If primal >> dual, increase δ and scale p downward. If dual >> primal, decrease δ and scale p upward.
    "cap_only"
        Keep δ unchanged (only cap is enforced externally).
    """
    if name == "boyd":
        def boyd(delta: float, primal: float, dual: float, delta_max: float) -> Tuple[float, float]:
            if primal > 10 * dual:
                return min(delta * 2, delta_max), 0.5  # p /= 2
            if dual > 10 * primal:
                return max(delta / 2, 1e-12), 2.0      # p *= 2
            return min(delta, delta_max), 1.0
        return boyd
    else:  # cap_only
        def cap_only(delta: float, *_args) -> Tuple[float, float]:
            return delta, 1.0
        return cap_only


def make_stop_policy(ext_cfg: ExtensionCompressiveSensingConfig) -> StopPolicy:
    """Factory for stopping rules based on ``ext_cfg.stop_policy``."""
    if ext_cfg.stop_policy == "residual":
        def residual_stop(it: int, primal: float, dual: float,
                          core_cfg: CompressiveSensingConfig,
                          history: List[float]) -> bool:
            return max(primal, dual) < core_cfg.tol
        return residual_stop
    elif ext_cfg.stop_policy == "relative":
        def relative_stop(it: int, _p: float, _d: float,
                          _cfg: CompressiveSensingConfig,
                          history: List[float]) -> bool:
            # Require history collection for relative stopping
            if not history or len(history) < ext_cfg.relative_window:
                return False
            if it <= ext_cfg.relative_window:
                return False
            before = history[-ext_cfg.relative_window]
            now = history[-1]
            return (before - now) / max(before, 1e-12) < ext_cfg.relative_drop
        return relative_stop
    else:  # both
        residual = make_stop_policy(ExtensionCompressiveSensingConfig(stop_policy="residual"))
        relative = make_stop_policy(ExtensionCompressiveSensingConfig(stop_policy="relative",
                                                                      relative_window=ext_cfg.relative_window,
                                                                      relative_drop=ext_cfg.relative_drop))

        def both(it: int, p: float, d: float,
                 cfg: CompressiveSensingConfig,
                 history: List[float]) -> bool:
            return residual(it, p, d, cfg, history) or relative(it, p, d, cfg, history)
        return both


def noop_metrics_hook(*_args, **_kwargs) -> None:
    """Default no-op metrics hook."""
    return None


# ------------------------------------------------------------------
# 4. Metrics container
# ------------------------------------------------------------------

@dataclass
class CompressiveSensingMetrics:
    """Summary statistics returned after ``solve``.

    Attributes
    ----------
    iterations : int
        Number of iterations actually performed.
    converged : bool
        Whether a stopping policy triggered before ``max_iter``.
    primal_residual : float
        Final primal residual norm.
    dual_residual : float
        Final dual residual norm.
    objective : float
        Final objective value.
    delta_final : float
        Final δ value.
    history : list[float]
        Residual history if ``collect_history`` is True; empty otherwise.
    time_sec : float
        Wall-clock time for ``solve`` in seconds.
    """

    iterations: int
    converged: bool
    primal_residual: float
    dual_residual: float
    objective: float
    delta_final: float
    history: List[float]
    time_sec: float


# ------------------------------------------------------------------
# 5. Algorithm Core
# ------------------------------------------------------------------

class TensorCompressiveSensing:
    """ADMM-based solver for tensor compressive sensing (TBMD‑CS core).

    The class is agnostic to most implementation details through dependency
    injection of strategies (solver, δ policy, stopping policy, hooks).

    Parameters
    ----------
    A : (… , W) array_like
        Forward model flattened along the last axis. Spatial dims must match ``P`` and ``Y``.
    P : bool array_like, shape = A.shape[:-1]
        Sensor mask. Only entries with ``True`` are used.
    Y : array_like, shape = A.shape[:-1]
        Measurements corresponding to A.
    core_cfg : CoreCompressiveSensingConfig, optional
        Core algorithm configuration.
    ext_cfg : ExtensionCompressiveSensingConfig, optional
        Extensions configuration.
    solver : LinearSolver, optional
        Custom linear solver. If ``None``, a solver is built from ``ext_cfg``.
    delta_policy : DeltaPolicy, optional
        Custom δ policy. If ``None``, created from ``ext_cfg``.
    stop_policy : StopPolicy, optional
        Custom stop policy. If ``None``, created from ``ext_cfg``.
    hook : MetricsHook, optional
        Callback executed each iteration.
    """

    def __init__(
        self,
        A: Union[np.ndarray, torch.Tensor],
        P: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        core_cfg: Optional[CompressiveSensingConfig] = None,
        ext_cfg: Optional[ExtensionCompressiveSensingConfig] = None,
        solver: Optional[LinearSolver] = None,
        delta_policy: Optional[DeltaPolicy] = None,
        stop_policy: Optional[StopPolicy] = None,
        hook: Optional[MetricsHook] = None,
    ) -> None:
        self.cfg = core_cfg or CompressiveSensingConfig()
        self.ext = ext_cfg or ExtensionCompressiveSensingConfig()
        device = get_torch_device(self.cfg.device)
        dtype = self.cfg.dtype

        # --- inputs conversion ---
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

        A_flat = A_t.reshape(-1, W)
        Y_flat = Y_t.reshape(-1, 1)
        self.As = A_flat[mask]       # Ns×W
        self.Ys = Y_flat[mask]       # Ns×1

        # --- precomputations ---
        self.W = W
        self.device = device
        self.dtype = dtype
        self.AtA = self.As.T @ self.As
        self.AtY = self.As.T @ self.Ys
        self.I = torch.eye(W, device=device, dtype=dtype)

        # --- ADMM variables ---
        self.delta = self.cfg.delta_init
        self.x = torch.zeros(W, 1, device=device, dtype=dtype)
        self.d = torch.zeros_like(self.x)
        self.p = torch.zeros_like(self.x)
        self._d_prev = torch.zeros_like(self.x)

        # --- strategies ---
        self.solver = solver or make_linear_solver(self.ext)
        self.delta_policy = delta_policy or make_delta_policy(self.ext.delta_policy)
        self.stop_policy = stop_policy or make_stop_policy(self.ext)
        self.hook = hook or noop_metrics_hook

        self.history: List[float] = []

    def reset(self) -> None:
        """Reset ADMM variables for a fresh solve.
        
        Call this method before solve() if you want to re-run the algorithm
        from scratch on the same data.
        """
        self.delta = self.cfg.delta_init
        self.x = torch.zeros(self.W, 1, device=self.device, dtype=self.dtype)
        self.d = torch.zeros_like(self.x)
        self.p = torch.zeros_like(self.x)
        self._d_prev = torch.zeros_like(self.x)
        self.history.clear()

    # --- helpers ---------------------------------------------------
    @staticmethod
    def _soft(z: torch.Tensor, kappa: float) -> torch.Tensor:
        """Soft-thresholding operator.

        Parameters
        ----------
        z : torch.Tensor
            Input vector.
        kappa : float
            Threshold level.

        Returns
        -------
        torch.Tensor
            Result of ``sign(z) * max(|z| - kappa, 0)``.
        """
        return torch.sign(z) * torch.clamp(torch.abs(z) - kappa, min=0.0)

    def _objective(self) -> float:
        """Compute the current objective value.

        Objective: 0.5‖Ax−y‖² + ε‖d‖₁
        """
        res = self.As @ self.x - self.Ys
        return 0.5 * torch.norm(res).pow(2).item() + self.cfg.epsilon_l1 * torch.norm(self.d, p=1).item()

    def _admm_step(self) -> Tuple[float, float, float]:
        """Perform one ADMM iteration and return residuals & objective.

        Returns
        -------
        primal : float
            ‖x − d‖₂
        dual : float
            ‖δ(d − d_prev)‖₂
        obj : float
            Objective value at the end of the iteration.
        """
        cfg = self.cfg
        # x‑update (32)
        lhs = self.AtA + self.delta * self.I
        rhs = self.AtY + self.delta * (self.d - self.p)
        self.x = self.solver(lhs, rhs)

        # relaxation
        x_hat = cfg.relax_lambda * self.x + (1 - cfg.relax_lambda) * self.d

        # d‑update (33)
        self._d_prev.copy_(self.d)
        self.d = self._soft(x_hat + self.p, cfg.epsilon_l1 / self.delta)

        # p‑update (34)
        self.p = self.p + (x_hat - self.d)

        # residuals
        primal = torch.norm(self.x - self.d).item()
        dual = torch.norm(self.delta * (self.d - self._d_prev)).item()

        # δ‑update
        new_delta, p_scale = self.delta_policy(self.delta, primal, dual, self.cfg.delta_max)
        if new_delta != self.delta:
            self.delta = new_delta
            if p_scale != 1.0:
                self.p *= p_scale

        obj = self._objective()
        return primal, dual, obj

    # --- public API ------------------------------------------------
    def solve(self) -> Tuple[torch.Tensor, CompressiveSensingMetrics]:
        """Run ADMM until convergence or ``max_iter``.

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
        return x_vec, metrics

    def reconstruction_error(self, x: Union[np.ndarray, torch.Tensor]) -> float:
        """Relative reconstruction error w.r.t. the observed measurements.

        Parameters
        ----------
        x : array_like
            Ground-truth or reference vector of shape (W,) or (W, 1).

        Returns
        -------
        float
            ‖A_s x − y_s‖ / ‖y_s‖, where the subscript ``s`` denotes rows selected by mask ``P``.
        """
        x_t = to_torch_tensor(x, device=self.device, dtype=self.dtype).view(-1, 1)
        res = self.As @ x_t - self.Ys
        return (torch.norm(res) / torch.norm(self.Ys)).item()


# Alias for backward compatibility
TensorBasedCompressiveSensing = TensorCompressiveSensing
TensorCSReconstructor = TensorCompressiveSensing
TensorCSConfig = CompressiveSensingConfig
