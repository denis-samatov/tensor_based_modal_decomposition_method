"""
# Tensor-based Compressive Sensing (TBMD‑CS) — *revised implementation*
# ====================================================================
# Version : 2.1  (2025‑06‑26)
#
# This version integrates the existing utility helpers `get_torch_device` and
# `to_torch_tensor` from *TBMD.utils.utils* instead of redefining similar
# functions locally.
#
# Key points vs. v2.0
# -------------------
# 1. **Removed duplicate helpers** `_torch_device`, `_to_tensor` ➜ now using the
#    canonical versions exported by the TBMD utilities package.
# 2. **Minor typing touch‑ups** so that passing `dtype=None` forwards the default
#    chosen by `to_torch_tensor` (float32).
# 3. **Imports cleaned**; module header, doc‑strings, and copyright
#    notice retained.
#
# —————————————————————————————————————————————————————————————————————————
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from ..utils.utils import get_torch_device, to_torch_tensor

__all__ = [
    "CompressiveSensingConfig",
    "ExtensionCompressiveSensingConfig",
    "CompressiveSensingMetrics",
    "TensorCompressiveSensing",
    "LinearSolver",
    "DeltaPolicy",
    "StopPolicy",
    "MetricsHook",
]


# ------------------------------------------------------------------
# 1. Configs
# ------------------------------------------------------------------


@dataclass
class CompressiveSensingConfig:
    """Core hyperparameters for the TBMD-CS algorithm.

    Attributes:
        max_iter (int): The maximum number of ADMM iterations.
        tol (float): The termination threshold for the maximum of the primal
            and dual residuals.
        epsilon_l1 (float): The L1 shrinkage parameter for the soft-thresholding
            step.
        delta_init (float): The initial value of the ADMM penalty parameter.
        delta_max (float): The maximum cap for the penalty parameter.
        relax_lambda (float): The over-relaxation mixing coefficient for `x`
            and `d`. Must be in the range (0, 1).
        device (str): The torch device to use for computations.
        dtype (torch.dtype): The torch dtype for tensors in the algorithm.
    """

    max_iter: int = 1000
    tol: float = 1e-4  # stop criterion on max(primal, dual)
    epsilon_l1: float = 1e-2  # ε in (28)
    delta_init: float = 1.0  # δ₀
    delta_max: float = 1.0  # δ_max (36)
    relax_lambda: float = 0.95  # mixing x and d
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Validate parameter ranges right after dataclass construction."""
        if not (0 < self.relax_lambda < 1):
            raise ValueError("relax_lambda ∈ (0,1)")
        if self.max_iter <= 0:
            raise ValueError("max_iter > 0")
        if self.epsilon_l1 <= 0:
            raise ValueError("epsilon_l1 > 0")
        if self.delta_init <= 0 or self.delta_max <= 0:
            raise ValueError("delta values must be > 0")


@dataclass
class ExtensionCompressiveSensingConfig:
    """Configuration for extended features of the TBMD-CS algorithm.

    Attributes:
        solver (str): The linear system solver to use in the x-update. Can be
            'cholesky', 'direct', or 'svd'.
        reg (float): A small diagonal regularization value for numerical
            stability.
        delta_policy (str): The strategy for adapting the penalty parameter
            during iterations ('boyd' or 'cap_only').
        stop_policy (str): The termination rule ('residual', 'relative', or
            'both').
        relative_window (int): The window size for the relative stopping rule.
        relative_drop (float): The required relative decrease for the relative
            stopping rule.
        collect_history (bool): Whether to store residual history.
    """

    # Linear solver
    solver: str = "cholesky"  # cholesky | direct | svd
    reg: float = 1e-8  # diagonal regularization
    # δ policy
    delta_policy: str = "boyd"  # boyd | cap_only
    # Stop conditions
    stop_policy: str = "residual"  # residual | relative | both
    relative_window: int = 5  # window for relative criterion
    relative_drop: float = 1e-3  # required relative drop
    # Metrics/logging
    collect_history: bool = True


# ------------------------------------------------------------------
# 2. Strategy Protocols (interfaces)
# ------------------------------------------------------------------


class LinearSolver(Protocol):
    """Callable signature for linear solvers.

    Should solve (lhs) x = rhs and return x.
    """

    def __call__(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        ...


class DeltaPolicy(Protocol):
    """Callable signature for δ update strategies.

    Returns
    -------
    new_delta : float
        Updated δ.
    p_scale_factor : float
        Scaling factor applied to the dual variable p when δ changes.
    """

    def __call__(
        self, delta: float, primal: float, dual: float, delta_max: float
    ) -> Tuple[float, float]:
        ...


class StopPolicy(Protocol):
    """Callable signature for stopping policies."""

    def __call__(
        self,
        it: int,
        primal: float,
        dual: float,
        cfg: CompressiveSensingConfig,
        history: List[float],
    ) -> bool:
        ...


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

    def __call__(
        self, it: int, primal: float, dual: float, obj: float, delta: float
    ) -> None:
        ...


# ------------------------------------------------------------------
# 3. Default strategy implementations
# ------------------------------------------------------------------


def _svd_solve(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Pseudo‑inverse via SVD with thresholding of small singular values."""
    U, S, Vh = torch.linalg.svd(lhs, full_matrices=False)
    eps = torch.finfo(S.dtype).eps
    thresh = eps * max(lhs.shape) * S.max()
    S_inv = torch.where(S > thresh, S.reciprocal(), torch.zeros_like(S))
    return Vh.T @ (S_inv.unsqueeze(1) * (U.T @ rhs))


class CholeskyCachedSolver:
    """A linear solver that caches the Cholesky factorization.

    It updates the factorization only when `delta` changes.
    """

    def __init__(self, reg: float, svd_fallback_fn: Callable = _svd_solve):
        self.reg = reg
        self.svd_fallback = svd_fallback_fn
        self.L: Optional[torch.Tensor] = None
        self.AtA: Optional[torch.Tensor] = None
        self.delta: Optional[float] = None

    def update(self, AtA: torch.Tensor, delta: float) -> None:
        """Update the cached factorization for the new AtA and delta."""
        self.AtA = AtA
        self.delta = delta
        lhs_reg = AtA.clone()
        # Add to diagonal: delta * I + reg * I
        lhs_reg.diagonal().add_(delta + self.reg)
        try:
            self.L = torch.linalg.cholesky(lhs_reg)
        except torch.linalg.LinAlgError:
            self.L = None

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve for rhs using the cached factorization (or fallback)."""
        if self.L is not None:
            return torch.cholesky_solve(rhs, self.L, upper=False)
        else:
            # Fallback to SVD using stored AtA and delta
            if self.AtA is None or self.delta is None:
                raise RuntimeError("Solver not updated before solve")
            lhs_reg = self.AtA + (self.delta + self.reg) * torch.eye(
                self.AtA.shape[0], device=self.AtA.device, dtype=self.AtA.dtype
            )
            return self.svd_fallback(lhs_reg, rhs)

    def __call__(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Stateless solve (compatible with LinearSolver protocol)."""
        if not hasattr(self, "eye_cache"):
            self.eye_cache = {}

        key = (lhs.shape[0], lhs.device, lhs.dtype)
        if key not in self.eye_cache:
            self.eye_cache[key] = self.reg * torch.eye(key[0], device=key[1], dtype=key[2])

        lhs_reg = lhs + self.eye_cache[key]
        try:
            L = torch.linalg.cholesky(lhs_reg)
            return torch.cholesky_solve(rhs, L, upper=False)
        except torch.linalg.LinAlgError:
            return self.svd_fallback(lhs_reg, rhs)


def make_linear_solver(cfg: ExtensionCompressiveSensingConfig) -> LinearSolver:
    """Creates a factory for a `LinearSolver` based on `cfg.solver`.

    Args:
        cfg (ExtensionCompressiveSensingConfig): The extended configuration object.

    Returns:
        LinearSolver: A callable that solves (lhs)x = rhs. Falls back to SVD if
        the chosen solver fails with a `LinAlgError`.
    """
    reg = cfg.reg

    if cfg.solver == "cholesky":
        return CholeskyCachedSolver(reg)

    eye_cache = {}

    def direct(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Direct linear solve (LU) with regularization and SVD fallback."""
        key = (lhs.shape[0], lhs.device, lhs.dtype)
        if key not in eye_cache:
            eye_cache[key] = reg * torch.eye(key[0], device=key[1], dtype=key[2])

        lhs_reg = lhs + eye_cache[key]
        try:
            return torch.linalg.solve(lhs_reg, rhs)
        except torch.linalg.LinAlgError:
            return _svd_solve(lhs_reg, rhs)

    if cfg.solver == "direct":
        return direct

    # Fallback to SVD solver
    return _svd_solve


def make_delta_policy(name: str) -> DeltaPolicy:
    """Creates a factory for delta update rules.

    Args:
        name (str): The name of the policy. Can be 'boyd' or 'cap_only'.

    Returns:
        DeltaPolicy: A callable that implements the specified delta update rule.
    """
    if name == "boyd":

        def boyd(
            delta: float, primal: float, dual: float, delta_max: float
        ) -> Tuple[float, float]:
            if primal > 10 * dual:
                return min(delta * 2, delta_max), 0.5  # p /= 2
            if dual > 10 * primal:
                return max(delta / 2, 1e-12), 2.0  # p *= 2
            return min(delta, delta_max), 1.0

        return boyd
    else:  # cap_only

        def cap_only(delta: float, *_args) -> Tuple[float, float]:
            return delta, 1.0

        return cap_only


def make_stop_policy(ext_cfg: ExtensionCompressiveSensingConfig) -> StopPolicy:
    """Creates a factory for stopping rules based on `ext_cfg.stop_policy`.

    Args:
        ext_cfg (ExtensionCompressiveSensingConfig): The extended configuration
            object.

    Returns:
        StopPolicy: A callable that implements the specified stopping rule.
    """
    if ext_cfg.stop_policy == "residual":

        def residual_stop(
            it: int,
            primal: float,
            dual: float,
            core_cfg: CompressiveSensingConfig,
            history: List[float],
        ) -> bool:
            return max(primal, dual) < core_cfg.tol

        return residual_stop
    elif ext_cfg.stop_policy == "relative":

        def relative_stop(
            it: int,
            _p: float,
            _d: float,
            _cfg: CompressiveSensingConfig,
            history: List[float],
        ) -> bool:
            if it <= ext_cfg.relative_window:
                return False
            before = history[-ext_cfg.relative_window]
            now = history[-1]
            return (before - now) / max(before, 1e-12) < ext_cfg.relative_drop

        return relative_stop
    else:  # both
        residual = make_stop_policy(
            ExtensionCompressiveSensingConfig(stop_policy="residual")
        )
        relative = make_stop_policy(
            ExtensionCompressiveSensingConfig(
                stop_policy="relative",
                relative_window=ext_cfg.relative_window,
                relative_drop=ext_cfg.relative_drop,
            )
        )

        def both(
            it: int,
            p: float,
            d: float,
            cfg: CompressiveSensingConfig,
            history: List[float],
        ) -> bool:
            return residual(it, p, d, cfg, history) or relative(
                it, p, d, cfg, history
            )

        return both


def noop_metrics_hook(*_args, **_kwargs) -> None:
    """Default no-op metrics hook."""
    return None


# ------------------------------------------------------------------
# 4. Metrics container
# ------------------------------------------------------------------


@dataclass
class CompressiveSensingMetrics:
    """Summary statistics returned after `solve`.

    Attributes:
        iterations (int): The number of iterations performed.
        converged (bool): Whether the algorithm converged before reaching the
            maximum number of iterations.
        primal_residual (float): The final primal residual norm.
        dual_residual (float): The final dual residual norm.
        objective (float): The final objective value.
        delta_final (float): The final value of the penalty parameter.
        history (List[float]): The residual history, if collected.
        time_sec (float): The wall-clock time for the `solve` method, in seconds.
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
    """An ADMM-based solver for tensor compressive sensing.

    This class provides a flexible implementation of the TBMD-CS core algorithm,
    allowing for dependency injection of strategies for solving, updating
    parameters, and stopping.

    Args:
        A (Union[np.ndarray, torch.Tensor]): The forward model, flattened
            along the last axis.
        P (Union[np.ndarray, torch.Tensor]): The sensor mask, with `True` for
            active sensors.
        Y (Union[np.ndarray, torch.Tensor]): The measurements corresponding to
            `A`.
        core_cfg (Optional[CompressiveSensingConfig]): The core algorithm
            configuration.
        ext_cfg (Optional[ExtensionCompressiveSensingConfig]): The extended
            features configuration.
        solver (Optional[LinearSolver]): A custom linear solver. If `None`, a
            solver is created from `ext_cfg`.
        delta_policy (Optional[DeltaPolicy]): A custom delta policy. If
            `None`, a policy is created from `ext_cfg`.
        stop_policy (Optional[StopPolicy]): A custom stop policy. If `None`, a
            policy is created from `ext_cfg`.
        hook (Optional[MetricsHook]): A callback executed at each iteration.
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
        self.As = A_flat[mask]  # Ns×W
        self.Ys = Y_flat[mask]  # Ns×1

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
        self._cached_delta = -1.0

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
        return 0.5 * torch.norm(res).pow(2).item() + self.cfg.epsilon_l1 * torch.norm(
            self.d, p=1
        ).item()

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
        rhs = self.AtY + self.delta * (self.d - self.p)

        if hasattr(self.solver, "update") and hasattr(self.solver, "solve"):
            if self.delta != self._cached_delta:
                self.solver.update(self.AtA, self.delta)
                self._cached_delta = self.delta
            self.x = self.solver.solve(rhs)
        else:
            lhs = self.AtA + self.delta * self.I
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
        new_delta, p_scale = self.delta_policy(
            self.delta, primal, dual, self.cfg.delta_max
        )
        if new_delta != self.delta:
            self.delta = new_delta
            if p_scale != 1.0:
                self.p *= p_scale

        obj = self._objective()
        return primal, dual, obj

    # --- public API ------------------------------------------------
    def solve(self) -> Tuple[torch.Tensor, CompressiveSensingMetrics]:
        """Runs the ADMM algorithm until convergence or `max_iter`.

        Returns:
            Tuple[torch.Tensor, CompressiveSensingMetrics]: A tuple containing:
                - x_vec (torch.Tensor): The recovered coefficients, with shape (W,).
                - metrics (CompressiveSensingMetrics): Summary metrics and
                  diagnostics.
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
        """Calculates the relative reconstruction error.

        This method computes the error with respect to the observed
        measurements.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The ground-truth or reference
                vector, with shape (W,) or (W, 1).

        Returns:
            float: The relative reconstruction error.
        """
        x_t = to_torch_tensor(x, device=self.device, dtype=self.dtype).view(-1, 1)
        res = self.As @ x_t - self.Ys
        return (torch.norm(res) / torch.norm(self.Ys)).item()
