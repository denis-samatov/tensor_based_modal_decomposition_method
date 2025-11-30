"""
Tensor-based Compressive Sensing (Algorithm 3) demo.

The revised `TensorCompressiveSensing` implementation exposes a clear API
based on two configuration dataclasses:
    * `CompressiveSensingConfig` – core ADMM hyper-parameters.
    * `ExtensionCompressiveSensingConfig` – numerical options & logging.

This example script covers:
  1. Solving a sparse recovery problem with default settings.
  2. Inspecting solver metrics and residual history.
  3. Performing a lightweight grid-search over two hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.modules.TensorBasedCompressiveSensing import (
    CompressiveSensingConfig,
    ExtensionCompressiveSensingConfig,
    TensorCompressiveSensing,
)


# ---------------------------------------------------------------------------
# Synthetic problem generator
# ---------------------------------------------------------------------------

@dataclass
class SyntheticProblem:
    A: np.ndarray          # (H, W, modes)
    mask: np.ndarray       # (H, W) boolean sensor mask
    measurements: np.ndarray  # (H, W) masked field
    ground_truth: np.ndarray  # (modes,)
    full_field: np.ndarray    # (H, W)


def build_problem(
    height: int = 32,
    width: int = 32,
    modes: int = 40,
    sparsity: int = 8,
    sensor_ratio: float = 0.25,
    noise_std: float = 0.0,
    seed: int = 7,
) -> SyntheticProblem:
    """Generate a reproducible compressive-sensing test case."""
    rng = np.random.default_rng(seed)

    # Create smooth spatial basis using low-frequency Fourier components
    x = np.linspace(-1.0, 1.0, height)
    y = np.linspace(-1.0, 1.0, width)
    X, Y = np.meshgrid(x, y, indexing="ij")
    A = np.zeros((height, width, modes), dtype=np.float32)
    frequencies = rng.uniform(0.5, 3.0, size=modes)
    angles = rng.uniform(0, 2 * np.pi, size=modes)
    for k in range(modes):
        A[..., k] = np.sin(frequencies[k] * X + angles[k]) * np.cos(
            0.5 * frequencies[k] * Y - angles[k] / 2
        )

    # Draw sparse coefficients
    coeff = np.zeros(modes, dtype=np.float32)
    support = rng.choice(modes, size=min(sparsity, modes), replace=False)
    coeff[support] = rng.normal(0, 1.0, size=len(support)).astype(np.float32)

    # Full field (A @ coeff)
    full_field = np.tensordot(A, coeff, axes=([2], [0]))

    # Add optional noise
    if noise_std > 0:
        full_field = full_field + noise_std * rng.normal(0, 1.0, size=full_field.shape)

    # Sensor mask
    mask = rng.random(size=(height, width)) < sensor_ratio
    if mask.sum() < modes:
        # Ensure at least `modes` observations for a well-posed solve
        additional = rng.choice(height * width, size=modes - mask.sum(), replace=False)
        mask.reshape(-1)[additional] = True

    measurements = np.where(mask, full_field, 0.0)

    return SyntheticProblem(
        A=A,
        mask=mask.astype(bool),
        measurements=measurements,
        ground_truth=coeff,
        full_field=full_field,
    )


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def solve_basic(problem: SyntheticProblem) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Solve with default settings and report basic metrics."""
    solver = TensorCompressiveSensing(
        problem.A,
        problem.mask,
        problem.measurements,
    )
    x_hat, metrics = solver.solve()

    rel_coeff_error = np.linalg.norm(x_hat.numpy() - problem.ground_truth) / np.linalg.norm(
        problem.ground_truth
    )
    recon = (problem.A.reshape(-1, problem.A.shape[-1]) @ x_hat.numpy()).reshape(
        problem.A.shape[:2]
    )
    rel_field_error = np.linalg.norm(recon - problem.full_field) / np.linalg.norm(problem.full_field)

    report = {
        "coeff_rel_error": rel_coeff_error,
        "field_rel_error": rel_field_error,
        "iterations": metrics.iterations,
        "converged": float(metrics.converged),
        "objective": metrics.objective,
    }
    return x_hat, report


def solve_with_history(problem: SyntheticProblem) -> Tuple[torch.Tensor, Dict[str, float], list[float]]:
    """Collect residual history for plotting."""
    core_cfg = CompressiveSensingConfig(
        max_iter=800,
        epsilon_l1=5e-3,
        delta_init=0.5,
        delta_max=2.0,
        relax_lambda=0.98,
    )
    ext_cfg = ExtensionCompressiveSensingConfig(
        solver="cholesky",
        delta_policy="boyd",
        stop_policy="both",
        relative_window=8,
        relative_drop=5e-4,
        collect_history=True,
    )
    solver = TensorCompressiveSensing(
        problem.A,
        problem.mask,
        problem.measurements,
        core_cfg=core_cfg,
        ext_cfg=ext_cfg,
    )
    x_hat, metrics = solver.solve()

    rel_coeff_error = np.linalg.norm(x_hat.numpy() - problem.ground_truth) / np.linalg.norm(
        problem.ground_truth
    )
    info = {
        "coeff_rel_error": rel_coeff_error,
        "iterations": metrics.iterations,
        "primal_residual": metrics.primal_residual,
        "dual_residual": metrics.dual_residual,
        "objective": metrics.objective,
        "delta_final": metrics.delta_final,
        "converged": float(metrics.converged),
    }
    return x_hat, info, metrics.history


def grid_search(
    problem: SyntheticProblem,
    epsilons: Iterable[float],
    relaxations: Iterable[float],
) -> Tuple[Dict[str, float], Dict[Tuple[float, float], float]]:
    """Simple two-parameter grid-search over (epsilon_l1, relax_lambda)."""
    best_error = float("inf")
    best_cfg: Dict[str, float] = {}
    all_results: Dict[Tuple[float, float], float] = {}

    for eps in epsilons:
        for relax in relaxations:
            cfg = CompressiveSensingConfig(
                epsilon_l1=eps,
                relax_lambda=relax,
                max_iter=700,
            )
            solver = TensorCompressiveSensing(
                problem.A,
                problem.mask,
                problem.measurements,
                core_cfg=cfg,
            )
            x_hat, metrics = solver.solve()
            rel_error = np.linalg.norm(x_hat.numpy() - problem.ground_truth) / np.linalg.norm(
                problem.ground_truth
            )
            key = (eps, relax)
            all_results[key] = rel_error
            if rel_error < best_error and metrics.converged:
                best_error = rel_error
                best_cfg = {"epsilon_l1": eps, "relax_lambda": relax, "rel_error": rel_error}

    return best_cfg, all_results


def plot_history(history: list[float]) -> None:
    if not history:
        print("No residual history collected.")
        return
    plt.figure(figsize=(6, 4))
    plt.semilogy(history, marker="o", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("max(primal, dual)")
    plt.title("ADMM Residual History")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    problem = build_problem(sensor_ratio=0.3, noise_std=0.01)

    print("=" * 70)
    print("Demo 1 – default configuration")
    print("=" * 70)
    _, summary = solve_basic(problem)
    for key, value in summary.items():
        print(f"{key:>20s}: {value:.4e}" if isinstance(value, float) else f"{key:>20s}: {value}")

    print("\n" + "=" * 70)
    print("Demo 2 – custom configuration with residual history")
    print("=" * 70)
    _, info, history = solve_with_history(problem)
    for key, value in info.items():
        print(f"{key:>20s}: {value:.4e}" if isinstance(value, float) else f"{key:>20s}: {value}")
    plot_history(history)

    print("\n" + "=" * 70)
    print("Demo 3 – grid search")
    print("=" * 70)
    best_cfg, results = grid_search(problem, epsilons=[5e-3, 1e-2, 2e-2], relaxations=[0.9, 0.95, 0.99])
    print(f"Best configuration: epsilon_l1={best_cfg.get('epsilon_l1')}, "
          f"relax_lambda={best_cfg.get('relax_lambda')}, "
          f"relative error={best_cfg.get('rel_error'):.4f}")
    worst = max(results.values())
    print(f"Search space size: {len(results)}, worst error: {worst:.4f}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
