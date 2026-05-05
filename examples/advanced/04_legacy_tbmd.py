#!/usr/bin/env python3
"""
Canonical TBMD (Tucker/HOSVD) usage examples.

This script demonstrates how to work with the refactored
`TuckerDecomposerInterface`:

1. Decompose and reconstruct a single tensor.
2. Process a collection of tensors in parallel.
3. Handle common validation/states errors gracefully.
4. Benchmark the threaded CPU strategy via `max_workers`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.modules.TensorHOSVD import (
    TuckerDecomposerInterface,
    TensorDecompositionError,
    InvalidRankError,
    StateError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_everything(seed: int = 42) -> None:
    """Seed NumPy and PyTorch RNGs for deterministic examples."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sample_tensor(
    shape: Tuple[int, int, int], noise: float = 0.0, seed: int = 42
) -> torch.Tensor:
    """Create a smooth 3-D tensor with optional Gaussian noise."""
    _seed_everything(seed)
    h, w, t = shape

    x = np.linspace(-1.0, 1.0, h)
    y = np.linspace(-1.0, 1.0, w)
    X, Y = np.meshgrid(x, y, indexing="ij")

    tensor = np.zeros(shape, dtype=np.float32)
    for k in range(t):
        phase = 2.0 * np.pi * k / t
        tensor[..., k] = (
            np.sin(np.pi * X + phase) * np.cos(np.pi * Y - 0.5 * phase)
            + 0.25 * X
            - 0.15 * Y
        )

    if noise > 0:
        tensor += noise * np.random.randn(*shape).astype(np.float32)

    return torch.from_numpy(tensor)


def plot_middle_slice(original: torch.Tensor, reconstructed: torch.Tensor) -> None:
    """Visualise a central slice of the original vs reconstructed tensor."""
    idx = original.shape[-1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original[..., idx], cmap="viridis")
    axes[0].set_title("Original (middle slice)")
    axes[1].imshow(reconstructed[..., idx], cmap="viridis")
    axes[1].set_title("Reconstructed (middle slice)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Example workflows
# ---------------------------------------------------------------------------

def single_tensor_demo() -> None:
    """Decompose and reconstruct a single tensor."""
    print("=" * 60)
    print("Single Tensor Demo")
    print("=" * 60)

    tensor = make_sample_tensor((40, 30, 25), noise=0.05)
    print(f"Tensor shape: {tuple(tensor.shape)}")

    decomposer = TuckerDecomposerInterface(
        tensors=tensor,
        ranks=[20, 15, 10],
        epsilon=1e-3,
        device="cpu",
        random_state=42,
    )

    decomposer.decompose()
    decomposer.reconstruct()

    error = float(decomposer.reconstruction_errors)
    print(f"Relative reconstruction error: {error:.4e}")

    reconstructed = decomposer.reconstructed_tensors
    plot_middle_slice(tensor, reconstructed)


def collection_demo() -> None:
    """Process a small collection of tensors with shared ranks."""
    print("=" * 60)
    print("Collection Demo")
    print("=" * 60)

    tensors: Dict[str, torch.Tensor] = {
        f"subject_{i:02d}": make_sample_tensor((30, 20, 18), seed=100 + i)
        for i in range(3)
    }
    decomposer = TuckerDecomposerInterface(
        tensors=tensors,
        ranks=12,  # uniform rank for all modes
        epsilon=5e-3,
        device="cpu",
        max_workers=2,
        random_state=123,
    )
    decomposer.decompose()
    decomposer.reconstruct()

    errors = decomposer.reconstruction_errors
    for name, err in errors.items():
        print(f"{name}: relative error = {err:.4e}")


def error_handling_demo() -> None:
    """Showcase validation / state errors raised by the interface."""
    print("=" * 60)
    print("Error Handling Demo")
    print("=" * 60)

    tensor = make_sample_tensor((10, 8, 6))

    try:
        TuckerDecomposerInterface(tensor, ranks=[-1, 5, 5])
    except InvalidRankError as exc:
        print(f"✔️  Caught invalid rank error: {exc}")

    try:
        decomposer = TuckerDecomposerInterface(tensor, ranks=5)
        _ = decomposer.cores  # accessing before decomposition
    except StateError as exc:
        print(f"✔️  Caught state error: {exc}")

    try:
        TuckerDecomposerInterface(tensor, epsilon=0.0)
    except ValidationError as exc:
        print(f"✔️  Caught epsilon validation error: {exc}")


@dataclass
class TimingResult:
    workers: int
    seconds: float


def performance_demo() -> None:
    """Benchmark CPUStrategy with different thread counts."""
    print("=" * 60)
    print("Performance Demo")
    print("=" * 60)

    tensors = {
        f"sample_{i:02d}": make_sample_tensor((35, 35, 25), seed=200 + i)
        for i in range(6)
    }

    results = []
    for workers in (1, 2, 4):
        start = time.perf_counter()
        decomposer = TuckerDecomposerInterface(
            tensors=tensors,
            ranks=[18, 18, 12],
            epsilon=1e-3,
            device="cpu",
            max_workers=workers,
        )
        decomposer.decompose()
        decomposer.reconstruct()
        elapsed = time.perf_counter() - start
        results.append(TimingResult(workers, elapsed))

    for result in results:
        print(f"{result.workers} worker(s): {result.seconds:.2f} s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        single_tensor_demo()
        collection_demo()
        error_handling_demo()
        performance_demo()
    except TensorDecompositionError as exc:
        print(f"Tensor decomposition failed: {exc}")


if __name__ == "__main__":
    main()
