"""
Standard TBMD pipeline demonstration (no geometry awareness).

Pipeline:
 1. Perform Tucker decomposition with `TuckerDecomposerInterface`.
 2. Select informative sensors via tensor tube QR factorisation.
 3. Reconstruct unseen frames through tensor compressive sensing.
 4. Evaluate reconstruction quality with common metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.modules.TensorHOSVD import TuckerDecomposerInterface
from TBMD.modules.TensorBasedCompressiveSensing import (
    CompressiveSensingConfig,
    TensorCompressiveSensing,
)
from TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import (
    TensorTubeQRDecomposition,
)
from TBMD.core.metrics.metrics import compute_metrics


def make_synthetic_dataset(
    spatial_shape: Tuple[int, int] = (48, 48),
    timesteps: int = 120,
    seed: int = 0,
) -> np.ndarray:
    """Generate smooth temporal fields with propagating patterns."""
    rng = np.random.default_rng(seed)
    h, w = spatial_shape
    t = np.linspace(0.0, 2 * np.pi, timesteps)
    x = np.linspace(-2.5, 2.5, w)
    y = np.linspace(-2.5, 2.5, h)
    X, Y = np.meshgrid(x, y, indexing="ij")

    data = np.empty((h, w, timesteps), dtype=np.float32)
    for i, tau in enumerate(t):
        waves = np.sin(1.5 * X + tau) + 0.75 * np.cos(1.3 * Y - tau / 2)
        swirl = np.sin(np.sqrt(X**2 + Y**2) - 0.5 * tau)
        data[..., i] = 0.6 * waves + 0.4 * swirl
    data += 0.025 * rng.standard_normal(size=data.shape).astype(np.float32)
    return data


@dataclass
class StandardPipelineResult:
    reconstruction: np.ndarray
    ground_truth: np.ndarray
    sensor_mask: np.ndarray
    metrics: Tuple[float, float, float, float]


def run_standard_pipeline() -> StandardPipelineResult:
    ranks = [25, 18, 40]
    n_sensors = 32

    dataset = make_synthetic_dataset()
    train_tensor = dataset[..., :80]
    test_tensor = dataset[..., 80:]

    # Tucker decomposition
    decomposer = TuckerDecomposerInterface(
        tensors=train_tensor,
        ranks=ranks,
        epsilon=5e-3,
        random_state=42,
    )
    decomposer.decompose()
    spatial_factor = decomposer.factors[0].detach().cpu().numpy()  # (hw, r1)
    basis = spatial_factor.reshape(train_tensor.shape[0], train_tensor.shape[1], ranks[0])

    # Sensor placement via tensor QR
    qr = TensorTubeQRDecomposition(
        tensor=torch.from_numpy(basis),
        N=n_sensors,
        random_state=42,
        check_orthogonality=True,
        uniform_distribution=True,
    )
    P, _, _ = qr.factorize()
    sensor_mask = P.detach().cpu().numpy().astype(bool)

    # Compressive sensing on the test sequence
    reconstructed = np.zeros_like(test_tensor)
    cs_cfg = CompressiveSensingConfig(
        max_iter=500,
        epsilon_l1=8e-3,
        relax_lambda=0.96,
        delta_init=0.7,
        delta_max=2.0,
    )

    for tidx in range(test_tensor.shape[-1]):
        measurements = np.where(sensor_mask, test_tensor[..., tidx], 0.0)
        solver = TensorCompressiveSensing(
            basis,
            sensor_mask,
            measurements,
            core_cfg=cs_cfg,
        )
        coeffs, _ = solver.solve()
        coeff_np = coeffs.numpy()
        recon_slice = (basis.reshape(-1, basis.shape[-1]) @ coeff_np).reshape(basis.shape[:2])
        reconstructed[..., tidx] = recon_slice

    metrics = compute_metrics(reconstructed, test_tensor)
    return StandardPipelineResult(
        reconstruction=reconstructed,
        ground_truth=test_tensor,
        sensor_mask=sensor_mask,
        metrics=metrics,
    )


def visualise_result(result: StandardPipelineResult, time_index: int = 0) -> None:
    original = result.ground_truth[..., time_index]
    recon = result.reconstruction[..., time_index]
    error = np.abs(original - recon)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im0 = axes[0].imshow(original, cmap="viridis")
    axes[0].set_title("Original")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(recon, cmap="viridis")
    axes[1].set_title("Reconstructed")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(error, cmap="magma")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(original, cmap="gray", alpha=0.5)
    ys, xs = np.nonzero(result.sensor_mask)
    axes[3].scatter(xs, ys, c="lime", s=25, marker="x")
    axes[3].set_title(f"Sensors ({len(xs)})")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    result = run_standard_pipeline()
    err_norm, mse, ssim, psnr = result.metrics
    print("Standard TBMD reconstruction metrics:")
    print(f"  Normalised Frobenius error : {err_norm:.4e}")
    print(f"  Mean-squared error         : {mse:.4e}")
    print(f"  SSIM                       : {ssim:.4f}")
    print(f"  PSNR                       : {psnr:.2f} dB")

    visualise_result(result)


if __name__ == "__main__":
    main()
