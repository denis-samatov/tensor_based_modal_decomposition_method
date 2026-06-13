"""
Geometry-aware TBMD pipeline demonstration.

Stages:
 1. Generate synthetic flow-like data on a 2-D grid.
 2. Build a mesh graph and run Laplacian-regularised Tucker decomposition.
 3. Place sensors via geometry-aware QR pivoting.
 4. Reconstruct unseen frames through tensor compressive sensing.
 5. Report reconstruction metrics and visualise sensor placement.
"""

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.core.decomposition import (
    GeometryAwareDecompositionConfig as GeometryAwareConfig,
)
from TBMD.core.decomposition import (
    GeometryAwareTuckerDecomposer,
)
from TBMD.core.geometry import MeshGraphBuilder
from TBMD.core.reconstruction import (
    CompressiveSensingConfig,
    TensorCompressiveSensing,
)
from TBMD.core.sensor_placement import (
    GeometricQRConfig,
    GeometryAwareTensorQR,
)
from TBMD.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def make_vortex_dataset(
    spatial_shape: Tuple[int, int] = (50, 50),
    timesteps: int = 120,
    noise_std: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Create a smooth, time-varying 2-D field."""
    rng = np.random.default_rng(seed)
    h, w = spatial_shape
    t = np.linspace(0.0, 2.0 * np.pi, timesteps)
    x = np.linspace(-2.0, 2.0, w)
    y = np.linspace(-2.0, 2.0, h)
    X, Y = np.meshgrid(x, y, indexing="ij")

    data = np.empty((h, w, timesteps), dtype=np.float32)
    for i, tau in enumerate(t):
        vortex = np.sin(np.sqrt(X**2 + Y**2) - tau)
        front = np.tanh(np.cos(tau) * X + np.sin(tau) * Y)
        data[..., i] = 0.6 * vortex + 0.4 * front

    data += noise_std * rng.standard_normal(size=data.shape).astype(np.float32)
    return data


@dataclass
class GeometryAwareResult:
    reconstruction: np.ndarray
    test_tensor: np.ndarray
    sensor_mask: np.ndarray
    metrics: Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_geometry_aware_pipeline() -> GeometryAwareResult:
    spatial_shape = (50, 50)
    ranks = (25, 15, 40)  # (spatial, spatial, temporal)
    n_sensors = 35

    dataset = make_vortex_dataset(spatial_shape, timesteps=140)
    train_tensor = dataset[..., :90]
    test_tensor = dataset[..., 90:]

    # Build mesh
    builder = MeshGraphBuilder(connectivity_type="grid")
    mesh = builder.build_from_shape(spatial_shape)

    # Geometry-aware decomposition
    hosvd = GeometryAwareTuckerDecomposer(
        tensor=train_tensor,
        mesh=mesh,
        geo_config=GeometryAwareConfig(alpha=0.1, spatial_modes=[0]),
        ranks=ranks,
        epsilon=1e-3,
        max_iter=60,
    )
    hosvd.decompose()
    spatial_factor = hosvd.factors[0].detach().cpu().numpy()  # (hw, r1)

    # Reshape spatial factor back to grid for QR & CS
    A_basis = spatial_factor.reshape(spatial_shape + (ranks[0],))

    # Geometry-aware QR
    qr = GeometryAwareTensorQR(
        tensor=A_basis,
        mesh=mesh,
        N=n_sensors,
        field_data=train_tensor,
        config=GeometricQRConfig(
            gradient_weight=0.7,
            proximity_weight=1.2,
            distribution_weight=0.4,
            min_distance_factor=1.5,
            adaptive_weights=True,
        ),
        random_state=42,
        device="cpu",
    )
    P, _, _ = qr.factorize()
    sensor_mask = P.detach().cpu().numpy().astype(bool)

    # Compressive sensing on the held-out frames
    reconstructed = np.zeros_like(test_tensor)
    cs_cfg = CompressiveSensingConfig(
        max_iter=600,
        epsilon_l1=5e-3,
        delta_init=0.5,
        delta_max=1.5,
        relax_lambda=0.97,
    )
    A_tensor = A_basis  # (H, W, r1)

    for idx in range(test_tensor.shape[-1]):
        measurements = np.where(sensor_mask, test_tensor[..., idx], 0.0)
        solver = TensorCompressiveSensing(
            A_tensor,
            sensor_mask,
            measurements,
            core_cfg=cs_cfg,
        )
        coeffs, _metrics = solver.solve()
        coeff_np = coeffs.numpy()
        recon_slice = (A_tensor.reshape(-1, A_tensor.shape[-1]) @ coeff_np).reshape(spatial_shape)
        reconstructed[..., idx] = recon_slice

    # Metrics
    err_norm, mse, ssim, psnr = compute_metrics(reconstructed, test_tensor)

    return GeometryAwareResult(
        reconstruction=reconstructed,
        test_tensor=test_tensor,
        sensor_mask=sensor_mask,
        metrics=(err_norm, mse, ssim, psnr),
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def show_results(result: GeometryAwareResult, time_index: int = 0) -> None:
    original = result.test_tensor[..., time_index]
    reconstructed = result.reconstruction[..., time_index]
    error = np.abs(original - reconstructed)
    sensors = result.sensor_mask

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im0 = axes[0].imshow(original, cmap="viridis")
    axes[0].set_title("Original")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(reconstructed, cmap="viridis")
    axes[1].set_title("Reconstructed")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(error, cmap="Reds")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(original, cmap="gray", alpha=0.5)
    ys, xs = np.nonzero(sensors)
    axes[3].scatter(xs, ys, c="red", s=30, marker="x")
    axes[3].set_title(f"{len(xs)} Sensors")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    result = run_geometry_aware_pipeline()
    err_norm, mse, ssim, psnr = result.metrics
    print("Geometry-aware TBMD reconstruction metrics:")
    print(f"  Normalised Frobenius error : {err_norm:.4e}")
    print(f"  Mean-squared error         : {mse:.4e}")
    print(f"  SSIM                       : {ssim:.4f}")
    print(f"  PSNR                       : {psnr:.2f} dB")

    show_results(result, time_index=0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
