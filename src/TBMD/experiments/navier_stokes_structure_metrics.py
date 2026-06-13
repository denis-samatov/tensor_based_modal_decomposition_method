"""Structure-aware diagnostics for Navier-Stokes one-step predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_frame_batch(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        return array[None]
    if array.ndim != 3:
        raise ValueError("Expected a frame `(H,W)` or frame batch `(N,H,W)`")
    return array


def _safe_rel_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.linalg.norm(numerator) / max(np.linalg.norm(denominator), 1e-12))


def periodic_gradient(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward finite-difference gradients with periodic boundaries."""
    batch = _as_frame_batch(frames)
    grad_x = np.roll(batch, -1, axis=1) - batch
    grad_y = np.roll(batch, -1, axis=2) - batch
    return grad_x, grad_y


def periodic_laplacian(frames: np.ndarray) -> np.ndarray:
    """Compute a 2D five-point periodic Laplacian."""
    batch = _as_frame_batch(frames)
    return (
        np.roll(batch, 1, axis=1)
        + np.roll(batch, -1, axis=1)
        + np.roll(batch, 1, axis=2)
        + np.roll(batch, -1, axis=2)
        - 4.0 * batch
    )


def radial_power_spectrum(frames: np.ndarray) -> np.ndarray:
    """Return the mean radial FFT power spectrum for a frame batch."""
    batch = _as_frame_batch(frames)
    height, width = batch.shape[1:]
    spectrum = np.fft.fftshift(np.fft.fft2(batch, axes=(1, 2)), axes=(1, 2))
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    yy, xx = np.indices((height, width))
    cy = height // 2
    cx = width // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    max_radius = int(radius.max())
    radial = np.zeros(max_radius + 1, dtype=np.float64)
    counts = np.zeros(max_radius + 1, dtype=np.float64)
    np.add.at(radial, radius.ravel(), power.ravel())
    np.add.at(counts, radius.ravel(), 1.0)
    return radial / np.maximum(counts, 1.0)


def high_frequency_energy_fraction(
    frames: np.ndarray,
    *,
    cutoff_fraction: float = 0.5,
) -> float:
    """Return fraction of FFT power at radii above `cutoff_fraction * max_radius`."""
    if not 0.0 <= cutoff_fraction <= 1.0:
        raise ValueError("cutoff_fraction must be in [0, 1]")
    batch = _as_frame_batch(frames)
    height, width = batch.shape[1:]
    spectrum = np.fft.fftshift(np.fft.fft2(batch, axes=(1, 2)), axes=(1, 2))
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    yy, xx = np.indices((height, width))
    cy = height // 2
    cx = width // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = float(cutoff_fraction) * float(radius.max())
    high_power = float(np.sum(power[radius >= cutoff]))
    total_power = float(np.sum(power))
    return high_power / max(total_power, 1e-12)


def _spatial_correlation(target: np.ndarray, pred: np.ndarray) -> float:
    target_flat = target.reshape(target.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_centered = target_flat - target_flat.mean(axis=1, keepdims=True)
    pred_centered = pred_flat - pred_flat.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(target_centered, axis=1) * np.linalg.norm(pred_centered, axis=1)
    corr = np.sum(target_centered * pred_centered, axis=1) / np.maximum(denom, 1e-12)
    return float(np.mean(corr))


def compute_structure_metrics(
    target_frames: np.ndarray,
    pred_frames: np.ndarray,
    *,
    high_freq_cutoff_fraction: float = 0.5,
) -> dict[str, float]:
    """Compute pixel and structure-aware metrics for t+1 field predictions."""
    target = _as_frame_batch(target_frames)
    pred = _as_frame_batch(pred_frames)
    if target.shape != pred.shape:
        raise ValueError("target_frames and pred_frames must have the same shape")

    diff = pred - target
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    rel_frob = _safe_rel_norm(diff, target)
    target_flat = target.reshape(target.shape[0], -1)
    diff_flat = diff.reshape(diff.shape[0], -1)
    ss_res = float(np.sum(diff_flat**2))
    ss_tot = float(np.sum((target_flat - target_flat.mean(axis=1, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    target_gx, target_gy = periodic_gradient(target)
    pred_gx, pred_gy = periodic_gradient(pred)
    grad_target = np.concatenate([target_gx.reshape(target.shape[0], -1), target_gy.reshape(target.shape[0], -1)], axis=1)
    grad_diff = np.concatenate([(pred_gx - target_gx).reshape(target.shape[0], -1), (pred_gy - target_gy).reshape(target.shape[0], -1)], axis=1)
    grad_rmse = float(np.sqrt(np.mean(grad_diff**2)))
    grad_rel = _safe_rel_norm(grad_diff, grad_target)

    target_lap = periodic_laplacian(target)
    pred_lap = periodic_laplacian(pred)
    lap_diff = pred_lap - target_lap
    lap_rmse = float(np.sqrt(np.mean(lap_diff**2)))
    lap_rel = _safe_rel_norm(lap_diff, target_lap)

    target_spectrum = radial_power_spectrum(target)
    pred_spectrum = radial_power_spectrum(pred)
    spectrum_rel = _safe_rel_norm(pred_spectrum - target_spectrum, target_spectrum)
    target_hf = high_frequency_energy_fraction(
        target,
        cutoff_fraction=high_freq_cutoff_fraction,
    )
    pred_hf = high_frequency_energy_fraction(
        pred,
        cutoff_fraction=high_freq_cutoff_fraction,
    )
    hf_rel = float(abs(pred_hf - target_hf) / max(abs(target_hf), 1e-12))

    abs_error = np.abs(diff)
    spatial_corr = _spatial_correlation(target, pred)
    mean_error = float(abs(float(np.mean(pred)) - float(np.mean(target))))
    std_error = float(abs(float(np.std(pred)) - float(np.std(target))))
    structure_score = float(
        0.35 * rel_frob
        + 0.25 * grad_rel
        + 0.20 * spectrum_rel
        + 0.10 * min(hf_rel, 10.0)
        + 0.10 * float(np.percentile(abs_error, 95)) / max(float(np.std(target)), 1e-12)
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rel_frob_err": rel_frob,
        "gradient_rmse": grad_rmse,
        "gradient_rel_frob_err": grad_rel,
        "laplacian_rmse": lap_rmse,
        "laplacian_rel_frob_err": lap_rel,
        "radial_spectrum_rel_err": spectrum_rel,
        "target_high_frequency_energy_fraction": float(target_hf),
        "pred_high_frequency_energy_fraction": float(pred_hf),
        "high_frequency_energy_rel_err": hf_rel,
        "spatial_corr": spatial_corr,
        "max_abs_error": float(np.max(abs_error)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "mean_error_abs": mean_error,
        "std_error_abs": std_error,
        "structure_score": structure_score,
    }


def aggregate_structure_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame structure metric rows."""
    if not rows:
        raise ValueError("rows must not be empty")
    higher_is_worse_suffixes = (
        "_err",
        "_error",
        "_rmse",
        "_mae",
        "_mse",
    )
    higher_is_worse_keys = {
        "mse",
        "rmse",
        "mae",
        "structure_score",
        "max_abs_error",
        "p95_abs_error",
        "mean_error_abs",
        "std_error_abs",
    }
    metric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key not in {"flat_index", "trajectory_index", "start_index", "target_index"}
    ]
    summary: dict[str, Any] = {"n_frames": len(rows)}
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_p90"] = float(np.percentile(values, 90))
        is_error_like = key in higher_is_worse_keys or key.endswith(higher_is_worse_suffixes)
        summary[f"{key}_worst"] = float(np.max(values) if is_error_like else np.min(values))
    r2_values = np.asarray([row.get("r2", np.nan) for row in rows], dtype=np.float64)
    summary["fraction_r2_below_0_8"] = float(np.mean(r2_values < 0.8))
    return summary


__all__ = [
    "aggregate_structure_rows",
    "compute_structure_metrics",
    "high_frequency_energy_fraction",
    "periodic_gradient",
    "periodic_laplacian",
    "radial_power_spectrum",
]
