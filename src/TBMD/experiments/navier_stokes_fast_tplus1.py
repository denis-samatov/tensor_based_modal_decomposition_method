"""Reusable fast t+1 Navier-Stokes forecaster using windowed TBMD + QR sensors."""

from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from TBMD.config import DecompositionConfig, SensorPlacementConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposerInterface
from TBMD.core.sensor_placement import TensorTubeQRDecomposition
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics


@dataclass
class FastWindowedTBMDQRCSConfig:
    """Configuration for `FastWindowedTBMDQRCSForecaster`."""

    history_length: int = 7
    ranks: list[int] | tuple[int, int, int, int] = (8, 32, 32, 300)
    n_spatial_sensors: int = 300
    segment_stride: int = 1
    max_train_segments: Optional[int] = 6144
    sensor_rcond: float = 1e-6
    sensor_decoder: str = "lstsq"
    decoder_ridge_lambda: float = 1e-4
    decoder_l1_lambda: float = 1e-3
    decoder_max_iter: int = 25
    decoder_tol: float = 1e-6
    coefficient_calibration_type: str = "none"
    coefficient_calibration_target: str = "target"
    coefficient_calibration_alpha: float = 1e-6
    coefficient_calibration_blend: float = 1.0
    coefficient_calibration_rcond: float = 1e-6
    coefficient_calibration_innovation_rank: int = 0
    coefficient_calibration_include_norms: bool = False
    coefficient_temporal_smoothing_alpha: float = 0.0
    coefficient_temporal_reset_on_gap: bool = True
    correction_alpha: float = 1e-8
    correction_scale: float = 1.0
    correction_hf_scale: float = 0.4
    correction_residual_rank: Optional[int] = None
    correction_residual_target: str = "field"
    correction_highpass_cutoff_fraction: float = 0.35
    correction_residual_weighting: str = "uniform"
    correction_residual_weight_floor: float = 0.1
    correction_sample_weighting: str = "uniform"
    correction_sample_weight_power: float = 1.0
    correction_sample_weight_floor: float = 0.25
    correction_sample_weight_clip: float = 4.0
    correction_patch_size: int = 16
    correction_patch_residual_rank: Optional[int] = None
    correction_gate_type: str = "none"
    correction_gate_threshold: float = 1.25
    correction_gate_strength: float = 1.0
    correction_gate_min: float = 0.5
    correction_innovation_rank: int = 0
    correction_innovation_include_norms: bool = False
    correction_head_type: str = "ridge"
    correction_hidden_size: int = 128
    correction_num_epochs: int = 120
    correction_batch_size: int = 256
    correction_learning_rate: float = 1e-3
    correction_weight_decay: float = 1e-6
    random_state: int = 0
    spatial_mean_centering: bool = True
    dtype: str = "float32"


def build_forecast_segment_tensor(
    states: np.ndarray,
    *,
    history_length: int,
    stride: int,
    max_segments: int | None,
) -> np.ndarray:
    """Build causal forecasting segments as `(history+1,H,W,N_segments)`."""
    segments, _ = build_forecast_segment_tensor_with_refs(
        states,
        history_length=history_length,
        stride=stride,
        max_segments=max_segments,
    )
    return segments


def build_forecast_segment_tensor_with_refs(
    states: np.ndarray,
    *,
    history_length: int,
    stride: int,
    max_segments: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal forecast segments and `(trajectory,start)` references."""
    series = np.asarray(states, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("states must have shape `(B,T,H,W)`")
    if history_length <= 0 or stride <= 0:
        raise ValueError("history_length and stride must be positive")
    segment_length = history_length + 1
    if series.shape[1] < segment_length:
        raise ValueError("history_length + 1 exceeds trajectory length")

    refs = [
        (traj_idx, start)
        for traj_idx in range(series.shape[0])
        for start in range(0, series.shape[1] - segment_length + 1, stride)
    ]
    if max_segments is not None and len(refs) > max_segments:
        selected = np.linspace(0, len(refs) - 1, num=max_segments, dtype=int)
        refs = [refs[idx] for idx in selected]

    segments = np.stack(
        [series[traj_idx, start : start + segment_length] for traj_idx, start in refs],
        axis=-1,
    )
    return segments, np.asarray(refs, dtype=np.int64)


def smooth_coefficients_by_segment_refs(
    coefficients: np.ndarray,
    refs: np.ndarray,
    *,
    alpha: float,
    reset_on_gap: bool = True,
) -> np.ndarray:
    """Causally smooth coefficient rows within each trajectory.

    `alpha` is the carry-over weight: `0` keeps the current coefficient unchanged,
    while larger values mix in the previous smoothed coefficient from the same
    contiguous trajectory stream.
    """
    coeffs = np.asarray(coefficients, dtype=np.float64)
    refs_arr = np.asarray(refs, dtype=np.int64)
    if coeffs.ndim != 2:
        raise ValueError("coefficients must have shape `(N,R)`")
    if refs_arr.shape != (coeffs.shape[0], 2):
        raise ValueError("refs must have shape `(N,2)` and match coefficients")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if coeffs.shape[0] == 0 or alpha == 0.0:
        return coeffs.copy()

    smoothed = np.empty_like(coeffs)
    previous_traj: int | None = None
    previous_start: int | None = None
    previous_value: np.ndarray | None = None
    current_weight = 1.0 - alpha
    for row_idx, ((traj_idx, start), coeff_row) in enumerate(zip(refs_arr, coeffs, strict=True)):
        contiguous = previous_traj == int(traj_idx)
        if reset_on_gap and previous_start is not None:
            contiguous = contiguous and int(start) == previous_start + 1
        if contiguous and previous_value is not None:
            smoothed[row_idx] = current_weight * coeff_row + alpha * previous_value
        else:
            smoothed[row_idx] = coeff_row
        previous_traj = int(traj_idx)
        previous_start = int(start)
        previous_value = smoothed[row_idx]
    return smoothed


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_segment_dictionary_from_tucker(
    core: np.ndarray | torch.Tensor,
    factors: list[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """Convert Tucker decomposition of `(S,H,W,N)` segments to `(S,H,W,R)` modes."""
    core_np = _as_numpy(core)
    factor_np = [_as_numpy(factor) for factor in factors]
    if core_np.ndim != 4 or len(factor_np) != 4:
        raise ValueError("Expected 4D core and four factors for `(S,H,W,N)`")
    u_tau, u_x, u_y, _ = factor_np
    return np.einsum("ta,xb,yc,abcq->txyq", u_tau, u_x, u_y, core_np, optimize=True)


def fit_segment_dictionary(
    train_segments: np.ndarray,
    *,
    ranks: list[int] | tuple[int, int, int, int],
    random_state: int,
    dtype: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = DecompositionConfig(
        ranks=list(ranks),
        epsilon=1e-5,
        random_state=random_state,
        device="cpu",
        dtype=dtype,
        verbose=False,
        max_workers=1,
    )
    decomposer = TuckerDecomposerInterface(train_segments.astype(np.float32), config=config)
    decomposer.decompose()
    decomposer.reconstruct()
    dictionary = compute_segment_dictionary_from_tucker(
        decomposer.core_tensor,
        decomposer.factors,
    )
    return dictionary.astype(np.float64), {
        "core_shape": list(_as_numpy(decomposer.core_tensor).shape),
        "factor_shapes": [list(_as_numpy(factor).shape) for factor in decomposer.factors],
        "segment_reconstruction_rel_frob": float(decomposer.reconstruction_errors),
    }


def history_and_target(dictionary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if dictionary.shape[0] < 2:
        raise ValueError("dictionary must include history and target slices")
    return dictionary[:-1], dictionary[-1]


def augment_spatial_mask_by_leverage(
    qr_tensor: np.ndarray,
    spatial_mask: np.ndarray,
    *,
    target_sensors: int,
) -> np.ndarray:
    flat_mask = spatial_mask.reshape(-1).copy()
    n_extra = min(target_sensors - int(flat_mask.sum()), flat_mask.size - int(flat_mask.sum()))
    if n_extra <= 0:
        return flat_mask.reshape(spatial_mask.shape)
    leverage = np.sum(qr_tensor.reshape(-1, qr_tensor.shape[-1]) ** 2, axis=1)
    leverage[flat_mask] = -np.inf
    extra_indices = np.argpartition(-leverage, n_extra - 1)[:n_extra]
    extra_indices = extra_indices[np.argsort(-leverage[extra_indices])]
    flat_mask[extra_indices] = True
    return flat_mask.reshape(spatial_mask.shape)


def place_fixed_spatial_sensors(
    history_dictionary: np.ndarray,
    *,
    n_spatial_sensors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select fixed spatial sensors using all history-time dictionary tubes."""
    if n_spatial_sensors <= 0:
        raise ValueError("n_spatial_sensors must be positive")
    qr_tensor = history_dictionary.transpose(1, 2, 0, 3).reshape(
        history_dictionary.shape[1],
        history_dictionary.shape[2],
        history_dictionary.shape[0] * history_dictionary.shape[-1],
    )
    core_sensor_count = min(n_spatial_sensors, qr_tensor.shape[-1])
    config = SensorPlacementConfig(
        n_sensors=core_sensor_count,
        random_state=random_state,
        verbose=False,
        dtype="float64",
    )
    qr = TensorTubeQRDecomposition(
        qr_tensor,
        N=core_sensor_count,
        config=config,
        dtype=torch.float64,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        mask, _, _ = qr.factorize()
    spatial_mask = mask.detach().cpu().numpy().astype(bool)
    if n_spatial_sensors > int(spatial_mask.sum()):
        spatial_mask = augment_spatial_mask_by_leverage(
            qr_tensor,
            spatial_mask,
            target_sensors=n_spatial_sensors,
        )
    return spatial_mask, np.flatnonzero(spatial_mask.reshape(-1))


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def fft_highpass_frames(
    frames: np.ndarray,
    *,
    cutoff_fraction: float = 0.35,
) -> np.ndarray:
    """Return periodic FFT high-pass component for `(N,H,W)` frames."""
    if not 0.0 <= cutoff_fraction <= 1.0:
        raise ValueError("cutoff_fraction must be in [0, 1]")
    array = np.asarray(frames, dtype=np.float64)
    squeeze = False
    if array.ndim == 2:
        array = array[None]
        squeeze = True
    if array.ndim != 3:
        raise ValueError("frames must have shape `(H,W)` or `(N,H,W)`")
    height, width = array.shape[1:]
    freq_y = np.fft.fftfreq(height)
    freq_x = np.fft.fftfreq(width)
    radius = np.sqrt(freq_y[:, None] ** 2 + freq_x[None, :] ** 2)
    cutoff = float(cutoff_fraction) * float(np.max(radius))
    mask = radius >= cutoff
    spectrum = np.fft.fft2(array, axes=(1, 2))
    highpass = np.fft.ifft2(spectrum * mask[None], axes=(1, 2)).real
    return highpass[0] if squeeze else highpass


def residual_target_frames(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    *,
    residual_target: str = "field",
    highpass_cutoff_fraction: float = 0.35,
) -> np.ndarray:
    """Build the residual field that the correction head should learn."""
    target = np.asarray(target_frames, dtype=np.float64)
    base = np.asarray(base_predictions, dtype=np.float64)
    if target.shape != base.shape:
        raise ValueError("target_frames and base_predictions must have the same shape")
    residual = target - base
    residual_target = residual_target.lower()
    if residual_target == "field":
        return residual
    if residual_target == "highpass":
        return fft_highpass_frames(
            residual,
            cutoff_fraction=highpass_cutoff_fraction,
        )
    raise ValueError(f"Unknown residual_target: {residual_target}")


def history_sensor_matrix(
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
) -> np.ndarray:
    """Build the fixed QR sensing matrix for all history slices."""
    history_dictionary, _ = history_and_target(dictionary)
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history_dictionary = history_dictionary.reshape(
        history_dictionary.shape[0],
        height_width,
        history_dictionary.shape[-1],
    )
    return flat_history_dictionary[:, spatial_sensor_indices, :].reshape(
        -1,
        history_dictionary.shape[-1],
    )


def history_sensor_measurements(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
) -> np.ndarray:
    """Extract fixed QR measurements from centered history `(N,L,H,W)`."""
    history_dictionary, _ = history_and_target(dictionary)
    if history.shape[1:] != history_dictionary.shape[:-1]:
        raise ValueError("history shape must match dictionary history slices")
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history = history.transpose(1, 2, 3, 0).reshape(
        history_dictionary.shape[0],
        height_width,
        history.shape[0],
    )
    return flat_history[:, spatial_sensor_indices, :].reshape(-1, history.shape[0]).T


def fit_sensor_coefficient_decoder(
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    decoder: str,
    rcond: float = 1e-6,
    ridge_lambda: float = 1e-4,
    l1_lambda: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Precompute a batched QR-sensor coefficient decoder."""
    sensing_matrix = history_sensor_matrix(dictionary, spatial_sensor_indices)
    decoder = decoder.lower()
    if decoder == "lstsq":
        decoder_matrix = np.linalg.pinv(sensing_matrix, rcond=rcond).T
        return {
            "type": "lstsq",
            "decoder_matrix": decoder_matrix,
            "sensing_matrix": sensing_matrix,
            "rcond": float(rcond),
        }
    if decoder == "ridge":
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        gram = sensing_matrix.T @ sensing_matrix
        penalty = ridge_lambda * np.eye(gram.shape[0], dtype=np.float64)
        rhs = sensing_matrix.T
        try:
            decoder_matrix = np.linalg.solve(gram + penalty, rhs).T
        except np.linalg.LinAlgError:
            decoder_matrix = (np.linalg.pinv(gram + penalty) @ rhs).T
        return {
            "type": "ridge",
            "decoder_matrix": decoder_matrix,
            "sensing_matrix": sensing_matrix,
            "ridge_lambda": float(ridge_lambda),
        }
    if decoder == "fista":
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        spectral_norm = float(np.linalg.norm(sensing_matrix, ord=2))
        lipschitz = max(spectral_norm * spectral_norm, 1e-12)
        return {
            "type": "fista",
            "sensing_matrix": sensing_matrix,
            "lipschitz": lipschitz,
            "l1_lambda": float(l1_lambda),
            "max_iter": int(max_iter),
            "tol": float(tol),
        }
    raise ValueError(f"Unknown sensor decoder: {decoder}")


def decode_sensor_coefficients(
    measurements: np.ndarray,
    decoder_payload: dict[str, Any],
) -> np.ndarray:
    """Decode a measurement batch to TBMD window coefficients."""
    decoder_type = decoder_payload["type"]
    batch = np.asarray(measurements, dtype=np.float64)
    if decoder_type in {"lstsq", "ridge"}:
        return batch @ np.asarray(decoder_payload["decoder_matrix"], dtype=np.float64)
    if decoder_type == "fista":
        sensing_matrix = np.asarray(decoder_payload["sensing_matrix"], dtype=np.float64)
        coeffs = np.zeros((batch.shape[0], sensing_matrix.shape[1]), dtype=np.float64)
        momentum_state = coeffs.copy()
        momentum = 1.0
        lipschitz = float(decoder_payload["lipschitz"])
        threshold = float(decoder_payload["l1_lambda"]) / lipschitz
        tol = float(decoder_payload.get("tol", 1e-6))
        for _ in range(int(decoder_payload["max_iter"])):
            residual = momentum_state @ sensing_matrix.T - batch
            gradient = residual @ sensing_matrix
            next_coeffs = _soft_threshold(momentum_state - gradient / lipschitz, threshold)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            momentum_state = next_coeffs + ((momentum - 1.0) / next_momentum) * (
                next_coeffs - coeffs
            )
            rel_change = np.linalg.norm(next_coeffs - coeffs) / max(
                np.linalg.norm(coeffs),
                1e-12,
            )
            coeffs = next_coeffs
            momentum = next_momentum
            if tol > 0.0 and rel_change < tol:
                break
        return coeffs
    raise ValueError(f"Unknown sensor decoder payload type: {decoder_type}")


def sensor_innovation_residual(
    measurements: np.ndarray,
    coeffs: np.ndarray,
    decoder_payload: dict[str, Any],
) -> np.ndarray:
    """Return QR sensor consistency residual `y - A c` for recovered coefficients."""
    sensing_matrix = np.asarray(decoder_payload["sensing_matrix"], dtype=np.float64)
    return (
        np.asarray(measurements, dtype=np.float64)
        - np.asarray(coeffs, dtype=np.float64) @ sensing_matrix.T
    )


def fit_sensor_innovation_encoder(
    measurements: np.ndarray,
    coeffs: np.ndarray,
    decoder_payload: dict[str, Any],
    *,
    rank: int,
    include_norms: bool,
    random_state: int,
) -> dict[str, Any]:
    """Fit a compact feature encoder for sensor innovation residuals."""
    if rank < 0:
        raise ValueError("correction_innovation_rank must be non-negative")
    innovation = sensor_innovation_residual(measurements, coeffs, decoder_payload)
    innovation_mean = innovation.mean(axis=0)
    centered = innovation - innovation_mean
    actual_rank = min(int(rank), centered.shape[0], centered.shape[1])
    basis = np.empty((0, centered.shape[1]), dtype=np.float64)
    projected = np.empty((centered.shape[0], 0), dtype=np.float64)
    if actual_rank > 0:
        _, _, vt = randomized_svd(
            centered,
            n_components=actual_rank,
            n_iter=4,
            random_state=random_state,
        )
        basis = vt.astype(np.float64)
        projected = centered @ basis.T
    pieces = [projected]
    if include_norms:
        innovation_rms = np.sqrt(np.mean(innovation**2, axis=1, keepdims=True))
        measurement_rms = np.sqrt(
            np.mean(np.asarray(measurements, dtype=np.float64) ** 2, axis=1, keepdims=True)
        )
        relative_rms = innovation_rms / np.maximum(measurement_rms, 1e-12)
        pieces.append(np.concatenate([innovation_rms, relative_rms], axis=1))
    features = np.concatenate(pieces, axis=1) if pieces else np.empty((innovation.shape[0], 0))
    feature_mean = features.mean(axis=0) if features.shape[1] else np.empty((0,), dtype=np.float64)
    feature_std = features.std(axis=0) if features.shape[1] else np.empty((0,), dtype=np.float64)
    feature_std[feature_std < 1e-8] = 1.0
    return {
        "innovation_rank": int(actual_rank),
        "innovation_include_norms": bool(include_norms),
        "innovation_mean": innovation_mean,
        "innovation_basis": basis,
        "innovation_feature_mean": feature_mean,
        "innovation_feature_std": feature_std,
    }


def transform_sensor_innovation_features(
    measurements: np.ndarray,
    coeffs: np.ndarray,
    decoder_payload: dict[str, Any],
    encoder: dict[str, Any],
) -> np.ndarray:
    """Transform sensor innovation residuals to standardized correction features."""
    rank = int(encoder.get("innovation_rank", 0))
    include_norms = bool(encoder.get("innovation_include_norms", False))
    if rank <= 0 and not include_norms:
        return np.empty((np.asarray(coeffs).shape[0], 0), dtype=np.float64)
    innovation = sensor_innovation_residual(measurements, coeffs, decoder_payload)
    centered = innovation - np.asarray(encoder["innovation_mean"], dtype=np.float64)
    pieces = []
    if rank > 0:
        pieces.append(centered @ np.asarray(encoder["innovation_basis"], dtype=np.float64).T)
    if include_norms:
        innovation_rms = np.sqrt(np.mean(innovation**2, axis=1, keepdims=True))
        measurement_rms = np.sqrt(
            np.mean(np.asarray(measurements, dtype=np.float64) ** 2, axis=1, keepdims=True)
        )
        relative_rms = innovation_rms / np.maximum(measurement_rms, 1e-12)
        pieces.append(np.concatenate([innovation_rms, relative_rms], axis=1))
    features = np.concatenate(pieces, axis=1) if pieces else np.empty((innovation.shape[0], 0))
    return (
        features - np.asarray(encoder["innovation_feature_mean"], dtype=np.float64)
    ) / np.asarray(encoder["innovation_feature_std"], dtype=np.float64)


def build_correction_feature_matrix(
    coeffs: np.ndarray,
    corrector: dict[str, Any],
    *,
    measurements: np.ndarray | None = None,
    decoder_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build residual-head features from coefficients and optional sensor innovation."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    encoder = corrector.get("innovation_encoder")
    if not encoder:
        return coeffs
    if measurements is None or decoder_payload is None:
        raise ValueError("measurements and decoder_payload are required for innovation features")
    innovation_features = transform_sensor_innovation_features(
        measurements,
        coeffs,
        decoder_payload,
        encoder,
    )
    return np.concatenate([coeffs, innovation_features], axis=1)


def predict_from_history_sensor_decoder(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    decoder_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict centered next frames using a precomputed sensor decoder."""
    _, target_dictionary = history_and_target(dictionary)
    measurements = history_sensor_measurements(history, dictionary, spatial_sensor_indices)
    coeffs = decode_sensor_coefficients(measurements, decoder_payload)
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs


def predict_from_history_sensor_decoder_with_measurements(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    decoder_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict centered next frames and return sensor measurements used for decoding."""
    _, target_dictionary = history_and_target(dictionary)
    measurements = history_sensor_measurements(history, dictionary, spatial_sensor_indices)
    coeffs = decode_sensor_coefficients(measurements, decoder_payload)
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs, measurements


def predict_next_sensor_decoder(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    decoder_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    history = segments[:-1].transpose(3, 0, 1, 2)
    return predict_from_history_sensor_decoder(
        history,
        dictionary,
        spatial_sensor_indices,
        decoder_payload,
    )


def predict_next_sensor_decoder_with_measurements(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    decoder_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = segments[:-1].transpose(3, 0, 1, 2)
    return predict_from_history_sensor_decoder_with_measurements(
        history,
        dictionary,
        spatial_sensor_indices,
        decoder_payload,
    )


def predict_from_history_sensor_lstsq(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict centered next frames from centered history shaped `(N,L,H,W)`."""
    history_dictionary, target_dictionary = history_and_target(dictionary)
    if history.shape[1:] != history_dictionary.shape[:-1]:
        raise ValueError("history shape must match dictionary history slices")
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history_dictionary = history_dictionary.reshape(
        history_dictionary.shape[0], height_width, -1
    )
    sensor_dictionary = flat_history_dictionary[:, spatial_sensor_indices, :].reshape(
        -1,
        history_dictionary.shape[-1],
    )
    flat_history = history.transpose(1, 2, 3, 0).reshape(
        history_dictionary.shape[0],
        height_width,
        history.shape[0],
    )
    measurements = flat_history[:, spatial_sensor_indices, :].reshape(-1, history.shape[0]).T
    coeffs = measurements @ np.linalg.pinv(sensor_dictionary, rcond=rcond).T
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs


def predict_next_sensor_lstsq(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    history = segments[:-1].transpose(3, 0, 1, 2)
    return predict_from_history_sensor_lstsq(
        history,
        dictionary,
        spatial_sensor_indices,
        rcond=rcond,
    )


def target_frames_from_segments(segments: np.ndarray) -> np.ndarray:
    return segments[-1].transpose(2, 0, 1)


def reconstruct_target_from_coefficients(coeffs: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    """Reconstruct centered target frames from TBMD target-slice coefficients."""
    _, target_dictionary = history_and_target(dictionary)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    flat_target_dictionary = target_dictionary.reshape(-1, target_dictionary.shape[-1])
    predictions = coeffs @ flat_target_dictionary.T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1])


def encode_target_coefficients(
    target_frames: np.ndarray,
    dictionary: np.ndarray,
    *,
    rcond: float = 1e-6,
) -> np.ndarray:
    """Encode true target frames in the target-slice TBMD dictionary."""
    _, target_dictionary = history_and_target(dictionary)
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    flat_target_dictionary = target_dictionary.reshape(-1, target_dictionary.shape[-1])
    target_encoder = np.linalg.pinv(flat_target_dictionary.T, rcond=rcond)
    return target_flat @ target_encoder


def fit_coefficient_calibrator(
    recovered_coeffs: np.ndarray,
    target_frames: np.ndarray,
    dictionary: np.ndarray,
    *,
    calibration_type: str,
    target: str,
    alpha: float,
    blend: float,
    rcond: float,
    measurements: np.ndarray | None = None,
    decoder_payload: dict[str, Any] | None = None,
    innovation_rank: int = 0,
    include_norms: bool = False,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit a low-dimensional map from recovered coefficients to target coefficients."""
    calibration_type = calibration_type.lower()
    target = target.lower()
    if calibration_type == "none":
        return {"type": "none", "blend": 0.0}
    if calibration_type not in {"ridge", "delta_ridge"}:
        raise ValueError(f"Unknown coefficient_calibration_type: {calibration_type}")
    if target != "target":
        raise ValueError(f"Unknown coefficient_calibration_target: {target}")
    if alpha < 0:
        raise ValueError("coefficient_calibration_alpha must be non-negative")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("coefficient_calibration_blend must be in [0, 1]")

    source = np.asarray(recovered_coeffs, dtype=np.float64)
    target_coeffs = encode_target_coefficients(
        target_frames,
        dictionary,
        rcond=rcond,
    )
    innovation_encoder = None
    feature_matrix = source
    if innovation_rank > 0 or include_norms:
        if measurements is None or decoder_payload is None:
            raise ValueError(
                "measurements and decoder_payload are required for innovation coefficient calibration"
            )
        innovation_encoder = fit_sensor_innovation_encoder(
            measurements,
            source,
            decoder_payload,
            rank=innovation_rank,
            include_norms=include_norms,
            random_state=random_state,
        )
        innovation_features = transform_sensor_innovation_features(
            measurements,
            source,
            decoder_payload,
            innovation_encoder,
        )
        feature_matrix = np.concatenate([source, innovation_features], axis=1)
    features = np.concatenate(
        [feature_matrix, np.ones((source.shape[0], 1), dtype=np.float64)], axis=1
    )
    regression_target = target_coeffs if calibration_type == "ridge" else target_coeffs - source
    gram = features.T @ features
    penalty = alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = features.T @ regression_target
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    train_correction = features @ weights
    train_calibrated = (
        train_correction if calibration_type == "ridge" else source + train_correction
    )
    payload = {
        "type": calibration_type,
        "target": "target",
        "alpha": float(alpha),
        "blend": float(blend),
        "rcond": float(rcond),
        "weights": weights,
        "source_dim": int(source.shape[1]),
        "feature_dim": int(feature_matrix.shape[1]),
        "target_dim": int(target_coeffs.shape[1]),
        "train_coeff_rmse": float(np.sqrt(np.mean((train_calibrated - target_coeffs) ** 2))),
    }
    if innovation_encoder is not None:
        payload["innovation_encoder"] = innovation_encoder
    return payload


def apply_coefficient_calibrator(
    recovered_coeffs: np.ndarray,
    calibrator: dict[str, Any],
    *,
    measurements: np.ndarray | None = None,
    decoder_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply optional coefficient calibration before target reconstruction."""
    coeffs = np.asarray(recovered_coeffs, dtype=np.float64)
    if not calibrator or calibrator.get("type", "none") == "none":
        return coeffs
    if calibrator.get("type") not in {"ridge", "delta_ridge"}:
        raise ValueError(f"Unknown coefficient calibrator type: {calibrator.get('type')}")
    feature_matrix = coeffs
    encoder = calibrator.get("innovation_encoder")
    if encoder:
        if measurements is None or decoder_payload is None:
            raise ValueError(
                "measurements and decoder_payload are required for innovation coefficient calibration"
            )
        innovation_features = transform_sensor_innovation_features(
            measurements,
            coeffs,
            decoder_payload,
            encoder,
        )
        feature_matrix = np.concatenate([coeffs, innovation_features], axis=1)
    features = np.concatenate(
        [feature_matrix, np.ones((coeffs.shape[0], 1), dtype=np.float64)], axis=1
    )
    correction = features @ np.asarray(calibrator["weights"], dtype=np.float64)
    blend = float(calibrator.get("blend", 1.0))
    if calibrator.get("type") == "delta_ridge":
        return coeffs + blend * correction
    return (1.0 - blend) * coeffs + blend * correction


def compute_residual_sample_weights(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    *,
    weighting: str,
    power: float,
    floor: float,
    clip: float,
) -> np.ndarray:
    """Compute normalized per-frame weights from current residual magnitude."""
    if power < 0:
        raise ValueError("sample weight power must be non-negative")
    if floor <= 0:
        raise ValueError("sample weight floor must be positive")
    if clip < floor:
        raise ValueError("sample weight clip must be >= floor")
    target = np.asarray(target_frames, dtype=np.float64)
    base = np.asarray(base_predictions, dtype=np.float64)
    if target.shape != base.shape:
        raise ValueError("target_frames and base_predictions must have the same shape")
    weighting = weighting.lower()
    if weighting == "uniform":
        return np.ones(target.shape[0], dtype=np.float64)
    if weighting != "hard_frame_rmse":
        raise ValueError(f"Unknown sample weighting: {weighting}")
    residual = (target - base).reshape(target.shape[0], -1)
    rmse = np.sqrt(np.mean(residual**2, axis=1) + 1e-12)
    normalized = rmse / max(float(np.mean(rmse)), 1e-12)
    weights = normalized**power
    weights = np.clip(weights, float(floor), float(clip))
    weights /= max(float(np.mean(weights)), 1e-12)
    return weights


def _fit_residual_svd_targets(
    target_flat: np.ndarray,
    base_flat: np.ndarray,
    residual_rank: int,
    *,
    residual_weighting: str,
    residual_weight_floor: float,
    spatial_shape: tuple[int, int] | None = None,
    highpass_cutoff_fraction: float = 0.35,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    return _fit_residual_svd_targets_from_residual(
        target_flat - base_flat,
        residual_rank,
        residual_weighting=residual_weighting,
        residual_weight_floor=residual_weight_floor,
        spatial_shape=spatial_shape,
        highpass_cutoff_fraction=highpass_cutoff_fraction,
        sample_weights=sample_weights,
    )


def _fit_residual_svd_targets_from_residual(
    residual: np.ndarray,
    residual_rank: int,
    *,
    residual_weighting: str,
    residual_weight_floor: float,
    spatial_shape: tuple[int, int] | None = None,
    highpass_cutoff_fraction: float = 0.35,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if residual_rank <= 0:
        raise ValueError("residual_rank must be positive when provided")
    if residual_weight_floor <= 0:
        raise ValueError("residual_weight_floor must be positive")
    residual = np.asarray(residual, dtype=np.float64)
    actual_rank = min(int(residual_rank), residual.shape[0], residual.shape[1])
    if sample_weights is None:
        sample_weights = np.ones(residual.shape[0], dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if sample_weights.shape != (residual.shape[0],):
        raise ValueError("sample_weights must have one value per residual sample")
    sample_weights = sample_weights / max(float(np.mean(sample_weights)), 1e-12)
    residual_mean = np.average(residual, axis=0, weights=sample_weights)
    centered_residual = residual - residual_mean
    if residual_weighting == "uniform":
        residual_weights = np.ones(centered_residual.shape[1], dtype=np.float64)
    elif residual_weighting == "residual_energy":
        residual_weights = np.sqrt(np.mean(centered_residual**2, axis=0) + 1e-12)
        residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
        residual_weights = np.maximum(residual_weights, float(residual_weight_floor))
        residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
    elif residual_weighting == "highpass_energy":
        if spatial_shape is None:
            raise ValueError("spatial_shape is required for highpass_energy weighting")
        highpass = fft_highpass_frames(
            centered_residual.reshape(centered_residual.shape[0], *spatial_shape),
            cutoff_fraction=highpass_cutoff_fraction,
        ).reshape(centered_residual.shape)
        residual_weights = np.sqrt(np.mean(highpass**2, axis=0) + 1e-12)
        residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
        residual_weights = np.maximum(residual_weights, float(residual_weight_floor))
        residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
    else:
        raise ValueError(f"Unknown residual_weighting: {residual_weighting}")
    weighted_residual = centered_residual * residual_weights
    svd_residual = weighted_residual * np.sqrt(sample_weights)[:, None]
    _, _, vt = np.linalg.svd(svd_residual, full_matrices=False)
    residual_basis = vt[:actual_rank]
    residual_codes = weighted_residual @ residual_basis.T
    return residual_mean, residual_basis, residual_codes, residual_weights, actual_rank


def fit_ridge_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float,
    residual_rank: Optional[int] = None,
    residual_weighting: str = "uniform",
    residual_weight_floor: float = 0.1,
    sample_weighting: str = "uniform",
    sample_weight_power: float = 1.0,
    sample_weight_floor: float = 0.25,
    sample_weight_clip: float = 4.0,
    residual_target: str = "field",
    highpass_cutoff_fraction: float = 0.35,
    feature_matrix: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    model_features = (
        coeffs if feature_matrix is None else np.asarray(feature_matrix, dtype=np.float64)
    )
    sample_weights = compute_residual_sample_weights(
        target_frames,
        base_predictions,
        weighting=sample_weighting,
        power=sample_weight_power,
        floor=sample_weight_floor,
        clip=sample_weight_clip,
    )
    features = np.concatenate(
        [model_features, np.ones((model_features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    residual = residual_target_frames(
        target_frames,
        base_predictions,
        residual_target=residual_target,
        highpass_cutoff_fraction=highpass_cutoff_fraction,
    ).reshape(target_frames.shape[0], -1)
    residual_basis = None
    residual_mean = None
    residual_weights = None
    regression_target = residual
    actual_residual_rank = None
    if residual_rank is not None:
        (
            residual_mean,
            residual_basis,
            regression_target,
            residual_weights,
            actual_residual_rank,
        ) = _fit_residual_svd_targets_from_residual(
            residual,
            residual_rank,
            residual_weighting=residual_weighting,
            residual_weight_floor=residual_weight_floor,
            spatial_shape=target_frames.shape[1:],
            highpass_cutoff_fraction=highpass_cutoff_fraction,
            sample_weights=sample_weights,
        )
    row_weights = np.sqrt(sample_weights)[:, None]
    weighted_features = features * row_weights
    weighted_target = regression_target * row_weights
    gram = weighted_features.T @ weighted_features
    penalty = alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = weighted_features.T @ weighted_target
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    corrector = {
        "alpha": float(alpha),
        "weights": weights,
        "feature_dim": int(model_features.shape[1]),
        "coefficient_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
        "mode": "residual_svd" if residual_basis is not None else "full",
        "residual_rank": actual_residual_rank,
        "residual_target": residual_target,
        "highpass_cutoff_fraction": float(highpass_cutoff_fraction),
        "residual_weighting": residual_weighting if residual_basis is not None else "uniform",
        "sample_weighting": sample_weighting,
        "sample_weight_power": float(sample_weight_power),
        "sample_weight_floor": float(sample_weight_floor),
        "sample_weight_clip": float(sample_weight_clip),
        "sample_weight_min": float(np.min(sample_weights)),
        "sample_weight_max": float(np.max(sample_weights)),
    }
    if residual_basis is not None:
        corrector["residual_basis"] = residual_basis
        corrector["residual_mean"] = residual_mean
        corrector["residual_weights"] = residual_weights
    return corrector


def fit_noop_residual_corrector(
    target_frames: np.ndarray,
    coeffs: np.ndarray,
) -> dict[str, Any]:
    """Return a corrector that intentionally keeps the calibrated base prediction."""
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return {
        "alpha": 0.0,
        "weights": np.empty((0, 0), dtype=np.float64),
        "feature_dim": int(coeffs.shape[1]),
        "coefficient_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
        "mode": "none",
        "residual_rank": None,
        "residual_weighting": "uniform",
    }


def fit_mlp_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    *,
    residual_rank: int,
    hidden_size: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    random_state: int,
    residual_weighting: str = "uniform",
    residual_weight_floor: float = 0.1,
    residual_target: str = "field",
    highpass_cutoff_fraction: float = 0.35,
    feature_matrix: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if hidden_size <= 0 or num_epochs < 0 or batch_size <= 0:
        raise ValueError(
            "hidden_size and batch_size must be positive; num_epochs must be non-negative"
        )
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    model_features = (
        coeffs if feature_matrix is None else np.asarray(feature_matrix, dtype=np.float64)
    )
    residual = residual_target_frames(
        target_frames,
        base_predictions,
        residual_target=residual_target,
        highpass_cutoff_fraction=highpass_cutoff_fraction,
    ).reshape(target_frames.shape[0], -1)
    residual_mean, residual_basis, residual_codes, residual_weights, actual_rank = (
        _fit_residual_svd_targets_from_residual(
            residual,
            residual_rank,
            residual_weighting=residual_weighting,
            residual_weight_floor=residual_weight_floor,
            spatial_shape=target_frames.shape[1:],
            highpass_cutoff_fraction=highpass_cutoff_fraction,
        )
    )
    input_mean = model_features.mean(axis=0)
    input_std = model_features.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    target_mean = residual_codes.mean(axis=0)
    target_std = residual_codes.std(axis=0)
    target_std[target_std < 1e-8] = 1.0

    x_np = ((model_features - input_mean) / input_std).astype(np.float32)
    y_np = ((residual_codes - target_mean) / target_std).astype(np.float32)
    torch.manual_seed(int(random_state))
    rng = np.random.default_rng(random_state)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_np.shape[1], hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, actual_rank),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.from_numpy(y_np)
    losses: list[float] = []
    for _ in range(num_epochs):
        order = rng.permutation(x_np.shape[0])
        epoch_loss = 0.0
        for start in range(0, x_np.shape[0], batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_tensor[idx])
            loss = torch.mean((pred - y_tensor[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
        losses.append(epoch_loss / x_np.shape[0])

    layer0 = model[0]
    layer1 = model[2]
    return {
        "mode": "residual_svd_mlp",
        "residual_rank": actual_rank,
        "residual_basis": residual_basis,
        "residual_mean": residual_mean,
        "residual_weights": residual_weights,
        "residual_target": residual_target,
        "highpass_cutoff_fraction": float(highpass_cutoff_fraction),
        "residual_weighting": residual_weighting,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "hidden_size": int(hidden_size),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "train_loss": losses,
        "feature_dim": int(model_features.shape[1]),
        "coefficient_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
        "mlp_w0": layer0.weight.detach().cpu().numpy().astype(np.float64),
        "mlp_b0": layer0.bias.detach().cpu().numpy().astype(np.float64),
        "mlp_w1": layer1.weight.detach().cpu().numpy().astype(np.float64),
        "mlp_b1": layer1.bias.detach().cpu().numpy().astype(np.float64),
    }


def _build_non_overlapping_patch_slices(
    height: int,
    width: int,
    patch_size: int,
) -> np.ndarray:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("patch_size must evenly divide both spatial dimensions")
    return np.asarray(
        [
            [row, row + patch_size, col, col + patch_size]
            for row in range(0, height, patch_size)
            for col in range(0, width, patch_size)
        ],
        dtype=np.int64,
    )


def fit_patch_residual_svd_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float,
    patch_size: int,
    patch_residual_rank: int,
    residual_weighting: str = "uniform",
    residual_weight_floor: float = 0.1,
    sample_weighting: str = "uniform",
    sample_weight_power: float = 1.0,
    sample_weight_floor: float = 0.25,
    sample_weight_clip: float = 4.0,
    residual_target: str = "field",
    highpass_cutoff_fraction: float = 0.35,
    feature_matrix: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Fit local residual-SVD ridge heads on non-overlapping spatial patches."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if patch_residual_rank <= 0:
        raise ValueError("patch_residual_rank must be positive")
    if residual_weight_floor <= 0:
        raise ValueError("residual_weight_floor must be positive")

    target = np.asarray(target_frames, dtype=np.float64)
    base = np.asarray(base_predictions, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    model_features = (
        coeffs if feature_matrix is None else np.asarray(feature_matrix, dtype=np.float64)
    )
    if target.shape != base.shape or target.ndim != 3:
        raise ValueError("target_frames and base_predictions must share shape `(N,H,W)`")
    sample_weights = compute_residual_sample_weights(
        target,
        base,
        weighting=sample_weighting,
        power=sample_weight_power,
        floor=sample_weight_floor,
        clip=sample_weight_clip,
    )
    patch_slices = _build_non_overlapping_patch_slices(
        target.shape[1],
        target.shape[2],
        patch_size,
    )
    features = np.concatenate(
        [model_features, np.ones((model_features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    residual = residual_target_frames(
        target,
        base,
        residual_target=residual_target,
        highpass_cutoff_fraction=highpass_cutoff_fraction,
    )
    highpass_residual = fft_highpass_frames(
        residual,
        cutoff_fraction=highpass_cutoff_fraction,
    )

    patch_means = []
    patch_bases = []
    patch_weights = []
    patch_codes = []
    actual_rank = None
    for row0, row1, col0, col1 in patch_slices:
        patch = residual[:, row0:row1, col0:col1].reshape(residual.shape[0], -1)
        patch_mean = np.average(patch, axis=0, weights=sample_weights)
        centered_patch = patch - patch_mean
        if residual_weighting == "uniform":
            weights = np.ones(centered_patch.shape[1], dtype=np.float64)
        elif residual_weighting == "residual_energy":
            weights = np.sqrt(np.mean(centered_patch**2, axis=0) + 1e-12)
            weights /= max(float(np.mean(weights)), 1e-12)
            weights = np.maximum(weights, float(residual_weight_floor))
            weights /= max(float(np.mean(weights)), 1e-12)
        elif residual_weighting == "highpass_energy":
            highpass_patch = highpass_residual[:, row0:row1, col0:col1].reshape(
                residual.shape[0],
                -1,
            )
            weights = np.sqrt(np.mean(highpass_patch**2, axis=0) + 1e-12)
            weights /= max(float(np.mean(weights)), 1e-12)
            weights = np.maximum(weights, float(residual_weight_floor))
            weights /= max(float(np.mean(weights)), 1e-12)
        else:
            raise ValueError(f"Unknown residual_weighting: {residual_weighting}")
        weighted_patch = centered_patch * weights
        svd_patch = weighted_patch * np.sqrt(sample_weights)[:, None]
        rank = min(int(patch_residual_rank), weighted_patch.shape[0], weighted_patch.shape[1])
        if actual_rank is None:
            actual_rank = rank
        elif actual_rank != rank:
            raise ValueError("All patches must have the same effective residual rank")
        _, _, vt = np.linalg.svd(svd_patch, full_matrices=False)
        basis = vt[:rank]
        patch_means.append(patch_mean)
        patch_bases.append(basis)
        patch_weights.append(weights)
        patch_codes.append(weighted_patch @ basis.T)

    regression_target = np.concatenate(patch_codes, axis=1)
    row_weights = np.sqrt(sample_weights)[:, None]
    weighted_features = features * row_weights
    weighted_target = regression_target * row_weights
    gram = weighted_features.T @ weighted_features
    penalty = alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = weighted_features.T @ weighted_target
    try:
        weights_matrix = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights_matrix = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    return {
        "alpha": float(alpha),
        "weights": weights_matrix,
        "feature_dim": int(model_features.shape[1]),
        "coefficient_dim": int(coeffs.shape[1]),
        "output_dim": int(np.prod(target.shape[1:])),
        "mode": "patch_residual_svd",
        "patch_size": int(patch_size),
        "patch_residual_rank": int(actual_rank or patch_residual_rank),
        "residual_target": residual_target,
        "highpass_cutoff_fraction": float(highpass_cutoff_fraction),
        "patch_slices": patch_slices,
        "patch_means": np.stack(patch_means, axis=0),
        "patch_bases": np.stack(patch_bases, axis=0),
        "patch_weights": np.stack(patch_weights, axis=0),
        "residual_weighting": residual_weighting,
        "residual_weight_floor": float(residual_weight_floor),
        "sample_weighting": sample_weighting,
        "sample_weight_power": float(sample_weight_power),
        "sample_weight_floor": float(sample_weight_floor),
        "sample_weight_clip": float(sample_weight_clip),
        "sample_weight_min": float(np.min(sample_weights)),
        "sample_weight_max": float(np.max(sample_weights)),
    }


def fit_composite_patch_hf_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float,
    patch_size: int,
    patch_residual_rank: int,
    hf_residual_rank: int,
    patch_scale: float,
    hf_scale: float,
    highpass_cutoff_fraction: float = 0.35,
    residual_weight_floor: float = 0.1,
    feature_matrix: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Fit a two-component residual head: local patch field + global HF-weighted field."""
    if hf_residual_rank <= 0:
        raise ValueError("hf_residual_rank must be positive")
    patch_corrector = fit_patch_residual_svd_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=alpha,
        patch_size=patch_size,
        patch_residual_rank=patch_residual_rank,
        residual_weighting="uniform",
        residual_weight_floor=residual_weight_floor,
        residual_target="field",
        feature_matrix=feature_matrix,
    )
    hf_corrector = fit_ridge_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=alpha,
        residual_rank=hf_residual_rank,
        residual_weighting="highpass_energy",
        residual_weight_floor=residual_weight_floor,
        residual_target="field",
        highpass_cutoff_fraction=highpass_cutoff_fraction,
        feature_matrix=feature_matrix,
    )
    return {
        "alpha": float(alpha),
        "weights": np.empty((0, 0), dtype=np.float64),
        "feature_dim": int(patch_corrector["feature_dim"]),
        "coefficient_dim": int(patch_corrector["coefficient_dim"]),
        "output_dim": int(patch_corrector["output_dim"]),
        "mode": "composite_patch_hf_svd",
        "residual_rank": int(hf_residual_rank),
        "residual_target": "field",
        "residual_weighting": "composite_patch_hf",
        "patch_size": int(patch_size),
        "patch_residual_rank": int(patch_residual_rank),
        "hf_residual_rank": int(hf_residual_rank),
        "patch_scale": float(patch_scale),
        "hf_scale": float(hf_scale),
        "highpass_cutoff_fraction": float(highpass_cutoff_fraction),
        "components": [
            {
                "name": "patch",
                "scale": float(patch_scale),
                "corrector": patch_corrector,
            },
            {
                "name": "hfweighted",
                "scale": float(hf_scale),
                "corrector": hf_corrector,
            },
        ],
    }


def attach_coefficient_gate(
    corrector: dict[str, Any],
    coeffs: np.ndarray,
    *,
    gate_type: str,
    threshold: float,
    strength: float,
    gate_min: float,
    measurements: np.ndarray | None = None,
    decoder_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach train-distribution stats for residual shrinkage gates."""
    gate_type = gate_type.lower()
    if gate_type == "none":
        corrector["gate_type"] = "none"
        return corrector
    if gate_type not in {"coefficient_rms", "sensor_innovation_rms"}:
        raise ValueError(f"Unknown correction_gate_type: {gate_type}")
    if threshold < 0:
        raise ValueError("correction_gate_threshold must be non-negative")
    if strength < 0:
        raise ValueError("correction_gate_strength must be non-negative")
    if not 0.0 <= gate_min <= 1.0:
        raise ValueError("correction_gate_min must be in [0, 1]")
    if gate_type == "coefficient_rms":
        coeffs = np.asarray(coeffs, dtype=np.float64)
        coeff_mean = coeffs.mean(axis=0)
        coeff_std = coeffs.std(axis=0)
        coeff_std[coeff_std < 1e-8] = 1.0
        corrector.update(
            {
                "gate_type": "coefficient_rms",
                "gate_coeff_mean": coeff_mean,
                "gate_coeff_std": coeff_std,
                "gate_threshold": float(threshold),
                "gate_strength": float(strength),
                "gate_min": float(gate_min),
            }
        )
        return corrector
    if measurements is None or decoder_payload is None:
        raise ValueError(
            "measurements and decoder_payload are required for sensor_innovation_rms gate"
        )
    innovation = sensor_innovation_residual(measurements, coeffs, decoder_payload)
    innovation_rms = np.sqrt(np.mean(innovation**2, axis=1))
    measurement_rms = np.sqrt(np.mean(np.asarray(measurements, dtype=np.float64) ** 2, axis=1))
    relative_rms = innovation_rms / np.maximum(measurement_rms, 1e-12)
    gate_mean = float(np.mean(relative_rms))
    gate_std = float(np.std(relative_rms))
    if gate_std < 1e-8:
        gate_std = 1.0
    corrector.update(
        {
            "gate_type": "sensor_innovation_rms",
            "gate_innovation_mean": gate_mean,
            "gate_innovation_std": gate_std,
            "gate_threshold": float(threshold),
            "gate_strength": float(strength),
            "gate_min": float(gate_min),
        }
    )
    return corrector


def _coefficient_gate_values(
    coeffs: np.ndarray,
    corrector: dict[str, Any],
    *,
    measurements: np.ndarray | None = None,
    decoder_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    gate_type = corrector.get("gate_type", "none")
    if gate_type == "none":
        return np.ones(coeffs.shape[0], dtype=np.float64)
    if gate_type == "sensor_innovation_rms":
        if measurements is None or decoder_payload is None:
            raise ValueError(
                "measurements and decoder_payload are required for sensor_innovation_rms gate"
            )
        innovation = sensor_innovation_residual(measurements, coeffs, decoder_payload)
        innovation_rms = np.sqrt(np.mean(innovation**2, axis=1))
        measurement_rms = np.sqrt(np.mean(np.asarray(measurements, dtype=np.float64) ** 2, axis=1))
        relative_rms = innovation_rms / np.maximum(measurement_rms, 1e-12)
        standardized = (relative_rms - float(corrector.get("gate_innovation_mean", 0.0))) / max(
            float(corrector.get("gate_innovation_std", 1.0)), 1e-12
        )
        excess = np.maximum(standardized - float(corrector["gate_threshold"]), 0.0)
        gate = 1.0 / (1.0 + float(corrector["gate_strength"]) * excess)
        return np.maximum(gate, float(corrector["gate_min"]))
    if gate_type != "coefficient_rms":
        raise ValueError(f"Unknown gate_type: {gate_type}")
    standardized = (
        np.asarray(coeffs, dtype=np.float64)
        - np.asarray(corrector["gate_coeff_mean"], dtype=np.float64)
    ) / np.asarray(corrector["gate_coeff_std"], dtype=np.float64)
    rms = np.sqrt(np.mean(standardized**2, axis=1))
    excess = np.maximum(rms - float(corrector["gate_threshold"]), 0.0)
    gate = 1.0 / (1.0 + float(corrector["gate_strength"]) * excess)
    return np.maximum(gate, float(corrector["gate_min"]))


def apply_ridge_residual_corrector(
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    corrector: dict[str, Any],
    *,
    scale: float,
    measurements: np.ndarray | None = None,
    decoder_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(
        base_predictions.shape[0], -1
    )
    coeffs = np.asarray(coeffs, dtype=np.float64)
    mode = corrector.get("mode", "full")
    if mode == "none":
        return base_flat.reshape(base_predictions.shape)
    if mode == "composite_patch_hf_svd":
        combined = base_flat.reshape(base_predictions.shape).copy()
        for component in corrector["components"]:
            component_pred = apply_ridge_residual_corrector(
                base_predictions,
                coeffs,
                component["corrector"],
                scale=float(component["scale"]),
                measurements=measurements,
                decoder_payload=decoder_payload,
            )
            combined += component_pred - base_predictions
        return combined
    model_features = build_correction_feature_matrix(
        coeffs,
        corrector,
        measurements=measurements,
        decoder_payload=decoder_payload,
    )
    features = np.concatenate(
        [model_features, np.ones((model_features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    if mode == "residual_svd_mlp":
        standardized = (
            model_features - np.asarray(corrector["input_mean"], dtype=np.float64)
        ) / np.asarray(corrector["input_std"], dtype=np.float64)
        hidden = np.tanh(
            standardized @ np.asarray(corrector["mlp_w0"], dtype=np.float64).T
            + np.asarray(corrector["mlp_b0"], dtype=np.float64)
        )
        normalized_codes = hidden @ np.asarray(
            corrector["mlp_w1"], dtype=np.float64
        ).T + np.asarray(corrector["mlp_b1"], dtype=np.float64)
        correction = normalized_codes * np.asarray(
            corrector["target_std"], dtype=np.float64
        ) + np.asarray(corrector["target_mean"], dtype=np.float64)
    else:
        correction = features @ np.asarray(corrector["weights"], dtype=np.float64)
    if mode == "patch_residual_svd":
        patch_slices = np.asarray(corrector["patch_slices"], dtype=np.int64)
        patch_bases = np.asarray(corrector["patch_bases"], dtype=np.float64)
        patch_means = np.asarray(corrector["patch_means"], dtype=np.float64)
        patch_weights = np.asarray(corrector["patch_weights"], dtype=np.float64)
        patch_rank = int(corrector["patch_residual_rank"])
        residual_field = np.zeros_like(base_predictions, dtype=np.float64)
        for patch_idx, (row0, row1, col0, col1) in enumerate(patch_slices):
            start = patch_idx * patch_rank
            stop = start + patch_rank
            patch_codes = correction[:, start:stop]
            patch_flat = (
                patch_means[patch_idx]
                + (patch_codes @ patch_bases[patch_idx]) / patch_weights[patch_idx]
            )
            residual_field[:, row0:row1, col0:col1] = patch_flat.reshape(
                base_predictions.shape[0],
                row1 - row0,
                col1 - col0,
            )
        correction = residual_field.reshape(base_predictions.shape[0], -1)
    elif mode in {"residual_svd", "residual_svd_mlp"}:
        residual_weights = np.asarray(
            corrector.get("residual_weights", np.ones(corrector["residual_basis"].shape[1])),
            dtype=np.float64,
        )
        correction = (
            np.asarray(corrector["residual_mean"], dtype=np.float64)
            + (correction @ np.asarray(corrector["residual_basis"], dtype=np.float64))
            / residual_weights
        )
    gate = _coefficient_gate_values(
        coeffs,
        corrector,
        measurements=measurements,
        decoder_payload=decoder_payload,
    )
    return (base_flat + scale * gate[:, None] * correction).reshape(base_predictions.shape)


class FastWindowedTBMDQRCSForecaster:
    """Fast one-step predictor based on windowed TBMD, QR sensors, and a small head."""

    def __init__(self, config: Optional[FastWindowedTBMDQRCSConfig] = None):
        self.config = config or FastWindowedTBMDQRCSConfig()
        self._spatial_mean: Optional[np.ndarray] = None
        self._dictionary: Optional[np.ndarray] = None
        self._spatial_mask: Optional[np.ndarray] = None
        self._spatial_sensor_indices: Optional[np.ndarray] = None
        self._sensor_decoder_payload: Optional[dict[str, Any]] = None
        self._coefficient_calibrator: Optional[dict[str, Any]] = None
        self._coefficient_corrector: Optional[dict[str, Any]] = None
        self._metrics: dict[str, Any] = {}
        self._fitted = False

    def fit(self, train_states: np.ndarray) -> "FastWindowedTBMDQRCSForecaster":
        states = np.asarray(train_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("train_states must have shape `(B,T,H,W)`")
        self._spatial_mean = (
            np.mean(states, axis=(0, 1))
            if self.config.spatial_mean_centering
            else np.zeros(states.shape[2:], dtype=np.float64)
        )
        centered = states - self._spatial_mean
        segments, segment_refs = build_forecast_segment_tensor_with_refs(
            centered,
            history_length=self.config.history_length,
            stride=self.config.segment_stride,
            max_segments=self.config.max_train_segments,
        )
        dictionary, tbmd_summary = fit_segment_dictionary(
            segments,
            ranks=list(self.config.ranks),
            random_state=self.config.random_state,
            dtype=self.config.dtype,
        )
        history_dictionary, _ = history_and_target(dictionary)
        spatial_mask, spatial_sensor_indices = place_fixed_spatial_sensors(
            history_dictionary,
            n_spatial_sensors=self.config.n_spatial_sensors,
            random_state=self.config.random_state,
        )
        sensor_decoder_payload = fit_sensor_coefficient_decoder(
            dictionary,
            spatial_sensor_indices,
            decoder=self.config.sensor_decoder,
            rcond=self.config.sensor_rcond,
            ridge_lambda=self.config.decoder_ridge_lambda,
            l1_lambda=self.config.decoder_l1_lambda,
            max_iter=self.config.decoder_max_iter,
            tol=self.config.decoder_tol,
        )
        raw_base_pred, raw_coeffs, measurements = predict_next_sensor_decoder_with_measurements(
            segments,
            dictionary,
            spatial_sensor_indices,
            sensor_decoder_payload,
        )
        targets = target_frames_from_segments(segments)
        coefficient_calibrator = fit_coefficient_calibrator(
            raw_coeffs,
            targets,
            dictionary,
            calibration_type=self.config.coefficient_calibration_type,
            target=self.config.coefficient_calibration_target,
            alpha=self.config.coefficient_calibration_alpha,
            blend=self.config.coefficient_calibration_blend,
            rcond=self.config.coefficient_calibration_rcond,
            measurements=measurements,
            decoder_payload=sensor_decoder_payload,
            innovation_rank=self.config.coefficient_calibration_innovation_rank,
            include_norms=self.config.coefficient_calibration_include_norms,
            random_state=self.config.random_state,
        )
        coeffs = apply_coefficient_calibrator(
            raw_coeffs,
            coefficient_calibrator,
            measurements=measurements,
            decoder_payload=sensor_decoder_payload,
        )
        if self.config.coefficient_temporal_smoothing_alpha > 0.0:
            coeffs = smooth_coefficients_by_segment_refs(
                coeffs,
                segment_refs,
                alpha=self.config.coefficient_temporal_smoothing_alpha,
                reset_on_gap=self.config.coefficient_temporal_reset_on_gap,
            )
        base_pred = reconstruct_target_from_coefficients(coeffs, dictionary)
        innovation_encoder = fit_sensor_innovation_encoder(
            measurements,
            raw_coeffs,
            sensor_decoder_payload,
            rank=self.config.correction_innovation_rank,
            include_norms=self.config.correction_innovation_include_norms,
            random_state=self.config.random_state,
        )
        feature_probe = {"innovation_encoder": innovation_encoder}
        feature_matrix = build_correction_feature_matrix(
            coeffs,
            feature_probe,
            measurements=measurements,
            decoder_payload=sensor_decoder_payload,
        )
        if self.config.correction_head_type == "ridge":
            corrector = fit_ridge_residual_corrector(
                targets,
                base_pred,
                coeffs,
                alpha=self.config.correction_alpha,
                residual_rank=self.config.correction_residual_rank,
                residual_weighting=self.config.correction_residual_weighting,
                residual_weight_floor=self.config.correction_residual_weight_floor,
                sample_weighting=self.config.correction_sample_weighting,
                sample_weight_power=self.config.correction_sample_weight_power,
                sample_weight_floor=self.config.correction_sample_weight_floor,
                sample_weight_clip=self.config.correction_sample_weight_clip,
                residual_target=self.config.correction_residual_target,
                highpass_cutoff_fraction=self.config.correction_highpass_cutoff_fraction,
                feature_matrix=feature_matrix,
            )
        elif self.config.correction_head_type == "mlp_residual_svd":
            if self.config.correction_residual_rank is None:
                raise ValueError("mlp_residual_svd requires correction_residual_rank")
            corrector = fit_mlp_residual_corrector(
                targets,
                base_pred,
                coeffs,
                residual_rank=self.config.correction_residual_rank,
                residual_weighting=self.config.correction_residual_weighting,
                residual_weight_floor=self.config.correction_residual_weight_floor,
                hidden_size=self.config.correction_hidden_size,
                num_epochs=self.config.correction_num_epochs,
                batch_size=self.config.correction_batch_size,
                learning_rate=self.config.correction_learning_rate,
                weight_decay=self.config.correction_weight_decay,
                random_state=self.config.random_state,
                feature_matrix=feature_matrix,
                residual_target=self.config.correction_residual_target,
                highpass_cutoff_fraction=self.config.correction_highpass_cutoff_fraction,
            )
        elif self.config.correction_head_type == "patch_residual_svd":
            patch_rank = (
                self.config.correction_patch_residual_rank
                if self.config.correction_patch_residual_rank is not None
                else self.config.correction_residual_rank
            )
            if patch_rank is None:
                raise ValueError(
                    "patch_residual_svd requires correction_patch_residual_rank "
                    "or correction_residual_rank"
                )
            corrector = fit_patch_residual_svd_corrector(
                targets,
                base_pred,
                coeffs,
                alpha=self.config.correction_alpha,
                patch_size=self.config.correction_patch_size,
                patch_residual_rank=patch_rank,
                residual_weighting=self.config.correction_residual_weighting,
                residual_weight_floor=self.config.correction_residual_weight_floor,
                sample_weighting=self.config.correction_sample_weighting,
                sample_weight_power=self.config.correction_sample_weight_power,
                sample_weight_floor=self.config.correction_sample_weight_floor,
                sample_weight_clip=self.config.correction_sample_weight_clip,
                residual_target=self.config.correction_residual_target,
                highpass_cutoff_fraction=self.config.correction_highpass_cutoff_fraction,
                feature_matrix=feature_matrix,
            )
        elif self.config.correction_head_type == "composite_patch_hf_svd":
            patch_rank = self.config.correction_patch_residual_rank
            hf_rank = self.config.correction_residual_rank
            if patch_rank is None or hf_rank is None:
                raise ValueError(
                    "composite_patch_hf_svd requires correction_patch_residual_rank "
                    "and correction_residual_rank"
                )
            corrector = fit_composite_patch_hf_residual_corrector(
                targets,
                base_pred,
                coeffs,
                alpha=self.config.correction_alpha,
                patch_size=self.config.correction_patch_size,
                patch_residual_rank=patch_rank,
                hf_residual_rank=hf_rank,
                patch_scale=self.config.correction_scale,
                hf_scale=self.config.correction_hf_scale,
                highpass_cutoff_fraction=self.config.correction_highpass_cutoff_fraction,
                residual_weight_floor=self.config.correction_residual_weight_floor,
                feature_matrix=feature_matrix,
            )
        elif self.config.correction_head_type == "none":
            corrector = fit_noop_residual_corrector(targets, coeffs)
        else:
            raise ValueError(f"Unknown correction_head_type: {self.config.correction_head_type}")
        corrector["innovation_encoder"] = innovation_encoder
        corrector = attach_coefficient_gate(
            corrector,
            coeffs,
            gate_type=self.config.correction_gate_type,
            threshold=self.config.correction_gate_threshold,
            strength=self.config.correction_gate_strength,
            gate_min=self.config.correction_gate_min,
            measurements=measurements,
            decoder_payload=sensor_decoder_payload,
        )
        corrected = apply_ridge_residual_corrector(
            base_pred,
            coeffs,
            corrector,
            scale=self.config.correction_scale,
            measurements=measurements,
            decoder_payload=sensor_decoder_payload,
        )
        self._dictionary = dictionary
        self._spatial_mask = spatial_mask
        self._spatial_sensor_indices = spatial_sensor_indices
        self._sensor_decoder_payload = sensor_decoder_payload
        self._coefficient_calibrator = coefficient_calibrator
        self._coefficient_corrector = corrector
        self._metrics = {
            "fit": {
                "n_train_segments": int(segments.shape[-1]),
                "train_raw_base": _compute_regression_metrics(targets, raw_base_pred),
                "train_base": _compute_regression_metrics(targets, base_pred),
                "train_corrected": _compute_regression_metrics(targets, corrected),
                "tbmd_summary": tbmd_summary,
                "dictionary_shape": list(dictionary.shape),
                "actual_spatial_sensors": int(spatial_mask.sum()),
                "total_history_measurements_per_prediction": int(
                    spatial_mask.sum() * self.config.history_length
                ),
                "sensor_decoder": self.config.sensor_decoder,
                "coefficient_temporal_smoothing_alpha": self.config.coefficient_temporal_smoothing_alpha,
                "coefficient_temporal_reset_on_gap": self.config.coefficient_temporal_reset_on_gap,
                "coefficient_calibrator": {
                    key: value for key, value in coefficient_calibrator.items() if key != "weights"
                },
            }
        }
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before prediction")

    def predict_next(self, history_states: np.ndarray) -> np.ndarray:
        pred, _ = self.predict_next_with_coefficients(history_states)
        return pred

    def predict_next_with_coefficients(
        self, history_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        history = np.asarray(history_states, dtype=np.float64)
        if history.ndim != 4:
            raise ValueError("history_states must have shape `(B,L,H,W)`")
        if history.shape[1:] != self._dictionary[:-1].shape[:-1]:
            raise ValueError("history_states must match fitted history shape")
        centered_history = history - self._spatial_mean
        _, raw_coeffs, measurements = predict_from_history_sensor_decoder_with_measurements(
            centered_history,
            self._dictionary,
            self._spatial_sensor_indices,
            self._sensor_decoder_payload,
        )
        coeffs = apply_coefficient_calibrator(
            raw_coeffs,
            self._coefficient_calibrator or {},
            measurements=measurements,
            decoder_payload=self._sensor_decoder_payload,
        )
        base_pred = reconstruct_target_from_coefficients(coeffs, self._dictionary)
        corrected = apply_ridge_residual_corrector(
            base_pred,
            coeffs,
            self._coefficient_corrector,
            scale=self.config.correction_scale,
            measurements=measurements,
            decoder_payload=self._sensor_decoder_payload,
        )
        return corrected + self._spatial_mean, coeffs

    def evaluate_one_step(self, test_states: np.ndarray) -> dict[str, Any]:
        self._require_fitted()
        states = np.asarray(test_states, dtype=np.float64)
        centered = states - self._spatial_mean
        segments, segment_refs = build_forecast_segment_tensor_with_refs(
            centered,
            history_length=self.config.history_length,
            stride=self.config.segment_stride,
            max_segments=None,
        )
        if self.config.coefficient_temporal_smoothing_alpha > 0.0:
            _, raw_coeffs, measurements = predict_next_sensor_decoder_with_measurements(
                segments,
                self._dictionary,
                self._spatial_sensor_indices,
                self._sensor_decoder_payload,
            )
            coeffs = apply_coefficient_calibrator(
                raw_coeffs,
                self._coefficient_calibrator or {},
                measurements=measurements,
                decoder_payload=self._sensor_decoder_payload,
            )
            coeffs = smooth_coefficients_by_segment_refs(
                coeffs,
                segment_refs,
                alpha=self.config.coefficient_temporal_smoothing_alpha,
                reset_on_gap=self.config.coefficient_temporal_reset_on_gap,
            )
            base_pred = reconstruct_target_from_coefficients(coeffs, self._dictionary)
            pred_centered = apply_ridge_residual_corrector(
                base_pred,
                coeffs,
                self._coefficient_corrector,
                scale=self.config.correction_scale,
                measurements=measurements,
                decoder_payload=self._sensor_decoder_payload,
            )
            pred_spatial = pred_centered + self._spatial_mean
            target_spatial = target_frames_from_segments(segments) + self._spatial_mean
        else:
            predictions = []
            targets = []
            for start in range(0, states.shape[1] - self.config.history_length):
                history = states[:, start : start + self.config.history_length]
                pred = self.predict_next(history)
                predictions.append(pred)
                targets.append(states[:, start + self.config.history_length])
            pred_spatial = np.concatenate(predictions, axis=0)
            target_spatial = np.concatenate(targets, axis=0)
        metrics = _compute_regression_metrics(target_spatial, pred_spatial)
        payload = {
            "spatial_mse": metrics["mse"],
            "spatial_rmse": metrics["rmse"],
            "spatial_rel_frob_err": metrics["rel_frob_err"],
            "spatial_r2": metrics["r2"],
            "n_eval_samples": int(target_spatial.shape[0]),
            "target_spatial": target_spatial,
            "pred_spatial": pred_spatial,
            "segments_shape": list(segments.shape),
        }
        self._metrics["last_one_step"] = {
            key: value
            for key, value in payload.items()
            if key not in {"target_spatial", "pred_spatial"}
        }
        return payload

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        corrector = self._coefficient_corrector
        sensor_decoder = self._sensor_decoder_payload or {}
        coefficient_calibrator = self._coefficient_calibrator or {"type": "none"}
        np.savez_compressed(
            path,
            config=np.asarray([asdict(self.config)], dtype=object),
            spatial_mean=self._spatial_mean.astype(np.float32),
            dictionary=self._dictionary.astype(np.float32),
            spatial_mask=self._spatial_mask.astype(bool),
            spatial_sensor_indices=self._spatial_sensor_indices.astype(np.int64),
            sensor_decoder_type=np.asarray(sensor_decoder.get("type", self.config.sensor_decoder)),
            sensor_decoder_matrix=np.asarray(
                sensor_decoder.get("decoder_matrix", np.empty((0, 0))),
                dtype=np.float32,
            ),
            sensor_decoder_sensing_matrix=np.asarray(
                sensor_decoder.get("sensing_matrix", np.empty((0, 0))),
                dtype=np.float32,
            ),
            sensor_decoder_lipschitz=np.asarray(
                sensor_decoder.get("lipschitz", np.nan),
                dtype=np.float64,
            ),
            coefficient_calibrator_type=np.asarray(
                coefficient_calibrator.get("type", "none"),
            ),
            coefficient_calibrator_target=np.asarray(
                coefficient_calibrator.get("target", "target"),
            ),
            coefficient_calibrator_alpha=np.asarray(
                coefficient_calibrator.get("alpha", np.nan),
                dtype=np.float64,
            ),
            coefficient_calibrator_blend=np.asarray(
                coefficient_calibrator.get("blend", 0.0),
                dtype=np.float64,
            ),
            coefficient_calibrator_rcond=np.asarray(
                coefficient_calibrator.get("rcond", np.nan),
                dtype=np.float64,
            ),
            coefficient_calibrator_weights=np.asarray(
                coefficient_calibrator.get("weights", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_calibrator_payload=np.asarray(
                [coefficient_calibrator],
                dtype=object,
            ),
            coefficient_corrector_weights=np.asarray(
                corrector.get("weights", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_alpha=np.asarray(
                corrector.get("alpha", np.nan), dtype=np.float64
            ),
            coefficient_corrector_mode=np.asarray(corrector.get("mode", "full")),
            coefficient_corrector_residual_rank=np.asarray(
                -1 if corrector.get("residual_rank") is None else corrector["residual_rank"],
                dtype=np.int64,
            ),
            coefficient_corrector_residual_basis=np.asarray(
                corrector.get("residual_basis", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_residual_mean=np.asarray(
                corrector.get("residual_mean", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_residual_weights=np.asarray(
                corrector.get("residual_weights", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_residual_weighting=np.asarray(
                corrector.get("residual_weighting", "uniform"),
            ),
            coefficient_corrector_residual_target=np.asarray(
                corrector.get("residual_target", "field"),
            ),
            coefficient_corrector_highpass_cutoff_fraction=np.asarray(
                corrector.get("highpass_cutoff_fraction", 0.35),
                dtype=np.float64,
            ),
            coefficient_corrector_patch_size=np.asarray(
                corrector.get("patch_size", -1),
                dtype=np.int64,
            ),
            coefficient_corrector_patch_residual_rank=np.asarray(
                corrector.get("patch_residual_rank", -1),
                dtype=np.int64,
            ),
            coefficient_corrector_patch_slices=np.asarray(
                corrector.get("patch_slices", np.empty((0, 4))),
                dtype=np.int64,
            ),
            coefficient_corrector_patch_means=np.asarray(
                corrector.get("patch_means", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_patch_bases=np.asarray(
                corrector.get("patch_bases", np.empty((0, 0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_patch_weights=np.asarray(
                corrector.get("patch_weights", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_gate_type=np.asarray(
                corrector.get("gate_type", "none"),
            ),
            coefficient_corrector_gate_coeff_mean=np.asarray(
                corrector.get("gate_coeff_mean", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_gate_coeff_std=np.asarray(
                corrector.get("gate_coeff_std", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_gate_innovation_mean=np.asarray(
                corrector.get("gate_innovation_mean", np.nan),
                dtype=np.float64,
            ),
            coefficient_corrector_gate_innovation_std=np.asarray(
                corrector.get("gate_innovation_std", np.nan),
                dtype=np.float64,
            ),
            coefficient_corrector_gate_threshold=np.asarray(
                corrector.get("gate_threshold", np.nan),
                dtype=np.float64,
            ),
            coefficient_corrector_gate_strength=np.asarray(
                corrector.get("gate_strength", np.nan),
                dtype=np.float64,
            ),
            coefficient_corrector_gate_min=np.asarray(
                corrector.get("gate_min", np.nan),
                dtype=np.float64,
            ),
            coefficient_corrector_innovation_rank=np.asarray(
                corrector.get("innovation_encoder", {}).get("innovation_rank", 0),
                dtype=np.int64,
            ),
            coefficient_corrector_innovation_include_norms=np.asarray(
                corrector.get("innovation_encoder", {}).get("innovation_include_norms", False),
                dtype=bool,
            ),
            coefficient_corrector_innovation_mean=np.asarray(
                corrector.get("innovation_encoder", {}).get("innovation_mean", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_innovation_basis=np.asarray(
                corrector.get("innovation_encoder", {}).get(
                    "innovation_basis",
                    np.empty((0, 0)),
                ),
                dtype=np.float32,
            ),
            coefficient_corrector_innovation_feature_mean=np.asarray(
                corrector.get("innovation_encoder", {}).get(
                    "innovation_feature_mean",
                    np.empty((0,)),
                ),
                dtype=np.float32,
            ),
            coefficient_corrector_innovation_feature_std=np.asarray(
                corrector.get("innovation_encoder", {}).get(
                    "innovation_feature_std",
                    np.empty((0,)),
                ),
                dtype=np.float32,
            ),
            coefficient_corrector_input_mean=np.asarray(
                corrector.get("input_mean", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_input_std=np.asarray(
                corrector.get("input_std", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_target_mean=np.asarray(
                corrector.get("target_mean", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_target_std=np.asarray(
                corrector.get("target_std", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_mlp_w0=np.asarray(
                corrector.get("mlp_w0", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_mlp_b0=np.asarray(
                corrector.get("mlp_b0", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_mlp_w1=np.asarray(
                corrector.get("mlp_w1", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_mlp_b1=np.asarray(
                corrector.get("mlp_b1", np.empty((0,))),
                dtype=np.float32,
            ),
            coefficient_corrector_payload=np.asarray(
                [corrector if corrector.get("mode") == "composite_patch_hf_svd" else None],
                dtype=object,
            ),
            metrics=np.asarray([self._metrics], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FastWindowedTBMDQRCSForecaster":
        data = np.load(Path(path), allow_pickle=True)
        config_payload = data["config"][0]
        if hasattr(config_payload, "item"):
            config_payload = config_payload.item()
        config = FastWindowedTBMDQRCSConfig(**config_payload)
        model = cls(config)
        model._spatial_mean = data["spatial_mean"].astype(np.float64)
        model._dictionary = data["dictionary"].astype(np.float64)
        model._spatial_mask = data["spatial_mask"].astype(bool)
        model._spatial_sensor_indices = data["spatial_sensor_indices"].astype(int)
        if "sensor_decoder_type" in data:
            decoder_type = str(data["sensor_decoder_type"].item())
            model._sensor_decoder_payload = {"type": decoder_type}
            decoder_matrix = data["sensor_decoder_matrix"].astype(np.float64)
            sensing_matrix = data["sensor_decoder_sensing_matrix"].astype(np.float64)
            if decoder_matrix.size:
                model._sensor_decoder_payload["decoder_matrix"] = decoder_matrix
            if sensing_matrix.size:
                model._sensor_decoder_payload["sensing_matrix"] = sensing_matrix
            if decoder_type == "lstsq":
                model._sensor_decoder_payload["rcond"] = float(model.config.sensor_rcond)
            elif decoder_type == "ridge":
                model._sensor_decoder_payload["ridge_lambda"] = float(
                    model.config.decoder_ridge_lambda
                )
            elif decoder_type == "fista":
                model._sensor_decoder_payload.update(
                    {
                        "lipschitz": float(data["sensor_decoder_lipschitz"]),
                        "l1_lambda": float(model.config.decoder_l1_lambda),
                        "max_iter": int(model.config.decoder_max_iter),
                        "tol": float(model.config.decoder_tol),
                    }
                )
        else:
            model._sensor_decoder_payload = fit_sensor_coefficient_decoder(
                model._dictionary,
                model._spatial_sensor_indices,
                decoder=model.config.sensor_decoder,
                rcond=model.config.sensor_rcond,
                ridge_lambda=model.config.decoder_ridge_lambda,
                l1_lambda=model.config.decoder_l1_lambda,
                max_iter=model.config.decoder_max_iter,
                tol=model.config.decoder_tol,
            )
        if "coefficient_calibrator_type" in data:
            calibrator_type = str(data["coefficient_calibrator_type"].item())
            model._coefficient_calibrator = {
                "type": calibrator_type,
                "target": str(data["coefficient_calibrator_target"].item()),
                "alpha": float(data["coefficient_calibrator_alpha"]),
                "blend": float(data["coefficient_calibrator_blend"]),
                "rcond": float(data["coefficient_calibrator_rcond"]),
            }
            calibrator_weights = data["coefficient_calibrator_weights"].astype(np.float64)
            if calibrator_weights.size:
                model._coefficient_calibrator["weights"] = calibrator_weights
                model._coefficient_calibrator["source_dim"] = int(calibrator_weights.shape[0] - 1)
                model._coefficient_calibrator["feature_dim"] = int(calibrator_weights.shape[0] - 1)
                model._coefficient_calibrator["target_dim"] = int(calibrator_weights.shape[1])
        else:
            model._coefficient_calibrator = {"type": "none", "blend": 0.0}
        if "coefficient_calibrator_payload" in data:
            calibrator_payload = data["coefficient_calibrator_payload"][0]
            if hasattr(calibrator_payload, "item") and not isinstance(calibrator_payload, dict):
                calibrator_payload = calibrator_payload.item()
            if calibrator_payload is not None:
                model._coefficient_calibrator = calibrator_payload
        weights = data["coefficient_corrector_weights"].astype(np.float64)
        mode = "full"
        if "coefficient_corrector_mode" in data:
            mode = str(data["coefficient_corrector_mode"].item())
        model._coefficient_corrector = {
            "alpha": float(data["coefficient_corrector_alpha"]),
            "weights": weights,
            "feature_dim": int(weights.shape[0] - 1),
            "output_dim": int(np.prod(model._dictionary.shape[1:3])),
            "mode": mode,
            "residual_rank": None,
        }
        if mode == "none":
            model._coefficient_corrector["feature_dim"] = int(model._dictionary.shape[-1])
            model._coefficient_corrector["coefficient_dim"] = int(model._dictionary.shape[-1])
        if "coefficient_corrector_gate_type" in data:
            gate_type = str(data["coefficient_corrector_gate_type"].item())
            model._coefficient_corrector["gate_type"] = gate_type
            if gate_type == "coefficient_rms":
                model._coefficient_corrector.update(
                    {
                        "gate_coeff_mean": data["coefficient_corrector_gate_coeff_mean"].astype(
                            np.float64
                        ),
                        "gate_coeff_std": data["coefficient_corrector_gate_coeff_std"].astype(
                            np.float64
                        ),
                        "gate_threshold": float(data["coefficient_corrector_gate_threshold"]),
                        "gate_strength": float(data["coefficient_corrector_gate_strength"]),
                        "gate_min": float(data["coefficient_corrector_gate_min"]),
                    }
                )
            elif gate_type == "sensor_innovation_rms":
                model._coefficient_corrector.update(
                    {
                        "gate_innovation_mean": float(
                            data["coefficient_corrector_gate_innovation_mean"]
                        ),
                        "gate_innovation_std": float(
                            data["coefficient_corrector_gate_innovation_std"]
                        ),
                        "gate_threshold": float(data["coefficient_corrector_gate_threshold"]),
                        "gate_strength": float(data["coefficient_corrector_gate_strength"]),
                        "gate_min": float(data["coefficient_corrector_gate_min"]),
                    }
                )
        else:
            model._coefficient_corrector["gate_type"] = "none"
        if "coefficient_corrector_innovation_rank" in data:
            innovation_rank = int(data["coefficient_corrector_innovation_rank"])
            innovation_include_norms = bool(data["coefficient_corrector_innovation_include_norms"])
            if innovation_rank > 0 or innovation_include_norms:
                model._coefficient_corrector["innovation_encoder"] = {
                    "innovation_rank": innovation_rank,
                    "innovation_include_norms": innovation_include_norms,
                    "innovation_mean": data["coefficient_corrector_innovation_mean"].astype(
                        np.float64
                    ),
                    "innovation_basis": data["coefficient_corrector_innovation_basis"].astype(
                        np.float64
                    ),
                    "innovation_feature_mean": data[
                        "coefficient_corrector_innovation_feature_mean"
                    ].astype(np.float64),
                    "innovation_feature_std": data[
                        "coefficient_corrector_innovation_feature_std"
                    ].astype(np.float64),
                }
            else:
                model._coefficient_corrector["innovation_encoder"] = None
        else:
            model._coefficient_corrector["innovation_encoder"] = None
        if mode in {"residual_svd", "residual_svd_mlp"}:
            residual_rank = int(data["coefficient_corrector_residual_rank"])
            model._coefficient_corrector["residual_rank"] = residual_rank
            model._coefficient_corrector["residual_basis"] = data[
                "coefficient_corrector_residual_basis"
            ].astype(np.float64)
            model._coefficient_corrector["residual_mean"] = data[
                "coefficient_corrector_residual_mean"
            ].astype(np.float64)
            if "coefficient_corrector_residual_weights" in data:
                model._coefficient_corrector["residual_weights"] = data[
                    "coefficient_corrector_residual_weights"
                ].astype(np.float64)
            else:
                model._coefficient_corrector["residual_weights"] = np.ones(
                    model._coefficient_corrector["residual_basis"].shape[1],
                    dtype=np.float64,
                )
            if "coefficient_corrector_residual_weighting" in data:
                model._coefficient_corrector["residual_weighting"] = str(
                    data["coefficient_corrector_residual_weighting"].item()
                )
            else:
                model._coefficient_corrector["residual_weighting"] = "uniform"
            if "coefficient_corrector_residual_target" in data:
                model._coefficient_corrector["residual_target"] = str(
                    data["coefficient_corrector_residual_target"].item()
                )
                model._coefficient_corrector["highpass_cutoff_fraction"] = float(
                    data["coefficient_corrector_highpass_cutoff_fraction"]
                )
            else:
                model._coefficient_corrector["residual_target"] = "field"
                model._coefficient_corrector["highpass_cutoff_fraction"] = 0.35
        if mode == "patch_residual_svd":
            model._coefficient_corrector.update(
                {
                    "patch_size": int(data["coefficient_corrector_patch_size"]),
                    "patch_residual_rank": int(data["coefficient_corrector_patch_residual_rank"]),
                    "patch_slices": data["coefficient_corrector_patch_slices"].astype(np.int64),
                    "patch_means": data["coefficient_corrector_patch_means"].astype(np.float64),
                    "patch_bases": data["coefficient_corrector_patch_bases"].astype(np.float64),
                    "patch_weights": data["coefficient_corrector_patch_weights"].astype(np.float64),
                }
            )
            if "coefficient_corrector_residual_weighting" in data:
                model._coefficient_corrector["residual_weighting"] = str(
                    data["coefficient_corrector_residual_weighting"].item()
                )
            else:
                model._coefficient_corrector["residual_weighting"] = "uniform"
            if "coefficient_corrector_residual_target" in data:
                model._coefficient_corrector["residual_target"] = str(
                    data["coefficient_corrector_residual_target"].item()
                )
                model._coefficient_corrector["highpass_cutoff_fraction"] = float(
                    data["coefficient_corrector_highpass_cutoff_fraction"]
                )
            else:
                model._coefficient_corrector["residual_target"] = "field"
                model._coefficient_corrector["highpass_cutoff_fraction"] = 0.35
        if mode == "residual_svd_mlp":
            model._coefficient_corrector.update(
                {
                    "input_mean": data["coefficient_corrector_input_mean"].astype(np.float64),
                    "input_std": data["coefficient_corrector_input_std"].astype(np.float64),
                    "target_mean": data["coefficient_corrector_target_mean"].astype(np.float64),
                    "target_std": data["coefficient_corrector_target_std"].astype(np.float64),
                    "mlp_w0": data["coefficient_corrector_mlp_w0"].astype(np.float64),
                    "mlp_b0": data["coefficient_corrector_mlp_b0"].astype(np.float64),
                    "mlp_w1": data["coefficient_corrector_mlp_w1"].astype(np.float64),
                    "mlp_b1": data["coefficient_corrector_mlp_b1"].astype(np.float64),
                }
            )
            model._coefficient_corrector["feature_dim"] = int(
                model._coefficient_corrector["mlp_w0"].shape[1]
            )
        if "coefficient_corrector_payload" in data:
            corrector_payload = data["coefficient_corrector_payload"][0]
            if hasattr(corrector_payload, "item") and not isinstance(corrector_payload, dict):
                corrector_payload = corrector_payload.item()
            if corrector_payload is not None:
                model._coefficient_corrector = corrector_payload
        metrics_payload = data["metrics"][0]
        if hasattr(metrics_payload, "item"):
            metrics_payload = metrics_payload.item()
        model._metrics = metrics_payload
        model._fitted = True
        return model

    def get_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)


__all__ = [
    "FastWindowedTBMDQRCSConfig",
    "FastWindowedTBMDQRCSForecaster",
    "apply_coefficient_calibrator",
    "attach_coefficient_gate",
    "build_forecast_segment_tensor",
    "compute_residual_sample_weights",
    "encode_target_coefficients",
    "fft_highpass_frames",
    "fit_coefficient_calibrator",
    "fit_composite_patch_hf_residual_corrector",
    "fit_noop_residual_corrector",
    "fit_ridge_residual_corrector",
    "fit_sensor_innovation_encoder",
    "fit_patch_residual_svd_corrector",
    "fit_sensor_coefficient_decoder",
    "predict_from_history_sensor_decoder",
    "predict_from_history_sensor_decoder_with_measurements",
    "predict_from_history_sensor_lstsq",
    "reconstruct_target_from_coefficients",
    "residual_target_frames",
    "sensor_innovation_residual",
    "transform_sensor_innovation_features",
]
