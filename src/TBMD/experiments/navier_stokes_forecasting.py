"""
Navier-Stokes-specific helpers for trajectory-aware forecasting experiments.

This module intentionally lives in the experiment layer. It fixes dataset
interpretation and evaluation for the bundled Navier-Stokes data without
changing the shared forecaster APIs.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader, TensorDataset

from TBMD.config import (
    CompressiveSensingConfig,
    LatentModalForecasterConfig,
    LinearForecasterConfig,
    MLPForecasterConfig,
    SensorPlacementConfig,
    LSTMForecasterConfig,
    MultiResolutionTBMDConfig,
)
from TBMD.core.forecasting.LatentModalForecaster import LatentModalForecaster, LatentModalResult
from TBMD.core.forecasting.LinearForecaster import LinearForecaster
from TBMD.core.forecasting.MLPForecaster import MLPForecaster
from TBMD.core.forecasting.LSTMForecaster import LSTMForecaster
from TBMD.core.forecasting.ScheduledSamplingLSTMForecaster import ScheduledSamplingLSTMForecaster
from TBMD.core.reconstruction.tensor_compressive_sensing import (
    ExtensionCompressiveSensingConfig,
    TensorCompressiveSensing,
)
from TBMD.core.sensor_placement import TensorTubeQRDecomposition


@dataclass
class NavierStokesTrajectoryDataset:
    """Container for trajectory-aware Navier-Stokes arrays."""

    train_inputs: np.ndarray
    train_labels: np.ndarray
    train_states: np.ndarray
    test_inputs: np.ndarray
    test_labels: np.ndarray
    test_states: np.ndarray


def _normalize_spatial_array(array: np.ndarray) -> np.ndarray:
    """
    Convert Navier-Stokes arrays to either `(N, H, W)` or `(B, T, H, W)`.
    """
    arr = np.asarray(array)

    if arr.ndim in (3, 4):
        return arr.astype(np.float64, copy=False)

    squeezed = arr
    while squeezed.ndim > 4 and squeezed.shape[-1] == 1:
        squeezed = np.squeeze(squeezed, axis=-1)

    if squeezed.ndim in (3, 4):
        return squeezed.astype(np.float64, copy=False)

    raise ValueError(
        f"Expected a squeezable Navier-Stokes array with 3 or 4 dims, got shape {array.shape}"
    )


def reshape_flattened_train_transitions(
    flat_inputs: np.ndarray,
    flat_labels: np.ndarray,
    trajectory_length: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restore the flattened train split into explicit trajectories.

    Parameters
    ----------
    flat_inputs, flat_labels:
        Arrays shaped like `(N, H, W)` after squeezing.
    trajectory_length:
        Number of transitions per trajectory.
    """
    inputs = _normalize_spatial_array(flat_inputs)
    labels = _normalize_spatial_array(flat_labels)

    while inputs.ndim > 3 and inputs.shape[-1] == 1:
        inputs = np.squeeze(inputs, axis=-1)
    while labels.ndim > 3 and labels.shape[-1] == 1:
        labels = np.squeeze(labels, axis=-1)

    if inputs.ndim != 3 or labels.ndim != 3:
        raise ValueError("Flattened train transitions must be 3D after squeezing: `(N, H, W)`")
    if inputs.shape != labels.shape:
        raise ValueError("Train inputs and labels must have identical shapes")
    if trajectory_length <= 0:
        raise ValueError("trajectory_length must be positive")
    if inputs.shape[0] % trajectory_length != 0:
        raise ValueError(
            f"Number of flattened samples ({inputs.shape[0]}) is not divisible by trajectory_length "
            f"({trajectory_length})"
        )

    n_trajectories = inputs.shape[0] // trajectory_length
    spatial_shape = inputs.shape[1:]
    new_shape = (n_trajectories, trajectory_length, *spatial_shape)

    return inputs.reshape(new_shape), labels.reshape(new_shape)


def stitch_inputs_and_labels_to_states(
    trajectory_inputs: np.ndarray,
    trajectory_labels: np.ndarray,
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """
    Reconstruct full trajectory states from `(input_t, label_t=input_{t+1})` pairs.

    Returns an array shaped `(B, T + 1, H, W)`.
    """
    inputs = _normalize_spatial_array(trajectory_inputs)
    labels = _normalize_spatial_array(trajectory_labels)

    if inputs.ndim != 4 or labels.ndim != 4:
        raise ValueError("Trajectory inputs and labels must be 4D: `(B, T, H, W)`")
    if inputs.shape != labels.shape:
        raise ValueError("Trajectory inputs and labels must have identical shapes")

    if inputs.shape[1] > 1:
        continuity_ok = np.allclose(labels[:, :-1], inputs[:, 1:], atol=atol, rtol=0.0)
        if not continuity_ok:
            raise ValueError("Input/label transition continuity check failed across the trajectory")

    first_state = inputs[:, :1]
    tail_states = labels
    return np.concatenate([first_state, tail_states], axis=1)


def load_navier_stokes_trajectory_dataset(
    data_root: str | Path,
    *,
    trajectory_length: int = 19,
) -> NavierStokesTrajectoryDataset:
    """
    Load the bundled Navier-Stokes dataset and return trajectory-aware arrays.
    """
    root = Path(data_root)
    train_inputs = np.load(root / "train" / "inputs.npy")
    train_labels = np.load(root / "train" / "labels.npy")
    test_inputs = np.load(root / "test" / "inputs.npy")
    test_labels = np.load(root / "test" / "labels.npy")

    train_inputs_t, train_labels_t = reshape_flattened_train_transitions(
        train_inputs,
        train_labels,
        trajectory_length=trajectory_length,
    )
    test_inputs_t = _normalize_spatial_array(test_inputs)
    test_labels_t = _normalize_spatial_array(test_labels)

    if test_inputs_t.ndim != 4 or test_labels_t.ndim != 4:
        raise ValueError("Official test split must remain trajectory-shaped `(B, T, H, W)`")

    return NavierStokesTrajectoryDataset(
        train_inputs=train_inputs_t,
        train_labels=train_labels_t,
        train_states=stitch_inputs_and_labels_to_states(train_inputs_t, train_labels_t),
        test_inputs=test_inputs_t,
        test_labels=test_labels_t,
        test_states=stitch_inputs_and_labels_to_states(test_inputs_t, test_labels_t),
    )

def build_one_step_pairs(
    trajectory_series: np.ndarray,
    *,
    predict_deltas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build explicit one-step `(x_t, x_{t+1})` or `(x_t, Δx_t)` pairs from trajectory-shaped data.

    Accepts arrays shaped `(B, T, ...)` and flattens the feature dimensions only.
    """
    series = np.asarray(trajectory_series, dtype=np.float64)
    if series.ndim < 3:
        raise ValueError("trajectory_series must have shape `(B, T, ...)`")

    batch, steps = series.shape[:2]
    feature_dim = int(np.prod(series.shape[2:]))
    flattened = series.reshape(batch, steps, feature_dim)

    x_pairs = flattened[:, :-1, :].reshape(-1, feature_dim)
    y_pairs = flattened[:, 1:, :].reshape(-1, feature_dim)
    if predict_deltas:
        y_pairs = y_pairs - x_pairs
    return x_pairs, y_pairs


def build_lagged_windows(
    trajectory_series: np.ndarray,
    seq_length: int,
    *,
    predict_deltas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build trajectory-safe lagged windows for sequence models.

    Accepts arrays shaped `(B, T, ...)` and returns:
    - windows: `(N, seq_length, W)`
    - targets: `(N, W)`
    """
    series = np.asarray(trajectory_series, dtype=np.float64)
    if series.ndim < 3:
        raise ValueError("trajectory_series must have shape `(B, T, ...)`")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    batch, steps = series.shape[:2]
    if steps <= seq_length:
        raise ValueError(
            f"Need more than seq_length={seq_length} steps per trajectory, got {steps}"
        )

    feature_dim = int(np.prod(series.shape[2:]))
    flattened = series.reshape(batch, steps, feature_dim)

    windows = []
    targets = []
    for trajectory in flattened:
        for end_idx in range(seq_length, steps):
            windows.append(trajectory[end_idx - seq_length : end_idx])
            target = trajectory[end_idx]
            if predict_deltas:
                target = target - trajectory[end_idx - 1]
            targets.append(target)

    return np.asarray(windows), np.asarray(targets)


def build_unrolled_lagged_windows(
    trajectory_series: np.ndarray,
    seq_length: int,
    unroll_steps: int,
    *,
    predict_deltas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build trajectory-safe lagged windows and unrolled targets for sequence models.

    Accepts arrays shaped `(B, T, ...)` and returns:
    - windows: `(N, seq_length, W)`
    - targets: `(N, unroll_steps, W)`
    """
    series = np.asarray(trajectory_series, dtype=np.float64)
    if series.ndim < 3:
        raise ValueError("trajectory_series must have shape `(B, T, ...)`")
    if seq_length <= 0 or unroll_steps <= 0:
        raise ValueError("seq_length and unroll_steps must be positive")

    batch, steps = series.shape[:2]
    if steps <= seq_length + unroll_steps - 1:
        raise ValueError(
            f"Need more than {seq_length + unroll_steps - 1} steps per trajectory, got {steps}"
        )

    feature_dim = int(np.prod(series.shape[2:]))
    flattened = series.reshape(batch, steps, feature_dim)

    windows = []
    targets = []
    for trajectory in flattened:
        for end_idx in range(seq_length, steps - unroll_steps + 1):
            windows.append(trajectory[end_idx - seq_length : end_idx])
            
            target_seq = trajectory[end_idx : end_idx + unroll_steps]
            if predict_deltas:
                prev_seq = trajectory[end_idx - 1 : end_idx + unroll_steps - 1]
                target_seq = target_seq - prev_seq
            targets.append(target_seq)

    return np.asarray(windows), np.asarray(targets)


def _split_trajectory_series_for_validation(
    trajectory_series: np.ndarray,
    val_split: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    series = np.asarray(trajectory_series, dtype=np.float64)
    if series.ndim < 3:
        raise ValueError("trajectory_series must have shape `(B, T, ...)`")
    if val_split <= 0 or series.shape[0] < 2:
        return series, None

    val_count = max(1, int(round(series.shape[0] * val_split)))
    if val_count >= series.shape[0]:
        return series, None

    train_count = series.shape[0] - val_count
    return series[:train_count], series[train_count:]


def _compute_latent_standardization_stats(
    latent_series: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(latent_series, dtype=np.float64)
    flat = series.reshape(-1, series.shape[-1])
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def _apply_latent_standardization(
    latent_series: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (np.asarray(latent_series, dtype=np.float64) - mean) / std


def _invert_latent_standardization(
    latent_series: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.asarray(latent_series, dtype=np.float64) * std + mean


def _augment_latent_series_with_deltas(latent_series: np.ndarray) -> np.ndarray:
    """
    Build an augmented latent state `[c_t, Δc_t]` with zero velocity at `t=0`.
    """
    series = np.asarray(latent_series, dtype=np.float64)
    if series.ndim != 3:
        raise ValueError("latent_series must have shape `(B, T, D)`")

    deltas = np.zeros_like(series)
    deltas[:, 1:, :] = series[:, 1:, :] - series[:, :-1, :]
    return np.concatenate([series, deltas], axis=-1)


def _states_to_decomposition_tensor(states: np.ndarray) -> np.ndarray:
    """Convert `(B, T, H, W)` states to a single `(H, W, N)` tensor."""
    series = np.asarray(states, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("Expected states shaped `(B, T, H, W)`")

    batch, steps, height, width = series.shape
    return series.transpose(2, 3, 0, 1).reshape(height, width, batch * steps)


def _validate_latent_ranks(
    tensor_shape: tuple[int, int, int],
    ranks: int | list[int] | None,
) -> list[int]:
    height, width, n_states = tensor_shape
    if ranks is None:
        return [height, width, min(n_states, 10)]
    if isinstance(ranks, int):
        return [min(ranks, height), min(ranks, width), min(ranks, n_states)]
    if len(ranks) != 3:
        raise ValueError(f"ranks list must have 3 elements, got {len(ranks)}")
    return [
        min(ranks[0], height),
        min(ranks[1], width),
        min(ranks[2], n_states),
    ]


def _can_use_full_spatial_rank_fast_path(
    states: np.ndarray,
    ranks: int | list[int] | None,
) -> bool:
    validated_ranks = _validate_latent_ranks(
        (states.shape[2], states.shape[3], states.shape[0] * states.shape[1]),
        ranks,
    )
    return validated_ranks[0] == states.shape[2] and validated_ranks[1] == states.shape[3]


def _fit_matrix_latent_basis(
    states: np.ndarray,
    ranks: int | list[int] | None,
    random_state: Optional[int],
) -> tuple[np.ndarray, list[np.ndarray], float, list[int], np.ndarray]:
    """
    Fast path for the common Navier-Stokes setup where spatial ranks are full.

    In this regime the Tucker model collapses to a rank-R factorization over the
    flattened spatial snapshots: `X ~= C @ V^T`, where `V^T` is the temporal
    latent basis and `C` are per-state latent coefficients.
    """
    series = np.asarray(states, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("states must have shape `(B, T, H, W)`")

    n_trajectories, steps, height, width = series.shape
    flat_states = series.reshape(n_trajectories * steps, height * width)
    validated_ranks = _validate_latent_ranks((height, width, flat_states.shape[0]), ranks)
    latent_rank = validated_ranks[2]

    if latent_rank <= 0:
        raise ValueError("Temporal rank must be positive")

    if latent_rank == min(flat_states.shape):
        left, singular_values, right_t = np.linalg.svd(flat_states, full_matrices=False)
        left = left[:, :latent_rank]
        singular_values = singular_values[:latent_rank]
        right_t = right_t[:latent_rank]
    else:
        left, singular_values, right_t = randomized_svd(
            flat_states,
            n_components=latent_rank,
            n_iter=5,
            random_state=random_state,
        )

    temporal_coeffs = left * singular_values
    basis_vectors = np.asarray(right_t, dtype=np.float64)
    core = basis_vectors.reshape(latent_rank, height, width).transpose(1, 2, 0).copy()

    fro_sq = float(np.sum(flat_states * flat_states))
    captured_sq = float(np.sum(singular_values * singular_values))
    residual_sq = max(fro_sq - captured_sq, 0.0)
    relative_error = float(np.sqrt(residual_sq / max(fro_sq, 1e-12)))

    factors = [
        np.eye(height, dtype=np.float64),
        np.eye(width, dtype=np.float64),
        np.asarray(temporal_coeffs, dtype=np.float64),
    ]
    return core, factors, relative_error, validated_ranks, basis_vectors


def _compute_regression_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    mse = float(np.mean((target - pred) ** 2))
    rmse = float(np.sqrt(mse))
    target_norm = np.linalg.norm(target)
    rel_frob = float(np.linalg.norm(target - pred) / max(target_norm, 1e-12))

    target_2d = target.reshape(target.shape[0], -1)
    pred_2d = pred.reshape(pred.shape[0], -1)
    ss_res = float(np.sum((target_2d - pred_2d) ** 2))
    ss_tot = float(np.sum((target_2d - np.mean(target_2d, axis=0)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))

    return {
        "mse": mse,
        "rmse": rmse,
        "rel_frob_err": rel_frob,
        "r2": r2,
    }


class TrajectoryAwarePersistenceForecaster:
    """
    Trajectory-aware persistence baseline for Navier-Stokes.

    One-step mode predicts `x_t` as `x_{t+1}` from the ground-truth current
    state. Rollout mode holds the first state fixed for the whole trajectory.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwarePersistenceForecaster":
        states = np.asarray(train_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("train_states must have shape `(B, T, H, W)`")
        self._fitted = True
        return self

    def evaluate_one_step(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("test_states must have shape `(B, T, H, W)`")
        spatial_shape = states.shape[2:]
        target = states[:, 1:, :, :].reshape(-1, *spatial_shape)
        pred = states[:, :-1, :, :].reshape(-1, *spatial_shape)
        spatial_metrics = _compute_regression_metrics(target, pred)

        return {
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(target.shape[0]),
            "target_spatial": target,
            "pred_spatial": pred,
        }

    def evaluate_rollout(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("test_states must have shape `(B, T, H, W)`")
        spatial_shape = states.shape[2:]
        n_steps = states.shape[1] - 1
        target = states[:, 1:, :, :].reshape(-1, *spatial_shape)
        pred = np.repeat(states[:, :1, :, :], repeats=n_steps, axis=1).reshape(-1, *spatial_shape)
        spatial_metrics = _compute_regression_metrics(target, pred)

        return {
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(target.shape[0]),
            "n_rollout_steps": int(n_steps),
            "target_spatial": target,
            "pred_spatial": pred,
        }


class TrajectoryAwareDMDForecaster:
    """
    Low-rank DMD/linear latent baseline over flattened spatial snapshots.

    The basis is learned from train transitions only. The linear operator is
    fit in the reduced coefficient space and then decoded back to the grid.
    """

    def __init__(
        self,
        *,
        rank: int = 20,
        spatial_mean_centering: bool = True,
        rcond: float = 1e-6,
        random_state: Optional[int] = 0,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.spatial_mean_centering = spatial_mean_centering
        self.rcond = rcond
        self.random_state = random_state
        self._spatial_mean: Optional[np.ndarray] = None
        self._basis: Optional[np.ndarray] = None
        self._operator: Optional[np.ndarray] = None
        self._spatial_shape: Optional[tuple[int, int]] = None
        self._fitted = False

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareDMDForecaster":
        states = np.asarray(train_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("train_states must have shape `(B, T, H, W)`")

        self._spatial_shape = states.shape[2:]
        feature_dim = int(np.prod(self._spatial_shape))
        x_train = states[:, :-1, :, :].reshape(-1, feature_dim)
        y_train = states[:, 1:, :, :].reshape(-1, feature_dim)

        if self.spatial_mean_centering:
            self._spatial_mean = np.mean(x_train, axis=0)
        else:
            self._spatial_mean = np.zeros(feature_dim, dtype=np.float64)

        x_centered = x_train - self._spatial_mean
        y_centered = y_train - self._spatial_mean
        effective_rank = min(self.rank, x_centered.shape[0], x_centered.shape[1])
        if effective_rank == min(x_centered.shape):
            _, _, right_t = np.linalg.svd(x_centered, full_matrices=False)
            basis = right_t[:effective_rank]
        else:
            _, _, right_t = randomized_svd(
                x_centered,
                n_components=effective_rank,
                n_iter=5,
                random_state=self.random_state,
            )
            basis = right_t

        x_coeffs = x_centered @ basis.T
        y_coeffs = y_centered @ basis.T
        self._operator = np.linalg.pinv(x_coeffs, rcond=self.rcond) @ y_coeffs
        self._basis = np.asarray(basis, dtype=np.float64)
        self._fitted = True
        return self

    def _predict_next_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before prediction")
        flat = np.asarray(frame, dtype=np.float64).reshape(1, -1)
        coeffs = (flat - self._spatial_mean) @ self._basis.T
        next_flat = (coeffs @ self._operator) @ self._basis + self._spatial_mean
        return next_flat.reshape(*self._spatial_shape)

    def evaluate_one_step(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("test_states must have shape `(B, T, H, W)`")
        spatial_shape = states.shape[2:]
        inputs = states[:, :-1, :, :].reshape(-1, *spatial_shape)
        target = states[:, 1:, :, :].reshape(-1, *spatial_shape)
        pred = np.stack([self._predict_next_frame(frame) for frame in inputs], axis=0)
        spatial_metrics = _compute_regression_metrics(target, pred)

        return {
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(target.shape[0]),
            "target_spatial": target,
            "pred_spatial": pred,
        }

    def evaluate_rollout(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("test_states must have shape `(B, T, H, W)`")
        spatial_shape = states.shape[2:]
        n_steps = states.shape[1] - 1
        target = states[:, 1:, :, :].reshape(-1, *spatial_shape)

        pred_trajs = []
        for traj_idx in range(states.shape[0]):
            current = states[traj_idx, 0]
            traj_pred = []
            for _ in range(n_steps):
                current = self._predict_next_frame(current)
                traj_pred.append(current)
            pred_trajs.append(np.asarray(traj_pred, dtype=np.float64))

        pred = np.concatenate(pred_trajs, axis=0)
        spatial_metrics = _compute_regression_metrics(target, pred)

        return {
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(target.shape[0]),
            "n_rollout_steps": int(n_steps),
            "target_spatial": target,
            "pred_spatial": pred,
        }


class TrajectoryAwareStableDMDForecaster(TrajectoryAwareDMDForecaster):
    """
    Spectral-radius constrained DMD baseline.

    This tests whether the strong local fit of DMD can be made safer for
    recursive rollout by constraining the reduced linear operator.
    """

    def __init__(
        self,
        *,
        rank: int = 20,
        spatial_mean_centering: bool = True,
        rcond: float = 1e-6,
        random_state: Optional[int] = 0,
        max_spectral_radius: float = 1.0,
    ):
        if max_spectral_radius <= 0:
            raise ValueError("max_spectral_radius must be positive")
        super().__init__(
            rank=rank,
            spatial_mean_centering=spatial_mean_centering,
            rcond=rcond,
            random_state=random_state,
        )
        self.max_spectral_radius = max_spectral_radius
        self._unconstrained_spectral_radius: Optional[float] = None
        self._operator_scale: float = 1.0

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareStableDMDForecaster":
        super().fit(train_states)
        eigenvalues = np.linalg.eigvals(self._operator)
        spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        self._unconstrained_spectral_radius = spectral_radius
        self._operator_scale = 1.0

        if spectral_radius > self.max_spectral_radius:
            self._operator_scale = self.max_spectral_radius / spectral_radius
            self._operator = self._operator * self._operator_scale

        return self


class TrajectoryAwareEigenvalueProjectedDMDForecaster(TrajectoryAwareDMDForecaster):
    """
    DMD baseline with mode-wise eigenvalue projection.

    Unlike uniform operator scaling, this only clips modes whose eigenvalue
    magnitude exceeds the configured radius and leaves already stable modes
    as close to the fitted operator as possible.
    """

    def __init__(
        self,
        *,
        rank: int = 20,
        spatial_mean_centering: bool = True,
        rcond: float = 1e-6,
        random_state: Optional[int] = 0,
        max_spectral_radius: float = 1.0,
    ):
        if max_spectral_radius <= 0:
            raise ValueError("max_spectral_radius must be positive")
        super().__init__(
            rank=rank,
            spatial_mean_centering=spatial_mean_centering,
            rcond=rcond,
            random_state=random_state,
        )
        self.max_spectral_radius = max_spectral_radius
        self._unconstrained_spectral_radius: Optional[float] = None
        self._operator_scale: float = 1.0
        self._n_projected_modes: int = 0
        self._projection_imag_max: float = 0.0

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareEigenvalueProjectedDMDForecaster":
        super().fit(train_states)
        eigenvalues, eigenvectors = np.linalg.eig(self._operator)
        magnitudes = np.abs(eigenvalues)
        self._unconstrained_spectral_radius = (
            float(np.max(magnitudes)) if eigenvalues.size else 0.0
        )
        self._operator_scale = 1.0
        self._n_projected_modes = int(np.sum(magnitudes > self.max_spectral_radius))

        if self._n_projected_modes > 0:
            safe_magnitudes = np.maximum(magnitudes, 1e-12)
            projected_eigenvalues = np.where(
                magnitudes > self.max_spectral_radius,
                eigenvalues * (self.max_spectral_radius / safe_magnitudes),
                eigenvalues,
            )
            try:
                inverse_eigenvectors = np.linalg.inv(eigenvectors)
            except np.linalg.LinAlgError:
                inverse_eigenvectors = np.linalg.pinv(eigenvectors)
            projected_operator = (
                eigenvectors @ np.diag(projected_eigenvalues) @ inverse_eigenvectors
            )
            projected_operator = np.real_if_close(projected_operator, tol=1000)
            if np.iscomplexobj(projected_operator):
                self._projection_imag_max = float(np.max(np.abs(np.imag(projected_operator))))
                projected_operator = np.real(projected_operator)
            self._operator = np.asarray(projected_operator, dtype=np.float64)

        return self


class TrajectoryAwareCSForecaster:
    """
    QR/CS-based coefficient forecaster for the Navier-Stokes experiments.

    Pipeline:
    1. Learn a spatial dictionary ``Psi`` from centered train snapshots.
    2. Select sensor locations by tube-pivot QR on ``Psi``.
    3. Recover coefficient series ``x_t`` from sparse sensor measurements.
    4. Train an LSTM on ``x_t`` and recursively forecast future coefficients.
    5. Reconstruct fields as ``Psi @ x_hat + spatial_mean``.
    """

    _VALID_COEFFICIENT_SOURCES = {"sensor_cs", "sensor_lstsq", "full_projection"}
    _VALID_FEATURE_MODES = {"coeff", "coeff_plus_delta"}
    _VALID_CS_INITIALIZATIONS = {"zero", "sensor_lstsq"}
    _VALID_CORRECTION_FEATURE_MODES = {"last", "window"}

    def __init__(
        self,
        *,
        rank: int = 30,
        n_sensors: int = 15,
        coefficient_source: str = "sensor_cs",
        feature_mode: str = "coeff_plus_delta",
        spatial_mean_centering: bool = True,
        random_state: Optional[int] = 0,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_seq_length: int = 7,
        lstm_num_epochs: int = 150,
        lstm_learning_rate: float = 1e-3,
        lstm_batch_size: int = 32,
        lstm_val_split: float = 0.2,
        lstm_early_stopping_patience: int = 20,
        cs_max_iter: int = 100,
        cs_tol: float = 1e-4,
        cs_epsilon_l1: float = 1e-2,
        cs_initialization: str = "zero",
        cs_solver: str = "cholesky",
        sensor_rcond: float = 1e-6,
        correction_hidden_size: int = 64,
        correction_num_layers: int = 2,
        correction_dropout: float = 0.0,
        correction_learning_rate: float = 1e-3,
        correction_weight_decay: float = 1e-5,
        correction_num_epochs: int = 0,
        correction_batch_size: int = 32,
        correction_val_split: float = 0.2,
        correction_early_stopping_patience: int = 20,
        correction_latent_loss_weight: float = 1.0,
        correction_spatial_loss_weight: float = 0.0,
        correction_rel_frob_loss_weight: float = 0.0,
        correction_feature_mode: str = "last",
        correction_scale: float = 1.0,
        verbose: bool = False,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if n_sensors <= 0:
            raise ValueError("n_sensors must be positive")
        if coefficient_source not in self._VALID_COEFFICIENT_SOURCES:
            raise ValueError(f"Unsupported coefficient_source: {coefficient_source}")
        if feature_mode not in self._VALID_FEATURE_MODES:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        if cs_initialization not in self._VALID_CS_INITIALIZATIONS:
            raise ValueError(f"Unsupported cs_initialization: {cs_initialization}")
        if correction_feature_mode not in self._VALID_CORRECTION_FEATURE_MODES:
            raise ValueError(f"Unsupported correction_feature_mode: {correction_feature_mode}")

        self.rank = int(rank)
        self.requested_n_sensors = int(n_sensors)
        self.n_sensors = int(n_sensors)
        self.coefficient_source = coefficient_source
        self.feature_mode = feature_mode
        self.spatial_mean_centering = spatial_mean_centering
        self.random_state = random_state
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.lstm_seq_length = int(lstm_seq_length)
        self.lstm_num_epochs = int(lstm_num_epochs)
        self.lstm_learning_rate = float(lstm_learning_rate)
        self.lstm_batch_size = int(lstm_batch_size)
        self.lstm_val_split = float(lstm_val_split)
        self.lstm_early_stopping_patience = int(lstm_early_stopping_patience)
        self.cs_max_iter = int(cs_max_iter)
        self.cs_tol = float(cs_tol)
        self.cs_epsilon_l1 = float(cs_epsilon_l1)
        self.cs_initialization = cs_initialization
        self.cs_solver = cs_solver
        self.sensor_rcond = float(sensor_rcond)
        self.correction_hidden_size = int(correction_hidden_size)
        self.correction_num_layers = int(correction_num_layers)
        self.correction_dropout = float(correction_dropout)
        self.correction_learning_rate = float(correction_learning_rate)
        self.correction_weight_decay = float(correction_weight_decay)
        self.correction_num_epochs = int(correction_num_epochs)
        self.correction_batch_size = int(correction_batch_size)
        self.correction_val_split = float(correction_val_split)
        self.correction_early_stopping_patience = int(correction_early_stopping_patience)
        self.correction_latent_loss_weight = float(correction_latent_loss_weight)
        self.correction_spatial_loss_weight = float(correction_spatial_loss_weight)
        self.correction_rel_frob_loss_weight = float(correction_rel_frob_loss_weight)
        self.correction_feature_mode = correction_feature_mode
        self.correction_scale = float(correction_scale)
        self.verbose = verbose

        self._spatial_mean: Optional[np.ndarray] = None
        self._basis_vectors: Optional[np.ndarray] = None
        self._dictionary_tensor: Optional[np.ndarray] = None
        self._sensor_mask: Optional[np.ndarray] = None
        self._sensor_indices: Optional[np.ndarray] = None
        self._sensor_selection_method: Optional[str] = None
        self._sensor_lstsq_pinv: Optional[np.ndarray] = None
        self._coeff_mean: Optional[np.ndarray] = None
        self._coeff_std: Optional[np.ndarray] = None
        self._train_coeffs: Optional[np.ndarray] = None
        self._sub_forecaster: Optional[LSTMForecaster] = None
        self._correction_model: Optional[MLPForecaster] = None
        self._correction_target_mean: Optional[np.ndarray] = None
        self._correction_target_std: Optional[np.ndarray] = None
        self._correction_training_history: Optional[Dict[str, list[float]]] = None
        self._spatial_shape: Optional[tuple[int, int]] = None
        self._fit_cs_metrics: list[dict[str, float | int | bool]] = []
        self._last_projection_metrics: list[dict[str, float | int | bool]] = []
        self._fit_reconstruction_metrics: Optional[dict[str, float]] = None
        self._fitted = False

    @property
    def sensor_mask(self) -> np.ndarray:
        if self._sensor_mask is None:
            raise RuntimeError("Call fit() before accessing sensor_mask")
        return self._sensor_mask.copy()

    @property
    def sensor_indices(self) -> np.ndarray:
        if self._sensor_indices is None:
            raise RuntimeError("Call fit() before accessing sensor_indices")
        return self._sensor_indices.copy()

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareCSForecaster":
        states = np.asarray(train_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("train_states must have shape `(B, T, H, W)`")

        self._spatial_shape = states.shape[2:]
        self._spatial_mean = (
            np.mean(states, axis=(0, 1))
            if self.spatial_mean_centering
            else np.zeros(self._spatial_shape, dtype=np.float64)
        )
        centered_states = states - self._spatial_mean
        self._fit_dictionary(centered_states)
        self._place_sensors()
        self._prepare_sensor_lstsq()

        self._train_coeffs = self._states_to_coefficients(centered_states)
        self._fit_cs_metrics = list(self._last_projection_metrics)
        self._fit_reconstruction_metrics = self._compute_centered_reconstruction_metrics(
            centered_states,
            self._train_coeffs,
        )
        self._coeff_mean, self._coeff_std = _compute_latent_standardization_stats(self._train_coeffs)
        self._fit_sub_forecaster()
        self._fit_correction_head(states)
        self._fitted = True
        return self

    def _fit_dictionary(self, centered_states: np.ndarray) -> None:
        n_trajectories, steps, height, width = centered_states.shape
        flat = centered_states.reshape(n_trajectories * steps, height * width)
        effective_rank = min(self.rank, flat.shape[0], flat.shape[1])
        if effective_rank <= 0:
            raise ValueError("effective rank must be positive")

        if effective_rank == min(flat.shape):
            _, _, right_t = np.linalg.svd(flat, full_matrices=False)
            basis = right_t[:effective_rank]
        else:
            _, _, basis = randomized_svd(
                flat,
                n_components=effective_rank,
                n_iter=5,
                random_state=self.random_state,
            )

        self.rank = int(effective_rank)
        self._basis_vectors = np.asarray(basis, dtype=np.float64)
        self._dictionary_tensor = self._basis_vectors.T.reshape(height, width, self.rank)

    def _place_sensors(self) -> None:
        effective_n_sensors = min(self.requested_n_sensors, self.rank)
        config = SensorPlacementConfig(
            n_sensors=effective_n_sensors,
            random_state=self.random_state,
            verbose=False,
            dtype="float64",
        )
        qr = TensorTubeQRDecomposition(
            self._dictionary_tensor,
            N=effective_n_sensors,
            config=config,
            dtype=torch.float64,
        )
        if self.verbose:
            placement, _, _ = qr.factorize()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                placement, _, _ = qr.factorize()

        mask = placement.detach().cpu().numpy().astype(bool)
        if not np.any(mask):
            raise RuntimeError("QR sensor placement produced an empty sensor mask")

        if self.requested_n_sensors > int(mask.sum()):
            mask = self._augment_sensor_mask_by_leverage(mask)
            self._sensor_selection_method = "qr_plus_leverage"
        else:
            self._sensor_selection_method = "qr"

        self._sensor_mask = mask
        self._sensor_indices = np.flatnonzero(mask.reshape(-1))
        self.n_sensors = int(mask.sum())

    def _augment_sensor_mask_by_leverage(self, qr_mask: np.ndarray) -> np.ndarray:
        flat_mask = np.asarray(qr_mask, dtype=bool).reshape(-1).copy()
        n_available = flat_mask.size - int(flat_mask.sum())
        n_extra = min(self.requested_n_sensors - int(flat_mask.sum()), n_available)
        if n_extra <= 0:
            return flat_mask.reshape(qr_mask.shape)

        flat_dictionary = self._dictionary_tensor.reshape(-1, self.rank)
        leverage_scores = np.sum(flat_dictionary * flat_dictionary, axis=1)
        leverage_scores[flat_mask] = -np.inf
        extra_indices = np.argpartition(-leverage_scores, n_extra - 1)[:n_extra]
        extra_indices = extra_indices[np.argsort(-leverage_scores[extra_indices])]
        flat_mask[extra_indices] = True
        return flat_mask.reshape(qr_mask.shape)

    def _prepare_sensor_lstsq(self) -> None:
        flat_dictionary = self._dictionary_tensor.reshape(-1, self.rank)
        sensor_dictionary = flat_dictionary[self._sensor_indices]
        self._sensor_lstsq_pinv = np.linalg.pinv(
            sensor_dictionary,
            rcond=self.sensor_rcond,
        )

    def _states_to_coefficients(self, centered_states: np.ndarray) -> np.ndarray:
        states = np.asarray(centered_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("states must have shape `(B, T, H, W)`")

        if self.coefficient_source == "full_projection":
            flat = states.reshape(-1, int(np.prod(states.shape[2:])))
            coeffs = flat @ self._basis_vectors.T
            self._last_projection_metrics = []
        elif self.coefficient_source == "sensor_lstsq":
            coeffs = self._recover_sensor_lstsq_coefficients(states)
            self._last_projection_metrics = []
        else:
            coeffs, metrics = self._recover_sensor_cs_coefficients(states)
            self._last_projection_metrics = metrics

        return coeffs.reshape(states.shape[0], states.shape[1], self.rank)

    def _recover_sensor_lstsq_coefficients(self, centered_states: np.ndarray) -> np.ndarray:
        flat = centered_states.reshape(-1, int(np.prod(centered_states.shape[2:])))
        measurements = flat[:, self._sensor_indices]
        return measurements @ self._sensor_lstsq_pinv.T

    def _recover_sensor_cs_coefficients(
        self,
        centered_states: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
        flat_states = centered_states.reshape(-1, *centered_states.shape[2:])
        coeffs = np.zeros((flat_states.shape[0], self.rank), dtype=np.float64)
        metrics_out: list[dict[str, float | int | bool]] = []
        initial_coeffs = None
        if self.cs_initialization == "sensor_lstsq":
            initial_coeffs = self._recover_sensor_lstsq_coefficients(centered_states)

        core_cfg = CompressiveSensingConfig(
            max_iter=self.cs_max_iter,
            tol=self.cs_tol,
            epsilon_l1=self.cs_epsilon_l1,
            device="cpu",
            dtype=torch.float32,
        )
        ext_cfg = ExtensionCompressiveSensingConfig(
            solver=self.cs_solver,
            collect_history=False,
        )

        for idx, field in enumerate(flat_states):
            solver = TensorCompressiveSensing(
                self._dictionary_tensor.astype(np.float32),
                self._sensor_mask,
                field.astype(np.float32),
                core_cfg=core_cfg,
                ext_cfg=ext_cfg,
            )
            if initial_coeffs is not None:
                initial = torch.as_tensor(
                    initial_coeffs[idx],
                    dtype=solver.dtype,
                    device=solver.device,
                ).reshape(-1, 1)
                solver.x = initial.clone()
                solver.d = initial.clone()
                solver._d_prev = initial.clone()
                solver.p = torch.zeros_like(initial)
            coeff, metrics = solver.solve()
            coeffs[idx] = coeff.numpy().astype(np.float64, copy=False)
            metrics_out.append(
                {
                    "iterations": int(metrics.iterations),
                    "converged": bool(metrics.converged),
                    "primal_residual": float(metrics.primal_residual),
                    "dual_residual": float(metrics.dual_residual),
                    "objective": float(metrics.objective),
                    "time_sec": float(metrics.time_sec),
                }
            )
            if self.verbose and (idx + 1) % 500 == 0:
                print(f"Recovered CS coefficients for {idx + 1}/{flat_states.shape[0]} states")

        return coeffs, metrics_out

    def _to_model_coeffs(self, coeff_series: np.ndarray) -> np.ndarray:
        standardized = _apply_latent_standardization(coeff_series, self._coeff_mean, self._coeff_std)
        if self.feature_mode == "coeff":
            return standardized
        return _augment_latent_series_with_deltas(standardized)

    def _from_model_coeffs(self, coeff_series: np.ndarray) -> np.ndarray:
        series = np.asarray(coeff_series, dtype=np.float64)
        if self.feature_mode == "coeff_plus_delta":
            series = series[..., : self.rank]
        return _invert_latent_standardization(series, self._coeff_mean, self._coeff_std)

    def _fit_sub_forecaster(self) -> None:
        model_series = self._to_model_coeffs(self._train_coeffs)
        model_dim = model_series.shape[-1]
        model = LSTMForecaster(
            in_dim=model_dim,
            out_dim=model_dim,
            config=LSTMForecasterConfig(
                in_dim=model_dim,
                out_dim=model_dim,
                seq_length=self.lstm_seq_length,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                learning_rate=self.lstm_learning_rate,
                num_epochs=self.lstm_num_epochs,
                batch_size=self.lstm_batch_size,
                val_split=self.lstm_val_split,
                early_stopping_patience=self.lstm_early_stopping_patience,
                delta_forecast=False,
                verbose=self.verbose,
            ),
        )
        train_series, val_series = _split_trajectory_series_for_validation(
            model_series,
            self.lstm_val_split,
        )
        train_windows, train_targets = build_lagged_windows(
            train_series,
            self.lstm_seq_length,
            predict_deltas=False,
        )
        if val_series is not None:
            val_windows, val_targets = build_lagged_windows(
                val_series,
                self.lstm_seq_length,
                predict_deltas=False,
            )
        else:
            val_windows, val_targets = None, None
        _fit_explicit_lstm(model, train_windows, train_targets, val_windows, val_targets)
        self._sub_forecaster = model

    def _predict_next_model_coeff(self, current_input: np.ndarray) -> np.ndarray:
        return self._sub_forecaster.predict_next(current_input)

    def _make_latest_model_state(self, raw_history: list[np.ndarray]) -> np.ndarray:
        if self.feature_mode == "coeff":
            latest = np.asarray(raw_history[-1], dtype=np.float64).reshape(1, 1, -1)
            return self._to_model_coeffs(latest)[0, -1]

        tail = np.asarray(raw_history[-2:], dtype=np.float64)
        if tail.shape[0] == 1:
            tail = np.concatenate([tail, tail], axis=0)
        return self._to_model_coeffs(tail[np.newaxis, ...])[0, -1]

    def _split_correction_trajectories(
        self,
        raw_series: np.ndarray,
        model_series: np.ndarray,
        spatial_states: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if self.correction_val_split <= 0 or raw_series.shape[0] < 2:
            return raw_series, model_series, spatial_states, None, None, None

        val_count = max(1, int(round(raw_series.shape[0] * self.correction_val_split)))
        if val_count >= raw_series.shape[0]:
            return raw_series, model_series, spatial_states, None, None, None

        split_idx = raw_series.shape[0] - val_count
        return (
            raw_series[:split_idx],
            model_series[:split_idx],
            spatial_states[:split_idx],
            raw_series[split_idx:],
            model_series[split_idx:],
            spatial_states[split_idx:],
        )

    def _build_correction_dataset(
        self,
        raw_coeff_series: np.ndarray,
        model_coeff_series: np.ndarray,
        spatial_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        windows, _ = build_lagged_windows(model_coeff_series, self.lstm_seq_length)
        baseline_pred_model = np.stack(
            [self._predict_next_model_coeff(window) for window in windows],
            axis=0,
        )
        target_raw = raw_coeff_series[:, self.lstm_seq_length :, :].reshape(
            -1,
            raw_coeff_series.shape[-1],
        )
        target_spatial = spatial_states[:, self.lstm_seq_length :, :, :].reshape(
            -1,
            *spatial_states.shape[2:],
        )
        baseline_pred_raw = self._from_model_coeffs(baseline_pred_model)
        correction_features = self._build_correction_features(windows, baseline_pred_model)
        correction_targets = target_raw - baseline_pred_raw
        return correction_features, correction_targets, baseline_pred_raw, target_spatial

    def _build_correction_features(
        self,
        current_input_model: np.ndarray,
        baseline_pred_model: np.ndarray,
    ) -> np.ndarray:
        current = np.asarray(current_input_model, dtype=np.float64)
        baseline = np.asarray(baseline_pred_model, dtype=np.float64)

        if current.ndim == 1:
            if baseline.ndim != 1:
                raise ValueError("Single correction input requires a 1-D baseline prediction")
            return _build_t_plus_one_correction_features(current, baseline)

        if current.ndim == 2:
            if baseline.ndim == 1:
                last_state = current[-1]
                if self.correction_feature_mode == "window":
                    return np.concatenate(
                        [current.reshape(-1), baseline, baseline - last_state],
                        axis=-1,
                    )
                return _build_t_plus_one_correction_features(last_state, baseline)

            if baseline.ndim == 2:
                return _build_t_plus_one_correction_features(current, baseline)

        if current.ndim == 3:
            if baseline.ndim != 2 or baseline.shape[0] != current.shape[0]:
                raise ValueError("Batched correction windows require `(N, D)` baseline predictions")
            last_states = current[:, -1, :]
            if self.correction_feature_mode == "window":
                return np.concatenate(
                    [
                        current.reshape(current.shape[0], -1),
                        baseline,
                        baseline - last_states,
                    ],
                    axis=-1,
                )
            return _build_t_plus_one_correction_features(last_states, baseline)

        raise ValueError("correction input must have shape `(D,)`, `(T, D)`, or `(N, T, D)`")

    def _fit_correction_head(self, train_states: np.ndarray) -> None:
        if self.correction_num_epochs <= 0:
            self._correction_model = None
            self._correction_target_mean = np.zeros(self.rank, dtype=np.float64)
            self._correction_target_std = np.ones(self.rank, dtype=np.float64)
            self._correction_training_history = None
            return

        raw_coeffs = np.asarray(self._train_coeffs, dtype=np.float64)
        model_coeffs = self._to_model_coeffs(raw_coeffs)
        spatial_states = np.asarray(train_states, dtype=np.float64)
        train_raw, train_model, train_states_split, val_raw, val_model, val_states = (
            self._split_correction_trajectories(raw_coeffs, model_coeffs, spatial_states)
        )
        train_x, train_y_raw, train_baseline_raw, train_target_spatial = (
            self._build_correction_dataset(train_raw, train_model, train_states_split)
        )
        self._correction_target_mean, self._correction_target_std = (
            _compute_vector_standardization_stats(train_y_raw)
        )

        if val_raw is not None and val_model is not None and val_states is not None:
            val_x, val_y_raw, val_baseline_raw, val_target_spatial = (
                self._build_correction_dataset(val_raw, val_model, val_states)
            )
        else:
            val_x = val_y_raw = val_baseline_raw = val_target_spatial = None

        self._correction_model = MLPForecaster(
            in_dim=train_x.shape[-1],
            out_dim=train_y_raw.shape[-1],
            config=MLPForecasterConfig(
                in_dim=train_x.shape[-1],
                out_dim=train_y_raw.shape[-1],
                hidden_size=self.correction_hidden_size,
                num_layers=self.correction_num_layers,
                dropout=self.correction_dropout,
                learning_rate=self.correction_learning_rate,
                weight_decay=self.correction_weight_decay,
                num_epochs=self.correction_num_epochs,
                batch_size=self.correction_batch_size,
                val_split=self.correction_val_split,
                early_stopping_patience=self.correction_early_stopping_patience,
                delta_forecast=False,
                device="cpu",
                verbose=self.verbose,
            ),
        )
        if self.correction_spatial_loss_weight <= 0.0 and self.correction_rel_frob_loss_weight <= 0.0:
            train_y = (train_y_raw - self._correction_target_mean) / self._correction_target_std
            if val_y_raw is not None:
                val_y = (val_y_raw - self._correction_target_mean) / self._correction_target_std
            else:
                val_y = None
            _fit_explicit_mlp(self._correction_model, train_x, train_y, val_x, val_y)
            self._correction_training_history = dict(self._correction_model.training_history)
            return

        decoder_basis = self._basis_vectors.reshape(self.rank, *self._spatial_shape)
        self._correction_training_history = _fit_explicit_mlp_with_mixed_loss(
            self._correction_model,
            train_x=train_x,
            train_residual_raw=train_y_raw,
            train_baseline_raw=train_baseline_raw,
            train_target_spatial=train_target_spatial,
            residual_mean=self._correction_target_mean,
            residual_std=self._correction_target_std,
            decoder_basis=decoder_basis,
            spatial_mean=self._spatial_mean,
            latent_loss_weight=self.correction_latent_loss_weight,
            spatial_loss_weight=self.correction_spatial_loss_weight,
            rel_frob_loss_weight=self.correction_rel_frob_loss_weight,
            val_x=val_x,
            val_residual_raw=val_y_raw,
            val_baseline_raw=val_baseline_raw,
            val_target_spatial=val_target_spatial,
        )

    def _predict_residual_raw(self, correction_features: np.ndarray) -> np.ndarray:
        if self._correction_model is None:
            return np.zeros(self.rank, dtype=np.float64)

        residual_normalized = self._correction_model.predict_next(correction_features)
        return residual_normalized * self._correction_target_std + self._correction_target_mean

    def _predict_corrected_next_raw(
        self,
        current_input_model: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        baseline_pred_model = self._predict_next_model_coeff(current_input_model)
        baseline_pred_raw = self._from_model_coeffs(
            np.asarray(baseline_pred_model, dtype=np.float64).reshape(1, -1)
        )[0]

        correction_features = self._build_correction_features(
            current_input_model,
            baseline_pred_model,
        )
        corrected_raw = (
            baseline_pred_raw
            + self.correction_scale * self._predict_residual_raw(correction_features)
        )
        return baseline_pred_model, baseline_pred_raw, corrected_raw

    def _reconstruct_coeff_batch(self, coeff_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(coeff_vectors, dtype=np.float64)
        recon = (vectors @ self._basis_vectors).reshape(vectors.shape[0], *self._spatial_shape)
        return recon + self._spatial_mean

    def _compute_centered_reconstruction_metrics(
        self,
        centered_states: np.ndarray,
        coeff_series: np.ndarray,
    ) -> dict[str, float]:
        target = np.asarray(centered_states, dtype=np.float64).reshape(-1, *self._spatial_shape)
        coeffs = np.asarray(coeff_series, dtype=np.float64).reshape(-1, self.rank)
        recon = (coeffs @ self._basis_vectors).reshape(-1, *self._spatial_shape)
        metrics = _compute_regression_metrics(target, recon)
        return {
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "rel_frob_err": metrics["rel_frob_err"],
            "r2": metrics["r2"],
        }

    def evaluate_one_step(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        centered = states - self._spatial_mean
        coeffs_raw = self._states_to_coefficients(centered)
        coeffs_model = self._to_model_coeffs(coeffs_raw)
        spatial_shape = states.shape[2:]

        windows, target_model = build_lagged_windows(coeffs_model, self.lstm_seq_length)
        baseline_model = np.stack(
            [self._predict_next_model_coeff(window) for window in windows],
            axis=0,
        )
        target_raw = self._from_model_coeffs(target_model)
        baseline_raw = self._from_model_coeffs(baseline_model)
        if self._correction_model is None:
            pred_raw = baseline_raw
        else:
            pred_raw = np.stack(
                [self._predict_corrected_next_raw(window)[2] for window in windows],
                axis=0,
            )
        spatial_target = states[:, self.lstm_seq_length :, :, :].reshape(-1, *spatial_shape)
        spatial_pred = self._reconstruct_coeff_batch(pred_raw)
        baseline_spatial_pred = self._reconstruct_coeff_batch(baseline_raw)

        coeff_metrics = _compute_regression_metrics(target_raw, pred_raw)
        spatial_metrics = _compute_regression_metrics(spatial_target, spatial_pred)
        baseline_spatial_metrics = _compute_regression_metrics(spatial_target, baseline_spatial_pred)
        return {
            "coeff_mse": coeff_metrics["mse"],
            "coeff_rmse": coeff_metrics["rmse"],
            "coeff_rel_frob_err": coeff_metrics["rel_frob_err"],
            "coeff_r2": coeff_metrics["r2"],
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "baseline_spatial_r2": baseline_spatial_metrics["r2"],
            "n_eval_samples": int(target_raw.shape[0]),
            "target_spatial": spatial_target,
            "pred_spatial": spatial_pred,
            "target_coeff": target_raw,
            "pred_coeff": pred_raw,
            "baseline_pred_spatial": baseline_spatial_pred,
            "baseline_pred_coeff": baseline_raw,
        }

    def evaluate_rollout(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        centered = states - self._spatial_mean
        coeffs_raw = self._states_to_coefficients(centered)
        spatial_shape = states.shape[2:]
        warmup = self.lstm_seq_length

        pred_coeff_all = []
        baseline_coeff_all = []
        target_spatial_all = []
        for traj_idx in range(coeffs_raw.shape[0]):
            raw_history = [state.copy() for state in coeffs_raw[traj_idx, :warmup, :]]
            baseline_raw_history = [
                state.copy() for state in coeffs_raw[traj_idx, :warmup, :]
            ]
            model_history = [
                state.copy()
                for state in self._to_model_coeffs(coeffs_raw[traj_idx : traj_idx + 1, :warmup, :])[0]
            ]
            baseline_model_history = [state.copy() for state in model_history]
            target_traj = coeffs_raw[traj_idx, warmup:, :]
            pred_traj = []
            baseline_traj = []
            for _ in range(target_traj.shape[0]):
                baseline_input = np.asarray(baseline_model_history[-warmup:], dtype=np.float64)
                baseline_model = self._predict_next_model_coeff(baseline_input)
                baseline_raw = self._from_model_coeffs(baseline_model.reshape(1, -1))[0]
                baseline_traj.append(baseline_raw)
                baseline_raw_history.append(baseline_raw)
                baseline_model_history.append(self._make_latest_model_state(baseline_raw_history))

                if self._correction_model is None:
                    pred_raw = baseline_raw
                else:
                    current_input = np.asarray(model_history[-warmup:], dtype=np.float64)
                    _, _, pred_raw = self._predict_corrected_next_raw(current_input)
                pred_traj.append(pred_raw)
                raw_history.append(pred_raw)
                model_history.append(self._make_latest_model_state(raw_history))

            pred_coeff_all.append(np.asarray(pred_traj, dtype=np.float64))
            baseline_coeff_all.append(np.asarray(baseline_traj, dtype=np.float64))
            target_spatial_all.append(states[traj_idx, warmup:, :, :])

        pred_raw = np.concatenate(pred_coeff_all, axis=0)
        baseline_raw = np.concatenate(baseline_coeff_all, axis=0)
        target_raw = coeffs_raw[:, warmup:, :].reshape(-1, coeffs_raw.shape[-1])
        spatial_target = np.concatenate(target_spatial_all, axis=0)
        spatial_pred = self._reconstruct_coeff_batch(pred_raw)
        baseline_spatial_pred = self._reconstruct_coeff_batch(baseline_raw)

        coeff_metrics = _compute_regression_metrics(target_raw, pred_raw)
        spatial_metrics = _compute_regression_metrics(spatial_target, spatial_pred)
        baseline_spatial_metrics = _compute_regression_metrics(spatial_target, baseline_spatial_pred)
        return {
            "coeff_mse": coeff_metrics["mse"],
            "coeff_rmse": coeff_metrics["rmse"],
            "coeff_rel_frob_err": coeff_metrics["rel_frob_err"],
            "coeff_r2": coeff_metrics["r2"],
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "baseline_spatial_r2": baseline_spatial_metrics["r2"],
            "n_eval_samples": int(target_raw.shape[0]),
            "n_rollout_steps": int(states.shape[1] - warmup),
            "target_spatial": spatial_target,
            "pred_spatial": spatial_pred,
            "target_coeff": target_raw,
            "pred_coeff": pred_raw,
            "baseline_pred_spatial": baseline_spatial_pred,
            "baseline_pred_coeff": baseline_raw,
        }

    def get_sensor_summary(self) -> dict[str, object]:
        if self._sensor_mask is None:
            raise RuntimeError("Call fit() before sensor summary")
        return {
            "rank": self.rank,
            "requested_sensors": self.requested_n_sensors,
            "actual_sensors": int(np.sum(self._sensor_mask)),
            "sensor_selection_method": self._sensor_selection_method,
            "sensor_indices": self._sensor_indices.astype(int).tolist(),
            "coefficient_source": self.coefficient_source,
            "feature_mode": self.feature_mode,
            "cs_initialization": self.cs_initialization,
            "correction_enabled": self._correction_model is not None,
            "correction_num_epochs": self.correction_num_epochs,
            "correction_feature_mode": self.correction_feature_mode,
            "correction_scale": self.correction_scale,
            "correction_training_history": self._correction_training_history,
            "fit_cs_mean_iterations": (
                float(np.mean([m["iterations"] for m in self._fit_cs_metrics]))
                if self._fit_cs_metrics
                else None
            ),
            "fit_cs_convergence_rate": (
                float(np.mean([m["converged"] for m in self._fit_cs_metrics]))
                if self._fit_cs_metrics
                else None
            ),
            "fit_reconstruction": self._fit_reconstruction_metrics,
        }


def _build_t_plus_one_correction_features(
    last_model_state: np.ndarray,
    baseline_pred_model: np.ndarray,
) -> np.ndarray:
    """
    Concatenate the current state, baseline prediction, and the implied step.

    Accepts `(D,)` or `(N, D)` arrays and returns the same leading dimensions
    with feature size `3 * D`.
    """
    last_state = np.asarray(last_model_state, dtype=np.float64)
    baseline_pred = np.asarray(baseline_pred_model, dtype=np.float64)
    if last_state.shape != baseline_pred.shape:
        raise ValueError("last_model_state and baseline_pred_model must have identical shapes")
    return np.concatenate([last_state, baseline_pred, baseline_pred - last_state], axis=-1)


def _compute_vector_standardization_stats(
    vectors: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(vectors, dtype=np.float64).reshape(-1, vectors.shape[-1])
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def _decode_spatial_from_latent_torch(
    latent_vectors: torch.Tensor,
    decoder_basis: torch.Tensor,
    spatial_mean: torch.Tensor,
) -> torch.Tensor:
    """Differentiate through the fixed TBMD decoder in latent-space training loops."""
    return torch.einsum("nr,rhw->nhw", latent_vectors, decoder_basis) + spatial_mean.unsqueeze(0)


def _compute_mixed_one_step_loss_terms(
    *,
    pred_residual_normalized: torch.Tensor,
    target_residual_normalized: torch.Tensor,
    pred_spatial: torch.Tensor,
    target_spatial: torch.Tensor,
    latent_loss_weight: float,
    spatial_loss_weight: float,
    rel_frob_loss_weight: float,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Combine latent residual, spatial MSE, and relative Frobenius terms."""
    latent_loss = torch.mean((pred_residual_normalized - target_residual_normalized) ** 2)
    spatial_loss = torch.mean((pred_spatial - target_spatial) ** 2)

    pred_flat = pred_spatial.reshape(pred_spatial.shape[0], -1)
    target_flat = target_spatial.reshape(target_spatial.shape[0], -1)
    rel_frob = torch.mean(
        torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
        / torch.clamp(torch.linalg.vector_norm(target_flat, dim=1), min=eps)
    )

    total = (
        latent_loss_weight * latent_loss
        + spatial_loss_weight * spatial_loss
        + rel_frob_loss_weight * rel_frob
    )
    return {
        "total": total,
        "latent": latent_loss,
        "spatial": spatial_loss,
        "rel_frob": rel_frob,
    }


def _fit_explicit_mlp(
    forecaster: MLPForecaster,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: Optional[np.ndarray] = None,
    val_y: Optional[np.ndarray] = None,
) -> None:
    batch_size = forecaster.config.batch_size
    num_epochs = forecaster.config.num_epochs
    patience = forecaster.config.early_stopping_patience
    verbose = forecaster.config.verbose

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

    if val_x is not None and val_y is not None and len(val_x) > 0:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(
            torch.tensor(val_x, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=forecaster.config.shuffle,
    )

    forecaster.training_history = {"train_loss": [], "val_loss": []}
    forecaster.best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = forecaster.train_epoch(train_loader)
        forecaster.training_history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = forecaster.validate(val_loader)
            forecaster.training_history["val_loss"].append(val_loss)
            if val_loss < forecaster.best_val_loss:
                forecaster.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if verbose and (epoch + 1) % 50 == 0:
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{num_epochs} - train={train_loss:.6f} val={val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - train={train_loss:.6f}")


def _fit_explicit_mlp_with_mixed_loss(
    forecaster: MLPForecaster,
    *,
    train_x: np.ndarray,
    train_residual_raw: np.ndarray,
    train_baseline_raw: np.ndarray,
    train_target_spatial: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    decoder_basis: np.ndarray,
    spatial_mean: np.ndarray,
    latent_loss_weight: float,
    spatial_loss_weight: float,
    rel_frob_loss_weight: float,
    val_x: Optional[np.ndarray] = None,
    val_residual_raw: Optional[np.ndarray] = None,
    val_baseline_raw: Optional[np.ndarray] = None,
    val_target_spatial: Optional[np.ndarray] = None,
) -> Dict[str, list[float]]:
    """Train the correction MLP against a mixed latent/spatial one-step objective."""
    batch_size = forecaster.config.batch_size
    num_epochs = forecaster.config.num_epochs
    patience = forecaster.config.early_stopping_patience
    verbose = forecaster.config.verbose
    device = forecaster.device

    residual_mean_t = torch.tensor(residual_mean, dtype=torch.float32, device=device)
    residual_std_t = torch.tensor(residual_std, dtype=torch.float32, device=device)
    decoder_basis_t = torch.tensor(decoder_basis, dtype=torch.float32, device=device)
    spatial_mean_t = torch.tensor(spatial_mean, dtype=torch.float32, device=device)

    def _make_dataset(
        x: np.ndarray,
        residual_raw: np.ndarray,
        baseline_raw: np.ndarray,
        target_spatial: np.ndarray,
    ) -> TensorDataset:
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(residual_raw, dtype=torch.float32),
            torch.tensor(baseline_raw, dtype=torch.float32),
            torch.tensor(target_spatial, dtype=torch.float32),
        )

    train_loader = DataLoader(
        _make_dataset(train_x, train_residual_raw, train_baseline_raw, train_target_spatial),
        batch_size=batch_size,
        shuffle=forecaster.config.shuffle,
    )

    if (
        val_x is not None
        and val_residual_raw is not None
        and val_baseline_raw is not None
        and val_target_spatial is not None
        and len(val_x) > 0
    ):
        val_loader: Optional[DataLoader] = DataLoader(
            _make_dataset(val_x, val_residual_raw, val_baseline_raw, val_target_spatial),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    history: Dict[str, list[float]] = {
        "train_total_loss": [],
        "train_latent_loss": [],
        "train_spatial_loss": [],
        "train_rel_frob_loss": [],
        "val_total_loss": [],
        "val_latent_loss": [],
        "val_spatial_loss": [],
        "val_rel_frob_loss": [],
    }
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    def _run_epoch(loader: DataLoader, *, train: bool) -> Dict[str, float]:
        if train:
            forecaster.model.train()
        else:
            forecaster.model.eval()

        totals = {"total": 0.0, "latent": 0.0, "spatial": 0.0, "rel_frob": 0.0}
        sample_count = 0

        for features, residual_raw, baseline_raw, target_spatial in loader:
            features = features.to(device)
            residual_raw = residual_raw.to(device)
            baseline_raw = baseline_raw.to(device)
            target_spatial = target_spatial.to(device)

            target_residual_normalized = (residual_raw - residual_mean_t) / residual_std_t

            with torch.set_grad_enabled(train):
                pred_residual_normalized = forecaster.model(features)
                pred_residual_raw = pred_residual_normalized * residual_std_t + residual_mean_t
                corrected_raw = baseline_raw + pred_residual_raw
                pred_spatial = _decode_spatial_from_latent_torch(
                    corrected_raw,
                    decoder_basis_t,
                    spatial_mean_t,
                )
                losses = _compute_mixed_one_step_loss_terms(
                    pred_residual_normalized=pred_residual_normalized,
                    target_residual_normalized=target_residual_normalized,
                    pred_spatial=pred_spatial,
                    target_spatial=target_spatial,
                    latent_loss_weight=latent_loss_weight,
                    spatial_loss_weight=spatial_loss_weight,
                    rel_frob_loss_weight=rel_frob_loss_weight,
                )

                if train:
                    forecaster.optimizer.zero_grad()
                    losses["total"].backward()
                    forecaster.optimizer.step()

            batch_size_local = int(features.shape[0])
            sample_count += batch_size_local
            for key in totals:
                totals[key] += float(losses[key].detach().cpu().item()) * batch_size_local

        return {key: totals[key] / max(sample_count, 1) for key in totals}

    for epoch in range(num_epochs):
        train_metrics = _run_epoch(train_loader, train=True)
        history["train_total_loss"].append(train_metrics["total"])
        history["train_latent_loss"].append(train_metrics["latent"])
        history["train_spatial_loss"].append(train_metrics["spatial"])
        history["train_rel_frob_loss"].append(train_metrics["rel_frob"])

        if val_loader is not None:
            val_metrics = _run_epoch(val_loader, train=False)
            history["val_total_loss"].append(val_metrics["total"])
            history["val_latent_loss"].append(val_metrics["latent"])
            history["val_spatial_loss"].append(val_metrics["spatial"])
            history["val_rel_frob_loss"].append(val_metrics["rel_frob"])

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in forecaster.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if verbose and (epoch + 1) % 50 == 0:
            if val_loader is not None:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"train_total={train_metrics['total']:.6f} "
                    f"train_spatial={train_metrics['spatial']:.6f} "
                    f"val_total={val_metrics['total']:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"train_total={train_metrics['total']:.6f} "
                    f"train_spatial={train_metrics['spatial']:.6f}"
                )

    if best_state is not None:
        forecaster.model.load_state_dict(best_state)

    forecaster.training_history = {
        "train_loss": history["train_total_loss"],
        "val_loss": history["val_total_loss"],
    }
    forecaster.best_val_loss = best_val_loss
    return history


def _fit_explicit_lstm(
    forecaster: LSTMForecaster,
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    val_windows: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None,
) -> None:
    batch_size = forecaster.config.batch_size
    num_epochs = forecaster.config.num_epochs
    patience = forecaster.config.early_stopping_patience
    verbose = forecaster.config.verbose

    train_x_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_targets, dtype=torch.float32)

    if val_windows is not None and val_targets is not None and len(val_windows) > 0:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(
            torch.tensor(val_windows, dtype=torch.float32),
            torch.tensor(val_targets, dtype=torch.float32),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=forecaster.config.shuffle,
    )

    forecaster.training_history = {"train_loss": [], "val_loss": []}
    forecaster.best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = forecaster.train_epoch(train_loader, epoch=epoch)
        forecaster.training_history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = forecaster.validate(val_loader)
            forecaster.training_history["val_loss"].append(val_loss)
            if val_loss < forecaster.best_val_loss:
                forecaster.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if verbose and (epoch + 1) % 50 == 0:
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{num_epochs} - train={train_loss:.6f} val={val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - train={train_loss:.6f}")


class TrajectoryAwareLatentForecaster:
    """
    Navier-Stokes-specific latent forecaster that respects trajectory boundaries.
    """

    def __init__(
        self,
        config: Optional[LatentModalForecasterConfig] = None,
        *,
        feature_mode: str = "latent",
    ):
        self.config = config or LatentModalForecasterConfig(verbose=False)
        if feature_mode not in {"latent", "latent_plus_delta"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        self._adapter = LatentModalForecaster(config=self.config)
        self._feature_mode = feature_mode
        self._sub_forecaster = None
        self._train_states: Optional[np.ndarray] = None
        self._train_latent: Optional[np.ndarray] = None
        self._latent_dim: Optional[int] = None
        self._spatial_mean: Optional[np.ndarray] = None
        self._latent_mean: Optional[np.ndarray] = None
        self._latent_std: Optional[np.ndarray] = None
        self._matrix_basis: Optional[np.ndarray] = None
        self._spatial_decoder_basis: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareLatentForecaster":
        states = np.asarray(train_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("train_states must have shape `(B, T, H, W)`")

        self._train_states = states
        self._matrix_basis = None
        self._spatial_decoder_basis = None
        self._spatial_mean = (
            np.mean(states, axis=(0, 1))
            if self.config.spatial_mean_centering
            else np.zeros(states.shape[2:], dtype=np.float64)
        )
        centered_states = states - self._spatial_mean

        if _can_use_full_spatial_rank_fast_path(centered_states, self.config.ranks):
            core, factors, decomp_error, ranks, self._matrix_basis = _fit_matrix_latent_basis(
                centered_states,
                self.config.ranks,
                self.config.random_state,
            )
        else:
            train_tensor = _states_to_decomposition_tensor(centered_states)
            core, factors, decomp_error, ranks = self._adapter._decompose_training_data(
                train_tensor,
                self.config.ranks,
            )

        spatial_mode_1, spatial_mode_2, temporal_coeffs = factors
        n_trajectories, steps = states.shape[:2]
        latent_dim = temporal_coeffs.shape[1]
        self._latent_dim = latent_dim
        self._train_latent = temporal_coeffs.reshape(n_trajectories, steps, latent_dim)
        if self.config.latent_normalization:
            self._latent_mean, self._latent_std = _compute_latent_standardization_stats(self._train_latent)
        else:
            self._latent_mean = np.zeros(latent_dim, dtype=np.float64)
            self._latent_std = np.ones(latent_dim, dtype=np.float64)

        self._adapter._result = LatentModalResult(
            core=core,
            spatial_mode_1=spatial_mode_1,
            spatial_mode_2=spatial_mode_2,
            temporal_coeffs_train=temporal_coeffs,
            temporal_coeffs_test=np.empty((0, latent_dim), dtype=np.float64),
            ranks=ranks,
            decomposition_error=decomp_error,
        )
        self._adapter._fitted = True

        self._fit_sub_forecaster(latent_dim)
        self._fitted = True
        return self

    def _to_model_latent(self, latent_series: np.ndarray) -> np.ndarray:
        standardized = _apply_latent_standardization(latent_series, self._latent_mean, self._latent_std)
        if self._feature_mode == "latent":
            return standardized
        return _augment_latent_series_with_deltas(standardized)

    def _from_model_latent(self, latent_series: np.ndarray) -> np.ndarray:
        series = np.asarray(latent_series, dtype=np.float64)
        if self._feature_mode == "latent_plus_delta":
            series = series[..., : self._latent_dim]
        return _invert_latent_standardization(series, self._latent_mean, self._latent_std)

    def _fit_sub_forecaster(self, latent_dim: int) -> None:
        train_latent_model = self._to_model_latent(self._train_latent)
        model_dim = train_latent_model.shape[-1]

        if self.config.forecaster_type == "linear":
            x_pairs, y_pairs = build_one_step_pairs(
                train_latent_model,
                predict_deltas=self.config.delta_forecast,
            )
            model = LinearForecaster(
                config=LinearForecasterConfig(device=self._adapter._device, verbose=self.config.verbose)
            )
            model.M = np.linalg.pinv(x_pairs, rcond=1e-3) @ y_pairs
            model.trained = True
            self._sub_forecaster = model
            return

        if self.config.forecaster_type == "mlp":
            model = MLPForecaster(
                in_dim=model_dim,
                out_dim=model_dim,
                config=MLPForecasterConfig(
                    in_dim=model_dim,
                    out_dim=model_dim,
                    hidden_size=self.config.mlp_hidden_size,
                    num_layers=self.config.mlp_num_layers,
                    dropout=self.config.mlp_dropout,
                    learning_rate=self.config.mlp_learning_rate,
                    weight_decay=self.config.mlp_weight_decay,
                    num_epochs=self.config.mlp_num_epochs,
                    batch_size=self.config.mlp_batch_size,
                    val_split=self.config.mlp_val_split,
                    early_stopping_patience=self.config.mlp_early_stopping_patience,
                    delta_forecast=self.config.delta_forecast,
                    device=self._adapter._device,
                    verbose=self.config.verbose,
                ),
            )
            train_series, val_series = _split_trajectory_series_for_validation(
                train_latent_model,
                self.config.mlp_val_split,
            )
            train_x, train_y = build_one_step_pairs(
                train_series,
                predict_deltas=self.config.delta_forecast,
            )
            if val_series is not None:
                val_x, val_y = build_one_step_pairs(
                    val_series,
                    predict_deltas=self.config.delta_forecast,
                )
            else:
                val_x, val_y = None, None
            _fit_explicit_mlp(model, train_x, train_y, val_x, val_y)
            self._sub_forecaster = model
            return

        if self.config.forecaster_type == "lstm":
            config = LSTMForecasterConfig(
                in_dim=model_dim,
                out_dim=model_dim,
                seq_length=self.config.lstm_seq_length,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                learning_rate=self.config.lstm_learning_rate,
                num_epochs=self.config.lstm_num_epochs,
                batch_size=self.config.lstm_batch_size,
                val_split=self.config.lstm_val_split,
                early_stopping_patience=self.config.lstm_early_stopping_patience,
                delta_forecast=self.config.delta_forecast,
                use_scheduled_sampling=self.config.lstm_use_scheduled_sampling,
                ss_unroll_steps=self.config.lstm_ss_unroll_steps,
                ss_decay_rate=self.config.lstm_ss_decay_rate,
                ss_min_prob=self.config.lstm_ss_min_prob,
                device=self._adapter._device,
                verbose=self.config.verbose,
            )
            
            train_series, val_series = _split_trajectory_series_for_validation(
                train_latent_model,
                self.config.lstm_val_split,
            )

            if self.config.lstm_use_scheduled_sampling:
                model = ScheduledSamplingLSTMForecaster(
                    in_dim=model_dim,
                    out_dim=model_dim,
                    config=config,
                )
                train_windows, train_targets = build_unrolled_lagged_windows(
                    train_series,
                    self.config.lstm_seq_length,
                    self.config.lstm_ss_unroll_steps,
                    predict_deltas=self.config.delta_forecast,
                )
                if val_series is not None:
                    val_windows, val_targets = build_unrolled_lagged_windows(
                        val_series,
                        self.config.lstm_seq_length,
                        self.config.lstm_ss_unroll_steps,
                        predict_deltas=self.config.delta_forecast,
                    )
                else:
                    val_windows, val_targets = None, None
            else:
                model = LSTMForecaster(
                    in_dim=model_dim,
                    out_dim=model_dim,
                    config=config,
                )
                train_windows, train_targets = build_lagged_windows(
                    train_series,
                    self.config.lstm_seq_length,
                    predict_deltas=self.config.delta_forecast,
                )
                if val_series is not None:
                    val_windows, val_targets = build_lagged_windows(
                        val_series,
                        self.config.lstm_seq_length,
                        predict_deltas=self.config.delta_forecast,
                    )
                else:
                    val_windows, val_targets = None, None

            _fit_explicit_lstm(model, train_windows, train_targets, val_windows, val_targets)
            self._sub_forecaster = model
            return

        raise ValueError(f"Unsupported forecaster_type: {self.config.forecaster_type}")

    def _project_states_to_latent(self, states: np.ndarray) -> np.ndarray:
        centered_states = np.asarray(states, dtype=np.float64) - self._spatial_mean
        if self._matrix_basis is not None:
            flat = centered_states.reshape(-1, int(np.prod(centered_states.shape[2:])))
            projected = flat @ self._matrix_basis.T
        else:
            flat = _states_to_decomposition_tensor(centered_states)
            projected = self._adapter._project_to_latent_batch(
                flat,
                self._adapter._result.spatial_mode_1,
                self._adapter._result.spatial_mode_2,
                self._adapter._result.core,
            )
        n_trajectories, steps = states.shape[:2]
        latent_dim = projected.shape[1]
        return projected.reshape(n_trajectories, steps, latent_dim)

    def _get_spatial_decoder_basis(self) -> np.ndarray:
        if self._spatial_decoder_basis is not None:
            return self._spatial_decoder_basis

        spatial_shape = self._train_states.shape[2:]
        if self._matrix_basis is not None:
            self._spatial_decoder_basis = self._matrix_basis.reshape(self._latent_dim, *spatial_shape)
            return self._spatial_decoder_basis

        result = self._adapter._result
        basis = np.zeros((self._latent_dim, *spatial_shape), dtype=np.float64)
        for idx in range(self._latent_dim):
            basis[idx] = result.spatial_mode_1 @ result.core[:, :, idx] @ result.spatial_mode_2.T
        self._spatial_decoder_basis = basis
        return basis

    def _reconstruct_latent_batch(self, latent_vectors: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
        vectors = np.asarray(latent_vectors, dtype=np.float64)
        if self._matrix_basis is not None:
            recon = (vectors @ self._matrix_basis).reshape(vectors.shape[0], *spatial_shape)
            return recon + self._spatial_mean

        preds = np.zeros((vectors.shape[0], *spatial_shape), dtype=np.float64)
        for idx, c in enumerate(vectors):
            preds[idx] = self._adapter._reconstruct_from_latent(c)
        return preds + self._spatial_mean

    def _predict_next_model_latent(self, current_input: np.ndarray) -> np.ndarray:
        if self.config.forecaster_type == "lstm":
            return self._sub_forecaster.predict_next(current_input)

        pred = self._sub_forecaster.predict_next(current_input)
        if self.config.delta_forecast:
            return current_input + pred
        return pred

    def _predict_sequence_model_latent(self, current_input: np.ndarray, n_steps: int) -> np.ndarray:
        if self.config.forecaster_type == "lstm":
            return self._sub_forecaster.predict_sequence(current_input, n_steps)

        current = np.asarray(current_input, dtype=np.float64).copy()
        sequence = np.zeros((n_steps, current.shape[-1]), dtype=np.float64)
        for idx in range(n_steps):
            current = self._predict_next_model_latent(current)
            sequence[idx] = current
        return sequence

    def evaluate_one_step(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        latent = self._project_states_to_latent(states)
        latent_model = self._to_model_latent(latent)
        spatial_shape = states.shape[2:]

        if self.config.forecaster_type == "lstm":
            windows, latent_target_model = build_lagged_windows(latent_model, self.config.lstm_seq_length)
            latent_pred_model = np.stack([self._predict_next_model_latent(window) for window in windows], axis=0)
            spatial_target = states[:, self.config.lstm_seq_length :, :, :].reshape(-1, *spatial_shape)
        else:
            latent_input, latent_target_model = build_one_step_pairs(latent_model)
            latent_pred_model = np.stack([self._predict_next_model_latent(x) for x in latent_input], axis=0)
            spatial_target = states[:, 1:, :, :].reshape(-1, *spatial_shape)

        latent_target = self._from_model_latent(latent_target_model)
        latent_pred = self._from_model_latent(latent_pred_model)
        spatial_pred = self._reconstruct_latent_batch(latent_pred, spatial_shape)
        latent_metrics = _compute_regression_metrics(latent_target, latent_pred)
        spatial_metrics = _compute_regression_metrics(spatial_target, spatial_pred)

        return {
            "latent_mse": latent_metrics["mse"],
            "latent_rmse": latent_metrics["rmse"],
            "latent_rel_frob_err": latent_metrics["rel_frob_err"],
            "latent_r2": latent_metrics["r2"],
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(latent_target.shape[0]),
            "target_spatial": spatial_target,
            "pred_spatial": spatial_pred,
            "target_latent": latent_target,
            "pred_latent": latent_pred,
        }

    def evaluate_rollout(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        latent = self._project_states_to_latent(states)
        latent_model = self._to_model_latent(latent)
        spatial_shape = states.shape[2:]

        latent_target_model_all = []
        latent_pred_model_all = []
        spatial_target_all = []

        if self.config.forecaster_type == "lstm":
            warmup = self.config.lstm_seq_length
            for traj_idx in range(latent_model.shape[0]):
                target_model = latent_model[traj_idx, warmup:, :]
                pred_model = self._predict_sequence_model_latent(
                    latent_model[traj_idx, :warmup, :],
                    len(target_model),
                )
                latent_target_model_all.append(target_model)
                latent_pred_model_all.append(pred_model)
                spatial_target_all.append(states[traj_idx, warmup:, :, :])
            n_rollout_steps = states.shape[1] - warmup
        else:
            for traj_idx in range(latent_model.shape[0]):
                target_model = latent_model[traj_idx, 1:, :]
                pred_model = self._predict_sequence_model_latent(
                    latent_model[traj_idx, 0, :],
                    len(target_model),
                )
                latent_target_model_all.append(target_model)
                latent_pred_model_all.append(pred_model)
                spatial_target_all.append(states[traj_idx, 1:, :, :])
            n_rollout_steps = states.shape[1] - 1

        latent_target_model = np.concatenate(latent_target_model_all, axis=0)
        latent_pred_model = np.concatenate(latent_pred_model_all, axis=0)
        latent_target = self._from_model_latent(latent_target_model)
        latent_pred = self._from_model_latent(latent_pred_model)
        spatial_target = np.concatenate(spatial_target_all, axis=0)
        spatial_pred = self._reconstruct_latent_batch(latent_pred, spatial_shape)

        latent_metrics = _compute_regression_metrics(latent_target, latent_pred)
        spatial_metrics = _compute_regression_metrics(spatial_target, spatial_pred)

        return {
            "latent_mse": latent_metrics["mse"],
            "latent_rmse": latent_metrics["rmse"],
            "latent_rel_frob_err": latent_metrics["rel_frob_err"],
            "latent_r2": latent_metrics["r2"],
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(latent_target.shape[0]),
            "n_rollout_steps": int(n_rollout_steps),
            "target_spatial": spatial_target,
            "pred_spatial": spatial_pred,
            "target_latent": latent_target,
            "pred_latent": latent_pred,
        }


class TrajectoryAwareResidualCorrectedForecaster:
    """
    Two-stage Navier-Stokes forecaster for improving one-step (`t+1`) accuracy.

    Stage 1 uses an ordinary trajectory-aware latent forecaster. Stage 2 learns
    a lightweight residual correction head in absolute latent space:

    `c̃_{t+1} = ĉ_{t+1} + δc_{t+1}`.
    """

    def __init__(
        self,
        config: Optional[LatentModalForecasterConfig] = None,
        *,
        feature_mode: str = "latent",
        correction_hidden_size: int = 64,
        correction_num_layers: int = 2,
        correction_dropout: float = 0.0,
        correction_learning_rate: float = 1e-3,
        correction_weight_decay: float = 1e-5,
        correction_num_epochs: int = 120,
        correction_batch_size: int = 32,
        correction_val_split: float = 0.2,
        correction_early_stopping_patience: int = 20,
        correction_latent_loss_weight: float = 1.0,
        correction_spatial_loss_weight: float = 0.0,
        correction_rel_frob_loss_weight: float = 0.0,
    ):
        self.config = config or LatentModalForecasterConfig(verbose=False)
        self._feature_mode = feature_mode
        self._base_forecaster = TrajectoryAwareLatentForecaster(
            config=self.config,
            feature_mode=feature_mode,
        )
        self._correction_hidden_size = correction_hidden_size
        self._correction_num_layers = correction_num_layers
        self._correction_dropout = correction_dropout
        self._correction_learning_rate = correction_learning_rate
        self._correction_weight_decay = correction_weight_decay
        self._correction_num_epochs = correction_num_epochs
        self._correction_batch_size = correction_batch_size
        self._correction_val_split = correction_val_split
        self._correction_early_stopping_patience = correction_early_stopping_patience
        self._correction_latent_loss_weight = correction_latent_loss_weight
        self._correction_spatial_loss_weight = correction_spatial_loss_weight
        self._correction_rel_frob_loss_weight = correction_rel_frob_loss_weight
        self._correction_model: Optional[MLPForecaster] = None
        self._correction_target_mean: Optional[np.ndarray] = None
        self._correction_target_std: Optional[np.ndarray] = None
        self._correction_training_history: Optional[Dict[str, list[float]]] = None
        self._latent_dim: Optional[int] = None
        self._fitted = False

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareResidualCorrectedForecaster":
        self._base_forecaster.fit(train_states)
        self._latent_dim = self._base_forecaster._latent_dim
        self._fit_correction_head()
        self._fitted = True
        return self

    def _split_correction_trajectories(
        self,
        raw_series: np.ndarray,
        model_series: np.ndarray,
        spatial_states: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if self._correction_val_split <= 0 or raw_series.shape[0] < 2:
            return raw_series, model_series, spatial_states, None, None, None

        val_count = max(1, int(round(raw_series.shape[0] * self._correction_val_split)))
        if val_count >= raw_series.shape[0]:
            return raw_series, model_series, spatial_states, None, None, None

        split_idx = raw_series.shape[0] - val_count
        return (
            raw_series[:split_idx],
            model_series[:split_idx],
            spatial_states[:split_idx],
            raw_series[split_idx:],
            model_series[split_idx:],
            spatial_states[split_idx:],
        )

    def _build_correction_dataset(
        self,
        raw_latent_series: np.ndarray,
        model_latent_series: np.ndarray,
        spatial_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.config.forecaster_type == "lstm":
            windows, _ = build_lagged_windows(
                model_latent_series,
                self.config.lstm_seq_length,
            )
            last_states = windows[:, -1, :]
            baseline_pred_model = np.stack(
                [self._base_forecaster._predict_next_model_latent(window) for window in windows],
                axis=0,
            )
            target_raw = raw_latent_series[:, self.config.lstm_seq_length :, :].reshape(
                -1,
                raw_latent_series.shape[-1],
            )
            target_spatial = spatial_states[:, self.config.lstm_seq_length :, :, :].reshape(
                -1,
                *spatial_states.shape[2:],
            )
        else:
            current_model, _ = build_one_step_pairs(model_latent_series)
            last_states = current_model
            baseline_pred_model = np.stack(
                [self._base_forecaster._predict_next_model_latent(state) for state in current_model],
                axis=0,
            )
            target_raw = raw_latent_series[:, 1:, :].reshape(-1, raw_latent_series.shape[-1])
            target_spatial = spatial_states[:, 1:, :, :].reshape(-1, *spatial_states.shape[2:])

        baseline_pred_raw = self._base_forecaster._from_model_latent(baseline_pred_model)
        correction_features = _build_t_plus_one_correction_features(last_states, baseline_pred_model)
        correction_targets = target_raw - baseline_pred_raw
        return correction_features, correction_targets, baseline_pred_raw, target_spatial

    def _fit_correction_head(self) -> None:
        if self._correction_num_epochs <= 0:
            self._correction_model = None
            self._correction_target_mean = np.zeros(self._latent_dim, dtype=np.float64)
            self._correction_target_std = np.ones(self._latent_dim, dtype=np.float64)
            self._correction_training_history = None
            return

        raw_latent = np.asarray(self._base_forecaster._train_latent, dtype=np.float64)
        model_latent = self._base_forecaster._to_model_latent(raw_latent)
        spatial_states = np.asarray(self._base_forecaster._train_states, dtype=np.float64)
        train_raw, train_model, train_states, val_raw, val_model, val_states = self._split_correction_trajectories(
            raw_latent,
            model_latent,
            spatial_states,
        )
        train_x, train_y_raw, train_baseline_raw, train_target_spatial = self._build_correction_dataset(
            train_raw,
            train_model,
            train_states,
        )
        self._correction_target_mean, self._correction_target_std = _compute_vector_standardization_stats(
            train_y_raw
        )

        if val_raw is not None and val_model is not None and val_states is not None:
            val_x, val_y_raw, val_baseline_raw, val_target_spatial = self._build_correction_dataset(
                val_raw,
                val_model,
                val_states,
            )
        else:
            val_x = val_y_raw = val_baseline_raw = val_target_spatial = None

        self._correction_model = MLPForecaster(
            in_dim=train_x.shape[-1],
            out_dim=train_y_raw.shape[-1],
            config=MLPForecasterConfig(
                in_dim=train_x.shape[-1],
                out_dim=train_y_raw.shape[-1],
                hidden_size=self._correction_hidden_size,
                num_layers=self._correction_num_layers,
                dropout=self._correction_dropout,
                learning_rate=self._correction_learning_rate,
                weight_decay=self._correction_weight_decay,
                num_epochs=self._correction_num_epochs,
                batch_size=self._correction_batch_size,
                val_split=self._correction_val_split,
                early_stopping_patience=self._correction_early_stopping_patience,
                delta_forecast=False,
                device=self._base_forecaster._adapter._device,
                verbose=self.config.verbose,
            ),
        )
        if (
            self._correction_spatial_loss_weight <= 0.0
            and self._correction_rel_frob_loss_weight <= 0.0
        ):
            train_y = (train_y_raw - self._correction_target_mean) / self._correction_target_std
            if val_y_raw is not None:
                val_y = (val_y_raw - self._correction_target_mean) / self._correction_target_std
            else:
                val_y = None
            _fit_explicit_mlp(self._correction_model, train_x, train_y, val_x, val_y)
            self._correction_training_history = dict(self._correction_model.training_history)
            return

        self._correction_training_history = _fit_explicit_mlp_with_mixed_loss(
            self._correction_model,
            train_x=train_x,
            train_residual_raw=train_y_raw,
            train_baseline_raw=train_baseline_raw,
            train_target_spatial=train_target_spatial,
            residual_mean=self._correction_target_mean,
            residual_std=self._correction_target_std,
            decoder_basis=self._base_forecaster._get_spatial_decoder_basis(),
            spatial_mean=self._base_forecaster._spatial_mean,
            latent_loss_weight=self._correction_latent_loss_weight,
            spatial_loss_weight=self._correction_spatial_loss_weight,
            rel_frob_loss_weight=self._correction_rel_frob_loss_weight,
            val_x=val_x,
            val_residual_raw=val_y_raw,
            val_baseline_raw=val_baseline_raw,
            val_target_spatial=val_target_spatial,
        )

    def _predict_residual_raw(self, correction_features: np.ndarray) -> np.ndarray:
        if self._correction_model is None:
            return np.zeros(self._latent_dim, dtype=np.float64)

        residual_normalized = self._correction_model.predict_next(correction_features)
        return residual_normalized * self._correction_target_std + self._correction_target_mean

    def _predict_corrected_next_raw(
        self,
        current_input_model: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        baseline_pred_model = self._base_forecaster._predict_next_model_latent(current_input_model)
        baseline_pred_raw = self._base_forecaster._from_model_latent(
            np.asarray(baseline_pred_model, dtype=np.float64).reshape(1, -1)
        )[0]

        if np.asarray(current_input_model).ndim == 2:
            last_state = np.asarray(current_input_model, dtype=np.float64)[-1]
        else:
            last_state = np.asarray(current_input_model, dtype=np.float64)

        correction_features = _build_t_plus_one_correction_features(
            last_state.reshape(1, -1),
            np.asarray(baseline_pred_model, dtype=np.float64).reshape(1, -1),
        )[0]
        corrected_raw = baseline_pred_raw + self._predict_residual_raw(correction_features)
        return baseline_pred_model, baseline_pred_raw, corrected_raw

    def _make_latest_model_state(self, raw_history: list[np.ndarray]) -> np.ndarray:
        tail = np.asarray(raw_history[-2:], dtype=np.float64)
        if tail.ndim == 1:
            tail = tail[np.newaxis, :]
        tail = tail[np.newaxis, ...]
        return self._base_forecaster._to_model_latent(tail)[0, -1]

    def evaluate_one_step(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        latent_raw = self._base_forecaster._project_states_to_latent(states)
        latent_model = self._base_forecaster._to_model_latent(latent_raw)
        spatial_shape = states.shape[2:]

        if self.config.forecaster_type == "lstm":
            inputs, _ = build_lagged_windows(latent_model, self.config.lstm_seq_length)
            target_raw = latent_raw[:, self.config.lstm_seq_length :, :].reshape(-1, latent_raw.shape[-1])
            spatial_target = states[:, self.config.lstm_seq_length :, :, :].reshape(-1, *spatial_shape)
        else:
            inputs, _ = build_one_step_pairs(latent_model)
            target_raw = latent_raw[:, 1:, :].reshape(-1, latent_raw.shape[-1])
            spatial_target = states[:, 1:, :, :].reshape(-1, *spatial_shape)

        baseline_pred_raw = []
        corrected_pred_raw = []
        for current_input in inputs:
            _, baseline_raw, corrected_raw = self._predict_corrected_next_raw(current_input)
            baseline_pred_raw.append(baseline_raw)
            corrected_pred_raw.append(corrected_raw)

        baseline_pred_raw = np.asarray(baseline_pred_raw, dtype=np.float64)
        corrected_pred_raw = np.asarray(corrected_pred_raw, dtype=np.float64)
        baseline_pred_spatial = self._base_forecaster._reconstruct_latent_batch(
            baseline_pred_raw,
            spatial_shape,
        )
        corrected_pred_spatial = self._base_forecaster._reconstruct_latent_batch(
            corrected_pred_raw,
            spatial_shape,
        )

        latent_metrics = _compute_regression_metrics(target_raw, corrected_pred_raw)
        spatial_metrics = _compute_regression_metrics(spatial_target, corrected_pred_spatial)
        baseline_spatial_metrics = _compute_regression_metrics(spatial_target, baseline_pred_spatial)

        return {
            "latent_mse": latent_metrics["mse"],
            "latent_rmse": latent_metrics["rmse"],
            "latent_rel_frob_err": latent_metrics["rel_frob_err"],
            "latent_r2": latent_metrics["r2"],
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "baseline_spatial_r2": baseline_spatial_metrics["r2"],
            "n_eval_samples": int(target_raw.shape[0]),
            "target_spatial": spatial_target,
            "pred_spatial": corrected_pred_spatial,
            "target_latent": target_raw,
            "pred_latent": corrected_pred_raw,
            "baseline_pred_spatial": baseline_pred_spatial,
            "baseline_pred_latent": baseline_pred_raw,
        }

    def evaluate_rollout(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        latent_raw = self._base_forecaster._project_states_to_latent(states)
        spatial_shape = states.shape[2:]
        warmup = self.config.lstm_seq_length if self.config.forecaster_type == "lstm" else 1

        baseline_all = []
        corrected_all = []
        target_all = []

        for traj_idx in range(latent_raw.shape[0]):
            raw_history = [state.copy() for state in latent_raw[traj_idx, :warmup, :]]
            model_history = [
                state.copy()
                for state in self._base_forecaster._to_model_latent(
                    latent_raw[traj_idx : traj_idx + 1, :warmup, :]
                )[0]
            ]

            target_traj = latent_raw[traj_idx, warmup:, :]
            spatial_target_traj = states[traj_idx, warmup:, :, :]

            baseline_traj = []
            corrected_traj = []
            for _ in range(target_traj.shape[0]):
                if self.config.forecaster_type == "lstm":
                    current_input = np.asarray(model_history[-warmup:], dtype=np.float64)
                else:
                    current_input = np.asarray(model_history[-1], dtype=np.float64)

                _, baseline_raw, corrected_raw = self._predict_corrected_next_raw(current_input)
                baseline_traj.append(baseline_raw)
                corrected_traj.append(corrected_raw)

                raw_history.append(corrected_raw)
                model_history.append(self._make_latest_model_state(raw_history))

            baseline_all.append(np.asarray(baseline_traj, dtype=np.float64))
            corrected_all.append(np.asarray(corrected_traj, dtype=np.float64))
            target_all.append(spatial_target_traj)

        baseline_pred_raw = np.concatenate(baseline_all, axis=0)
        corrected_pred_raw = np.concatenate(corrected_all, axis=0)
        spatial_target = np.concatenate(target_all, axis=0)
        baseline_pred_spatial = self._base_forecaster._reconstruct_latent_batch(
            baseline_pred_raw,
            spatial_shape,
        )
        corrected_pred_spatial = self._base_forecaster._reconstruct_latent_batch(
            corrected_pred_raw,
            spatial_shape,
        )
        latent_target = latent_raw[:, warmup:, :].reshape(-1, latent_raw.shape[-1])

        latent_metrics = _compute_regression_metrics(latent_target, corrected_pred_raw)
        spatial_metrics = _compute_regression_metrics(spatial_target, corrected_pred_spatial)
        baseline_spatial_metrics = _compute_regression_metrics(spatial_target, baseline_pred_spatial)

        return {
            "latent_mse": latent_metrics["mse"],
            "latent_rmse": latent_metrics["rmse"],
            "latent_rel_frob_err": latent_metrics["rel_frob_err"],
            "latent_r2": latent_metrics["r2"],
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "baseline_spatial_r2": baseline_spatial_metrics["r2"],
            "n_eval_samples": int(latent_target.shape[0]),
            "n_rollout_steps": int(states.shape[1] - warmup),
            "target_spatial": spatial_target,
            "pred_spatial": corrected_pred_spatial,
            "target_latent": latent_target,
            "pred_latent": corrected_pred_raw,
            "baseline_pred_spatial": baseline_pred_spatial,
            "baseline_pred_latent": baseline_pred_raw,
        }


class TrajectoryAwareMultiResolutionForecaster:
    """
    Trajectory-aware multi-resolution wrapper for Navier-Stokes experiments.

    This implementation is intentionally experiment-scoped and currently targets
    the existing use case where all levels use one-step-compatible forecasters.
    """

    def __init__(self, config: Optional[MultiResolutionTBMDConfig] = None):
        self.config = config or MultiResolutionTBMDConfig(verbose=False)
        self._levels: list[TrajectoryAwareLatentForecaster] = []
        self._train_residuals: list[np.ndarray] = []
        self._test_residuals: list[np.ndarray] = []
        self._fitted = False

    def _make_level_config(self, level_idx: int) -> LatentModalForecasterConfig:
        return LatentModalForecasterConfig(
            ranks=self.config.level_ranks[level_idx],
            forecaster_type=self.config.level_forecaster_types[level_idx],
            train_ratio=self.config.train_ratio,
            epsilon=self.config.epsilon,
            random_state=self.config.random_state,
            spatial_mean_centering=self.config.spatial_mean_centering,
            latent_normalization=self.config.latent_normalization,
            delta_forecast=self.config.delta_forecast,
            projection_refinement_steps=self.config.projection_refinement_steps,
            projection_refinement_alpha=self.config.projection_refinement_alpha,
            mlp_hidden_size=self.config.mlp_hidden_size,
            mlp_num_layers=self.config.mlp_num_layers,
            mlp_dropout=self.config.mlp_dropout,
            mlp_num_epochs=self.config.mlp_num_epochs,
            mlp_learning_rate=self.config.mlp_learning_rate,
            mlp_weight_decay=self.config.mlp_weight_decay,
            mlp_batch_size=self.config.mlp_batch_size,
            mlp_val_split=self.config.mlp_val_split,
            mlp_early_stopping_patience=self.config.mlp_early_stopping_patience,
            lstm_hidden_size=self.config.lstm_hidden_size,
            lstm_num_layers=self.config.lstm_num_layers,
            lstm_seq_length=self.config.lstm_seq_length,
            lstm_num_epochs=self.config.lstm_num_epochs,
            lstm_learning_rate=self.config.lstm_learning_rate,
            lstm_batch_size=self.config.lstm_batch_size,
            lstm_val_split=self.config.lstm_val_split,
            lstm_early_stopping_patience=self.config.lstm_early_stopping_patience,
            verbose=self.config.verbose,
        )

    @staticmethod
    def _reconstruct_states(level: TrajectoryAwareLatentForecaster, latent_states: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
        flat = latent_states.reshape(-1, latent_states.shape[-1])
        recon = level._reconstruct_latent_batch(flat, spatial_shape)
        return recon.reshape(latent_states.shape[0], latent_states.shape[1], *spatial_shape)

    def fit(self, train_states: np.ndarray) -> "TrajectoryAwareMultiResolutionForecaster":
        residual_train = np.asarray(train_states, dtype=np.float64).copy()
        spatial_shape = residual_train.shape[2:]

        self._levels = []
        self._train_residuals = []
        self._test_residuals = []

        for level_idx in range(len(self.config.level_ranks)):
            level = TrajectoryAwareLatentForecaster(config=self._make_level_config(level_idx))
            level.fit(residual_train)
            self._levels.append(level)
            self._train_residuals.append(residual_train.copy())

            train_recon = self._reconstruct_states(level, level._train_latent, spatial_shape)
            residual_train = residual_train - train_recon

        self._fitted = True
        return self

    def evaluate_one_step(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        target = states[:, 1:, :, :].reshape(-1, *states.shape[2:])
        pred = np.zeros_like(target)

        residual_test = states.copy()
        for level in self._levels:
            level_metrics = level.evaluate_one_step(residual_test)
            pred += level_metrics["pred_spatial"]

            test_latent = level._project_states_to_latent(residual_test)
            residual_test = residual_test - self._reconstruct_states(level, test_latent, states.shape[2:])

        spatial_metrics = _compute_regression_metrics(target, pred)
        return {
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(target.shape[0]),
            "target_spatial": target,
            "pred_spatial": pred,
        }

    def evaluate_rollout(self, test_states: np.ndarray) -> Dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluation")

        states = np.asarray(test_states, dtype=np.float64)
        target = states[:, 1:, :, :].reshape(-1, *states.shape[2:])
        pred = np.zeros_like(target)

        residual_test = states.copy()
        for level in self._levels:
            level_metrics = level.evaluate_rollout(residual_test)
            pred += level_metrics["pred_spatial"]

            test_latent = level._project_states_to_latent(residual_test)
            residual_test = residual_test - self._reconstruct_states(level, test_latent, states.shape[2:])

        spatial_metrics = _compute_regression_metrics(target, pred)
        return {
            "spatial_mse": spatial_metrics["mse"],
            "spatial_rmse": spatial_metrics["rmse"],
            "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
            "spatial_r2": spatial_metrics["r2"],
            "n_eval_samples": int(target.shape[0]),
            "n_rollout_steps": int(states.shape[1] - 1),
            "target_spatial": target,
            "pred_spatial": pred,
        }


__all__ = [
    "NavierStokesTrajectoryDataset",
    "TrajectoryAwareDMDForecaster",
    "TrajectoryAwareCSForecaster",
    "TrajectoryAwareEigenvalueProjectedDMDForecaster",
    "TrajectoryAwareLatentForecaster",
    "TrajectoryAwareResidualCorrectedForecaster",
    "TrajectoryAwareMultiResolutionForecaster",
    "TrajectoryAwarePersistenceForecaster",
    "TrajectoryAwareStableDMDForecaster",
    "_build_t_plus_one_correction_features",
    "build_lagged_windows",
    "build_one_step_pairs",
    "load_navier_stokes_trajectory_dataset",
    "reshape_flattened_train_transitions",
    "stitch_inputs_and_labels_to_states",
]
