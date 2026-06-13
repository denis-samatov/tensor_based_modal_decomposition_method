#!/usr/bin/env python3
"""Evaluate causal next-step forecasting with windowed TBMD + QR + CS."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.config import CompressiveSensingConfig, DecompositionConfig, SensorPlacementConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposerInterface
from TBMD.core.reconstruction.tensor_compressive_sensing import (
    ExtensionCompressiveSensingConfig,
    TensorCompressiveSensing,
)
from TBMD.core.sensor_placement import TensorTubeQRDecomposition
from TBMD.experiments import (
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics
from TBMD.experiments.navier_stokes_model_registry import DEFAULT_N_TRAIN_TRAJECTORIES

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "windowed_tbmd_qr_cs_forecasting_summary.json"
)
TUNING_DEV_SPLIT = 0.2


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_forecast_segment_tensor(
    states: np.ndarray,
    *,
    history_length: int,
    stride: int,
    max_segments: int | None,
) -> np.ndarray:
    """Build causal forecasting segments as `(history+1, H, W, N_segments)`."""
    series = np.asarray(states, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("states must have shape `(B, T, H, W)`")
    if history_length <= 0 or stride <= 0:
        raise ValueError("history_length and stride must be positive")
    segment_length = history_length + 1
    if series.shape[1] < segment_length:
        raise ValueError("history_length + 1 exceeds trajectory length")

    segment_refs = [
        (traj_idx, start)
        for traj_idx in range(series.shape[0])
        for start in range(0, series.shape[1] - segment_length + 1, stride)
    ]
    if max_segments is not None and len(segment_refs) > max_segments:
        selected = np.linspace(
            0,
            len(segment_refs) - 1,
            num=max_segments,
            dtype=int,
        )
        segment_refs = [segment_refs[idx] for idx in selected]

    segments = []
    for traj_idx, start in segment_refs:
        segments.append(series[traj_idx, start : start + segment_length])
    return np.stack(segments, axis=-1)


def _compute_segment_dictionary_from_tucker(
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


def _fit_segment_dictionary(
    train_segments: np.ndarray,
    *,
    ranks: list[int],
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    config = DecompositionConfig(
        ranks=ranks,
        epsilon=1e-5,
        random_state=random_state,
        device="cpu",
        dtype="float32",
        verbose=False,
        max_workers=1,
    )
    decomposer = TuckerDecomposerInterface(train_segments.astype(np.float32), config=config)
    decomposer.decompose()
    decomposer.reconstruct()
    dictionary = _compute_segment_dictionary_from_tucker(
        decomposer.core_tensor,
        decomposer.factors,
    )
    return dictionary.astype(np.float64), {
        "core_shape": list(_as_numpy(decomposer.core_tensor).shape),
        "factor_shapes": [list(_as_numpy(factor).shape) for factor in decomposer.factors],
        "segment_reconstruction_rel_frob": float(decomposer.reconstruction_errors),
    }


def _augment_spatial_mask_by_leverage(
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


def _place_fixed_spatial_sensors(
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
        spatial_mask = _augment_spatial_mask_by_leverage(
            qr_tensor,
            spatial_mask,
            target_sensors=n_spatial_sensors,
        )
    return spatial_mask, np.flatnonzero(spatial_mask.reshape(-1))


def _history_and_target(dictionary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if dictionary.shape[0] < 2:
        raise ValueError("dictionary must include history and target slices")
    return dictionary[:-1], dictionary[-1]


def _predict_next_full_history_lstsq(
    segments: np.ndarray,
    dictionary: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    history_dictionary, target_dictionary = _history_and_target(dictionary)
    history = segments[:-1]
    history_matrix = history_dictionary.reshape(-1, history_dictionary.shape[-1])
    measurements = history.reshape(-1, history.shape[-1]).T
    coeffs = measurements @ np.linalg.pinv(history_matrix, rcond=rcond).T
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs


def _predict_next_sensor_lstsq(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    history_dictionary, target_dictionary = _history_and_target(dictionary)
    history = segments[:-1]
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history_dictionary = history_dictionary.reshape(history_dictionary.shape[0], height_width, -1)
    sensor_dictionary = flat_history_dictionary[:, spatial_sensor_indices, :].reshape(
        -1,
        history_dictionary.shape[-1],
    )
    flat_history = history.reshape(history.shape[0], height_width, history.shape[-1])
    measurements = flat_history[:, spatial_sensor_indices, :].reshape(
        -1,
        history.shape[-1],
    ).T
    coeffs = measurements @ np.linalg.pinv(sensor_dictionary, rcond=rcond).T
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs


def _predict_from_history_sensor_lstsq(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict target frames from history shaped `(N,L,H,W)` using sensor least squares."""
    history_dictionary, target_dictionary = _history_and_target(dictionary)
    if history.shape[1:] != history_dictionary.shape[:-1]:
        raise ValueError("history shape must match dictionary history slices")
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history_dictionary = history_dictionary.reshape(history_dictionary.shape[0], height_width, -1)
    sensor_dictionary = flat_history_dictionary[:, spatial_sensor_indices, :].reshape(
        -1,
        history_dictionary.shape[-1],
    )
    flat_history = history.transpose(1, 2, 3, 0).reshape(
        history_dictionary.shape[0],
        height_width,
        history.shape[0],
    )
    measurements = flat_history[:, spatial_sensor_indices, :].reshape(
        -1,
        history.shape[0],
    ).T
    coeffs = measurements @ np.linalg.pinv(sensor_dictionary, rcond=rcond).T
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs


def _predict_next_sensor_cs(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_mask: np.ndarray,
    *,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool]]]:
    history_dictionary, target_dictionary = _history_and_target(dictionary)
    history_mask = np.broadcast_to(spatial_mask, history_dictionary.shape[:-1]).copy()
    core_cfg = CompressiveSensingConfig(
        max_iter=cs_max_iter,
        tol=cs_tol,
        epsilon_l1=cs_epsilon_l1,
        device="cpu",
        dtype=torch.float32,
    )
    ext_cfg = ExtensionCompressiveSensingConfig(solver="cholesky", collect_history=False)
    coeffs = np.zeros((segments.shape[-1], dictionary.shape[-1]), dtype=np.float64)
    metrics_out = []
    for idx in range(segments.shape[-1]):
        solver = TensorCompressiveSensing(
            history_dictionary.astype(np.float32),
            history_mask,
            segments[:-1, :, :, idx].astype(np.float32),
            core_cfg=core_cfg,
            ext_cfg=ext_cfg,
        )
        coeff, metrics = solver.solve()
        coeffs[idx] = coeff.numpy().astype(np.float64, copy=False)
        metrics_out.append(
            {
                "iterations": int(metrics.iterations),
                "converged": bool(metrics.converged),
                "primal_residual": float(metrics.primal_residual),
                "dual_residual": float(metrics.dual_residual),
                "objective": float(metrics.objective),
            }
        )
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs, metrics_out


def _predict_from_history_sensor_cs(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_mask: np.ndarray,
    *,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool]]]:
    """Predict target frames from history shaped `(N,L,H,W)` using QR/CS recovery."""
    history_dictionary, target_dictionary = _history_and_target(dictionary)
    if history.shape[1:] != history_dictionary.shape[:-1]:
        raise ValueError("history shape must match dictionary history slices")
    history_mask = np.broadcast_to(spatial_mask, history_dictionary.shape[:-1]).copy()
    core_cfg = CompressiveSensingConfig(
        max_iter=cs_max_iter,
        tol=cs_tol,
        epsilon_l1=cs_epsilon_l1,
        device="cpu",
        dtype=torch.float32,
    )
    ext_cfg = ExtensionCompressiveSensingConfig(solver="cholesky", collect_history=False)
    coeffs = np.zeros((history.shape[0], dictionary.shape[-1]), dtype=np.float64)
    metrics_out = []
    for idx in range(history.shape[0]):
        solver = TensorCompressiveSensing(
            history_dictionary.astype(np.float32),
            history_mask,
            history[idx].astype(np.float32),
            core_cfg=core_cfg,
            ext_cfg=ext_cfg,
        )
        coeff, metrics = solver.solve()
        coeffs[idx] = coeff.numpy().astype(np.float64, copy=False)
        metrics_out.append(
            {
                "iterations": int(metrics.iterations),
                "converged": bool(metrics.converged),
                "primal_residual": float(metrics.primal_residual),
                "dual_residual": float(metrics.dual_residual),
                "objective": float(metrics.objective),
            }
        )
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs, metrics_out


def _target_frames_from_segments(segments: np.ndarray) -> np.ndarray:
    return segments[-1].transpose(2, 0, 1)


def _fit_ridge_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float,
) -> dict[str, object]:
    """Fit a linear ridge residual map from recovered coefficients to target-frame error."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if target_flat.shape != base_flat.shape:
        raise ValueError("target_frames and base_predictions must have matching sample/output shapes")
    if coeffs.shape[0] != target_flat.shape[0]:
        raise ValueError("coeffs and predictions must contain the same number of samples")

    features = np.concatenate(
        [coeffs, np.ones((coeffs.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    residual = target_flat - base_flat
    gram = features.T @ features
    penalty = alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = features.T @ residual
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    return {
        "alpha": float(alpha),
        "weights": weights,
        "feature_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
    }


def _apply_ridge_residual_corrector(
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    corrector: dict[str, object],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    features = np.concatenate(
        [coeffs, np.ones((coeffs.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    correction = features @ np.asarray(corrector["weights"], dtype=np.float64)
    return (base_flat + scale * correction).reshape(base_predictions.shape)


def _source_metrics(
    segments: np.ndarray,
    predictions: np.ndarray,
    coeffs: np.ndarray,
    reference_coeffs: np.ndarray,
) -> dict[str, object]:
    target = _target_frames_from_segments(segments)
    spatial = _compute_regression_metrics(target, predictions)
    coeff = _compute_regression_metrics(reference_coeffs, coeffs)
    return {
        "spatial_rmse": spatial["rmse"],
        "spatial_rel_frob_err": spatial["rel_frob_err"],
        "spatial_r2": spatial["r2"],
        "coeff_rmse_vs_full_history": coeff["rmse"],
        "coeff_rel_frob_vs_full_history": coeff["rel_frob_err"],
        "coeff_r2_vs_full_history": coeff["r2"],
        "n_eval_samples": int(target.shape[0]),
    }


def _evaluate_segments(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_mask: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    sensor_rcond: float,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
    ridge_correctors: dict[str, dict[str, object]] | None = None,
    ridge_correction_scale: float = 1.0,
) -> dict[str, object]:
    full_pred, full_coeffs = _predict_next_full_history_lstsq(
        segments,
        dictionary,
        rcond=sensor_rcond,
    )
    sensor_ls_pred, sensor_ls_coeffs = _predict_next_sensor_lstsq(
        segments,
        dictionary,
        spatial_sensor_indices,
        rcond=sensor_rcond,
    )
    sensor_cs_pred, sensor_cs_coeffs, cs_metrics = _predict_next_sensor_cs(
        segments,
        dictionary,
        spatial_mask,
        cs_max_iter=cs_max_iter,
        cs_tol=cs_tol,
        cs_epsilon_l1=cs_epsilon_l1,
    )
    result = {
        "full_history_lstsq": _source_metrics(
            segments,
            full_pred,
            full_coeffs,
            full_coeffs,
        ),
        "fixed_sensor_lstsq": _source_metrics(
            segments,
            sensor_ls_pred,
            sensor_ls_coeffs,
            full_coeffs,
        ),
        "fixed_sensor_cs": _source_metrics(
            segments,
            sensor_cs_pred,
            sensor_cs_coeffs,
            full_coeffs,
        ),
    }
    result["fixed_sensor_cs"]["cs_mean_iterations"] = float(
        np.mean([m["iterations"] for m in cs_metrics])
    )
    result["fixed_sensor_cs"]["cs_convergence_rate"] = float(
        np.mean([m["converged"] for m in cs_metrics])
    )
    result["fixed_sensor_cs"]["cs_mean_objective"] = float(
        np.mean([m["objective"] for m in cs_metrics])
    )
    if ridge_correctors:
        result["ridge_corrected"] = {}
        target = _target_frames_from_segments(segments)
        for label, corrector in ridge_correctors.items():
            corrected_ls = _apply_ridge_residual_corrector(
                sensor_ls_pred,
                sensor_ls_coeffs,
                corrector,
                scale=ridge_correction_scale,
            )
            corrected_cs = _apply_ridge_residual_corrector(
                sensor_cs_pred,
                sensor_cs_coeffs,
                corrector,
                scale=ridge_correction_scale,
            )
            result["ridge_corrected"][label] = {
                "fixed_sensor_lstsq": _compute_regression_metrics(target, corrected_ls),
                "fixed_sensor_cs": _compute_regression_metrics(target, corrected_cs),
            }
    return result


def _evaluate_recursive_rollout(
    trajectories: np.ndarray,
    dictionary: np.ndarray,
    *,
    spatial_mask: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    sensor_rcond: float,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
    ridge_corrector: dict[str, object] | None,
    recovery_source: str,
    ridge_correction_scale: float = 1.0,
    rollout_update_blend: float = 1.0,
) -> dict[str, object]:
    """Evaluate strict autoregressive rollout after a true-history warmup."""
    series = np.asarray(trajectories, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("trajectories must have shape `(B,T,H,W)`")
    history_length = dictionary.shape[0] - 1
    if series.shape[1] <= history_length:
        raise ValueError("trajectory length must exceed dictionary history length")
    if series.shape[2:] != dictionary.shape[1:3]:
        raise ValueError("trajectory spatial shape must match dictionary")
    if recovery_source not in {"sensor_lstsq", "sensor_cs"}:
        raise ValueError("recovery_source must be `sensor_lstsq` or `sensor_cs`")

    history = series[:, :history_length].copy()
    predictions = []
    targets = []
    coeffs_per_step = []
    cs_metrics = []
    for step_idx in range(history_length, series.shape[1]):
        if recovery_source == "sensor_lstsq":
            pred, coeffs = _predict_from_history_sensor_lstsq(
                history,
                dictionary,
                spatial_sensor_indices,
                rcond=sensor_rcond,
            )
            step_metrics = []
        else:
            pred, coeffs, step_metrics = _predict_from_history_sensor_cs(
                history,
                dictionary,
                spatial_mask,
                cs_max_iter=cs_max_iter,
                cs_tol=cs_tol,
                cs_epsilon_l1=cs_epsilon_l1,
            )
        if ridge_corrector is not None:
            pred = _apply_ridge_residual_corrector(
                pred,
                coeffs,
                ridge_corrector,
                scale=ridge_correction_scale,
            )
        pred = rollout_update_blend * pred + (1.0 - rollout_update_blend) * history[:, -1]
        predictions.append(pred)
        targets.append(series[:, step_idx])
        coeffs_per_step.append(coeffs)
        cs_metrics.extend(step_metrics)
        history = np.concatenate([history[:, 1:], pred[:, None]], axis=1)

    pred_spatial = np.concatenate(predictions, axis=0)
    target_spatial = np.concatenate(targets, axis=0)
    metrics = _compute_regression_metrics(target_spatial, pred_spatial)
    payload = {
        "spatial_rmse": metrics["rmse"],
        "spatial_rel_frob_err": metrics["rel_frob_err"],
        "spatial_r2": metrics["r2"],
        "n_eval_samples": int(target_spatial.shape[0]),
        "n_trajectories": int(series.shape[0]),
        "n_rollout_steps": int(series.shape[1] - history_length),
        "recovery_source": recovery_source,
        "target_spatial": target_spatial,
        "pred_spatial": pred_spatial,
        "coefficients": np.concatenate(coeffs_per_step, axis=0),
    }
    if cs_metrics:
        payload["cs_mean_iterations"] = float(
            np.mean([m["iterations"] for m in cs_metrics])
        )
        payload["cs_convergence_rate"] = float(
            np.mean([m["converged"] for m in cs_metrics])
        )
        payload["cs_mean_objective"] = float(
            np.mean([m["objective"] for m in cs_metrics])
        )
    return payload


def _collect_closed_loop_residual_pairs(
    trajectories: np.ndarray,
    dictionary: np.ndarray,
    *,
    spatial_mask: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    sensor_rcond: float,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
    history_corrector: dict[str, object] | None,
    history_correction_scale: float,
    recovery_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect residual-training pairs from autoregressive predicted histories."""
    series = np.asarray(trajectories, dtype=np.float64)
    history_length = dictionary.shape[0] - 1
    if series.ndim != 4:
        raise ValueError("trajectories must have shape `(B,T,H,W)`")
    if series.shape[1] <= history_length:
        raise ValueError("trajectory length must exceed dictionary history length")
    if recovery_source not in {"sensor_lstsq", "sensor_cs"}:
        raise ValueError("recovery_source must be `sensor_lstsq` or `sensor_cs`")

    history = series[:, :history_length].copy()
    targets = []
    base_predictions = []
    coeffs_out = []
    for step_idx in range(history_length, series.shape[1]):
        if recovery_source == "sensor_lstsq":
            base_pred, coeffs = _predict_from_history_sensor_lstsq(
                history,
                dictionary,
                spatial_sensor_indices,
                rcond=sensor_rcond,
            )
        else:
            base_pred, coeffs, _ = _predict_from_history_sensor_cs(
                history,
                dictionary,
                spatial_mask,
                cs_max_iter=cs_max_iter,
                cs_tol=cs_tol,
                cs_epsilon_l1=cs_epsilon_l1,
            )
        if history_corrector is None:
            history_update = base_pred
        else:
            history_update = _apply_ridge_residual_corrector(
                base_pred,
                coeffs,
                history_corrector,
                scale=history_correction_scale,
            )
        targets.append(series[:, step_idx])
        base_predictions.append(base_pred)
        coeffs_out.append(coeffs)
        history = np.concatenate([history[:, 1:], history_update[:, None]], axis=1)

    return (
        np.concatenate(targets, axis=0),
        np.concatenate(base_predictions, axis=0),
        np.concatenate(coeffs_out, axis=0),
    )


def _compact_rollout_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"target_spatial", "pred_spatial", "coefficients"}
    }


def _load_data(n_train_trajectories: int, n_test_trajectories: int | None):
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_states = dataset.train_states[:n_train_trajectories]
    test_states = dataset.test_states
    if n_test_trajectories is not None:
        test_states = test_states[:n_test_trajectories]
    return train_states, test_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train-trajectories", type=int, default=160)
    parser.add_argument("--n-test-trajectories", type=int, default=40)
    parser.add_argument("--history-length", type=int, default=7)
    parser.add_argument("--segment-stride", type=int, default=1)
    parser.add_argument("--max-train-segments", type=int, default=512)
    parser.add_argument("--max-dev-segments", type=int, default=None)
    parser.add_argument("--max-test-segments", type=int, default=None)
    parser.add_argument("--r-tau", type=int, default=8)
    parser.add_argument("--r-x", type=int, default=32)
    parser.add_argument("--r-y", type=int, default=32)
    parser.add_argument("--r-segment", type=int, default=45)
    parser.add_argument("--n-spatial-sensors", type=int, default=45)
    parser.add_argument("--cs-max-iter", type=int, default=100)
    parser.add_argument("--cs-tol", type=float, default=1e-4)
    parser.add_argument("--cs-epsilon-l1", type=float, default=1e-3)
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument(
        "--ridge-correction-alphas",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional ridge residual correction alphas. Each head is fit on "
            "training sensor least-squares coefficients and selected by dev "
            "fixed-sensor CS R2."
        ),
    )
    parser.add_argument(
        "--ridge-correction-scales",
        type=float,
        nargs="*",
        default=[1.0],
        help=(
            "Candidate multipliers for the ridge residual. Used for rollout-aware "
            "selection when --select-ridge-by-rollout is enabled."
        ),
    )
    parser.add_argument(
        "--select-ridge-by-rollout",
        action="store_true",
        help=(
            "Select ridge alpha/scale by strict dev rollout R2 instead of "
            "teacher-forced dev one-step R2."
        ),
    )
    parser.add_argument(
        "--closed-loop-correction",
        action="store_true",
        help=(
            "Refit the ridge residual head on rollout-generated train histories "
            "before final dev/test evaluation."
        ),
    )
    parser.add_argument(
        "--closed-loop-ridge-alphas",
        type=float,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--closed-loop-recovery-source",
        choices=("sensor_lstsq", "sensor_cs"),
        default="sensor_lstsq",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument(
        "--evaluate-rollout",
        action="store_true",
        help=(
            "Also run strict autoregressive rollout after the true-history warmup. "
            "After warmup, predicted frames are fed back into the history."
        ),
    )
    parser.add_argument(
        "--rollout-recovery-source",
        choices=("sensor_cs", "sensor_lstsq"),
        default="sensor_cs",
    )
    parser.add_argument(
        "--rollout-update-blends",
        type=float,
        nargs="*",
        default=[1.0],
        help=(
            "Candidate blends for recursive rollout updates. 1.0 uses the model "
            "prediction, 0.0 holds the last history frame."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train_trajectories > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(
            f"n_train_trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}"
        )

    all_train_states, official_test_states = _load_data(
        args.n_train_trajectories,
        args.n_test_trajectories,
    )
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=TUNING_DEV_SPLIT,
    )
    spatial_mean = np.mean(tuning_train_states, axis=(0, 1))
    centered_train = tuning_train_states - spatial_mean
    centered_dev = tuning_dev_states - spatial_mean
    centered_test = official_test_states - spatial_mean

    start = time.time()
    train_segments = _build_forecast_segment_tensor(
        centered_train,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_train_segments,
    )
    ranks = [args.r_tau, args.r_x, args.r_y, args.r_segment]
    dictionary, tbmd_summary = _fit_segment_dictionary(
        train_segments,
        ranks=ranks,
        random_state=args.random_state,
    )
    history_dictionary, _ = _history_and_target(dictionary)
    spatial_mask, spatial_sensor_indices = _place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=args.n_spatial_sensors,
        random_state=args.random_state,
    )
    ridge_correctors = {}
    ridge_train_metrics = {}
    if args.ridge_correction_alphas:
        train_sensor_pred, train_sensor_coeffs = _predict_next_sensor_lstsq(
            train_segments,
            dictionary,
            spatial_sensor_indices,
            rcond=args.sensor_rcond,
        )
        train_targets = _target_frames_from_segments(train_segments)
        for alpha in args.ridge_correction_alphas:
            label = f"alpha_{alpha:g}"
            corrector = _fit_ridge_residual_corrector(
                train_targets,
                train_sensor_pred,
                train_sensor_coeffs,
                alpha=alpha,
            )
            corrected_train = _apply_ridge_residual_corrector(
                train_sensor_pred,
                train_sensor_coeffs,
                corrector,
            )
            ridge_correctors[label] = corrector
            ridge_train_metrics[label] = {
                **_compute_regression_metrics(train_targets, corrected_train),
                "alpha": float(alpha),
                "feature_dim": corrector["feature_dim"],
                "output_dim": corrector["output_dim"],
            }
    fit_time = time.time() - start

    dev_segments = _build_forecast_segment_tensor(
        centered_dev,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_dev_segments,
    )
    dev_result = _evaluate_segments(
        dev_segments,
        dictionary,
        spatial_mask,
        spatial_sensor_indices,
        sensor_rcond=args.sensor_rcond,
        cs_max_iter=args.cs_max_iter,
        cs_tol=args.cs_tol,
        cs_epsilon_l1=args.cs_epsilon_l1,
        ridge_correctors=ridge_correctors or None,
    )
    selected_ridge_label = None
    selected_ridge_scale = 1.0
    selected_rollout_update_blend = float(args.rollout_update_blends[0])
    selected_ridge_summary = None
    if ridge_correctors:
        rollout_selection_metrics = None
        if args.select_ridge_by_rollout:
            rollout_selection_metrics = {}
            best_rollout_r2 = -np.inf
            for label, corrector in ridge_correctors.items():
                for scale in args.ridge_correction_scales:
                    for blend in args.rollout_update_blends:
                        rollout_metrics = _compact_rollout_result(
                            _evaluate_recursive_rollout(
                                centered_dev,
                                dictionary,
                                spatial_mask=spatial_mask,
                                spatial_sensor_indices=spatial_sensor_indices,
                                sensor_rcond=args.sensor_rcond,
                                cs_max_iter=args.cs_max_iter,
                                cs_tol=args.cs_tol,
                                cs_epsilon_l1=args.cs_epsilon_l1,
                                ridge_corrector=corrector,
                                recovery_source=args.rollout_recovery_source,
                                ridge_correction_scale=scale,
                                rollout_update_blend=blend,
                            )
                        )
                        candidate_key = f"{label}_scale_{scale:g}_blend_{blend:g}"
                        rollout_selection_metrics[candidate_key] = {
                            **rollout_metrics,
                            "alpha_label": label,
                            "scale": float(scale),
                            "rollout_update_blend": float(blend),
                        }
                        if rollout_metrics["spatial_r2"] > best_rollout_r2:
                            best_rollout_r2 = rollout_metrics["spatial_r2"]
                            selected_ridge_label = label
                            selected_ridge_scale = float(scale)
                            selected_rollout_update_blend = float(blend)
        else:
            selected_ridge_label = max(
                dev_result["ridge_corrected"],
                key=lambda label: dev_result["ridge_corrected"][label]["fixed_sensor_cs"]["r2"],
            )
            selected_ridge_scale = float(args.ridge_correction_scales[0])
            selected_rollout_update_blend = float(args.rollout_update_blends[0])
        selected_ridge_summary = {
            "selected_label": selected_ridge_label,
            "selected_alpha": ridge_correctors[selected_ridge_label]["alpha"],
            "selected_scale": selected_ridge_scale,
            "selected_rollout_update_blend": selected_rollout_update_blend,
            "selection_metric": "dev_result.ridge_corrected.<alpha>.fixed_sensor_cs.r2",
            "train_metrics": ridge_train_metrics,
            "dev_metrics": dev_result["ridge_corrected"],
        }
        if rollout_selection_metrics is not None:
            selected_ridge_summary["selection_metric"] = (
                "dev strict autoregressive rollout spatial_r2"
            )
            selected_ridge_summary["dev_rollout_selection_metrics"] = (
                rollout_selection_metrics
            )

    if args.closed_loop_correction:
        history_corrector = (
            ridge_correctors[selected_ridge_label]
            if selected_ridge_label is not None
            else None
        )
        closed_targets, closed_base_pred, closed_coeffs = (
            _collect_closed_loop_residual_pairs(
                centered_train,
                dictionary,
                spatial_mask=spatial_mask,
                spatial_sensor_indices=spatial_sensor_indices,
                sensor_rcond=args.sensor_rcond,
                cs_max_iter=args.cs_max_iter,
                cs_tol=args.cs_tol,
                cs_epsilon_l1=args.cs_epsilon_l1,
                history_corrector=history_corrector,
                history_correction_scale=selected_ridge_scale,
                recovery_source=args.closed_loop_recovery_source,
            )
        )
        closed_loop_alphas = (
            args.closed_loop_ridge_alphas
            or args.ridge_correction_alphas
            or [1e-6]
        )
        closed_loop_correctors = {}
        closed_loop_train_metrics = {}
        for alpha in closed_loop_alphas:
            label = f"closed_loop_alpha_{alpha:g}"
            corrector = _fit_ridge_residual_corrector(
                closed_targets,
                closed_base_pred,
                closed_coeffs,
                alpha=alpha,
            )
            corrected_train = _apply_ridge_residual_corrector(
                closed_base_pred,
                closed_coeffs,
                corrector,
            )
            closed_loop_correctors[label] = corrector
            closed_loop_train_metrics[label] = {
                **_compute_regression_metrics(closed_targets, corrected_train),
                "alpha": float(alpha),
                "feature_dim": corrector["feature_dim"],
                "output_dim": corrector["output_dim"],
            }

        rollout_selection_metrics = {}
        best_rollout_r2 = -np.inf
        selected_ridge_label = None
        selected_ridge_scale = 1.0
        selected_rollout_update_blend = float(args.rollout_update_blends[0])
        for label, corrector in closed_loop_correctors.items():
            for scale in args.ridge_correction_scales:
                for blend in args.rollout_update_blends:
                    rollout_metrics = _compact_rollout_result(
                        _evaluate_recursive_rollout(
                            centered_dev,
                            dictionary,
                            spatial_mask=spatial_mask,
                            spatial_sensor_indices=spatial_sensor_indices,
                            sensor_rcond=args.sensor_rcond,
                            cs_max_iter=args.cs_max_iter,
                            cs_tol=args.cs_tol,
                            cs_epsilon_l1=args.cs_epsilon_l1,
                            ridge_corrector=corrector,
                            recovery_source=args.rollout_recovery_source,
                            ridge_correction_scale=scale,
                            rollout_update_blend=blend,
                        )
                    )
                    candidate_key = f"{label}_scale_{scale:g}_blend_{blend:g}"
                    rollout_selection_metrics[candidate_key] = {
                        **rollout_metrics,
                        "alpha_label": label,
                        "scale": float(scale),
                        "rollout_update_blend": float(blend),
                    }
                    if rollout_metrics["spatial_r2"] > best_rollout_r2:
                        best_rollout_r2 = rollout_metrics["spatial_r2"]
                        selected_ridge_label = label
                        selected_ridge_scale = float(scale)
                        selected_rollout_update_blend = float(blend)
        ridge_correctors = closed_loop_correctors
        dev_result = _evaluate_segments(
            dev_segments,
            dictionary,
            spatial_mask,
            spatial_sensor_indices,
            sensor_rcond=args.sensor_rcond,
            cs_max_iter=args.cs_max_iter,
            cs_tol=args.cs_tol,
            cs_epsilon_l1=args.cs_epsilon_l1,
            ridge_correctors=ridge_correctors,
            ridge_correction_scale=selected_ridge_scale,
        )
        selected_ridge_summary = {
            "selected_label": selected_ridge_label,
            "selected_alpha": ridge_correctors[selected_ridge_label]["alpha"],
            "selected_scale": selected_ridge_scale,
            "selected_rollout_update_blend": selected_rollout_update_blend,
            "selection_metric": "closed-loop dev strict autoregressive rollout spatial_r2",
            "closed_loop": True,
            "closed_loop_recovery_source": args.closed_loop_recovery_source,
            "train_pair_count": int(closed_targets.shape[0]),
            "train_metrics": closed_loop_train_metrics,
            "dev_metrics": dev_result["ridge_corrected"],
            "dev_rollout_selection_metrics": rollout_selection_metrics,
        }

    final_test_result = None
    if not args.skip_official_test:
        test_segments = _build_forecast_segment_tensor(
            centered_test,
            history_length=args.history_length,
            stride=args.segment_stride,
            max_segments=args.max_test_segments,
        )
        final_test_result = _evaluate_segments(
            test_segments,
            dictionary,
            spatial_mask,
            spatial_sensor_indices,
            sensor_rcond=args.sensor_rcond,
            cs_max_iter=args.cs_max_iter,
            cs_tol=args.cs_tol,
            cs_epsilon_l1=args.cs_epsilon_l1,
            ridge_correctors=(
                {selected_ridge_label: ridge_correctors[selected_ridge_label]}
                if selected_ridge_label is not None
                else None
            ),
            ridge_correction_scale=selected_ridge_scale,
        )

    rollout_result = None
    if args.evaluate_rollout:
        rollout_corrector = (
            ridge_correctors[selected_ridge_label]
            if selected_ridge_label is not None
            else None
        )
        rollout_result = {
            "protocol": (
                "Strict autoregressive rollout: first history_length frames are true, "
                "then each predicted frame is fed back into the next history window. "
                "No future true frames or future sensor measurements are used."
            ),
            "recovery_source": args.rollout_recovery_source,
            "ridge_corrector_label": selected_ridge_label,
            "dev": _compact_rollout_result(
                _evaluate_recursive_rollout(
                    centered_dev,
                    dictionary,
                    spatial_mask=spatial_mask,
                    spatial_sensor_indices=spatial_sensor_indices,
                    sensor_rcond=args.sensor_rcond,
                    cs_max_iter=args.cs_max_iter,
                    cs_tol=args.cs_tol,
                    cs_epsilon_l1=args.cs_epsilon_l1,
                    ridge_corrector=rollout_corrector,
                    recovery_source=args.rollout_recovery_source,
                    ridge_correction_scale=selected_ridge_scale,
                    rollout_update_blend=selected_rollout_update_blend,
                )
            ),
        }
        if not args.skip_official_test:
            rollout_result["final_test"] = _compact_rollout_result(
                _evaluate_recursive_rollout(
                    centered_test,
                    dictionary,
                    spatial_mask=spatial_mask,
                    spatial_sensor_indices=spatial_sensor_indices,
                    sensor_rcond=args.sensor_rcond,
                    cs_max_iter=args.cs_max_iter,
                    cs_tol=args.cs_tol,
                    cs_epsilon_l1=args.cs_epsilon_l1,
                    ridge_corrector=rollout_corrector,
                    recovery_source=args.rollout_recovery_source,
                    ridge_correction_scale=selected_ridge_scale,
                    rollout_update_blend=selected_rollout_update_blend,
                )
            )

    config_payload = vars(args).copy()
    config_payload["output"] = str(config_payload["output"])
    payload = {
        "protocol": (
            "Causal next-step forecasting with windowed TBMD dictionary. "
            "The Tucker/HOSVD dictionary is fit on train segments containing "
            "history and next target; dev/test coefficients are recovered only "
            "from history measurements, then the target slice is reconstructed."
        ),
        "config": config_payload,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "train_segment_shape": list(train_segments.shape),
        "dictionary_shape": list(dictionary.shape),
        "tbmd_summary": tbmd_summary,
        "sensor_summary": {
            "mode": "fixed_spatial_repeated_over_history",
            "requested_spatial_sensors": args.n_spatial_sensors,
            "actual_spatial_sensors": int(spatial_mask.sum()),
            "total_history_measurements_per_prediction": int(
                spatial_mask.sum() * args.history_length
            ),
            "sensor_indices": spatial_sensor_indices.astype(int).tolist(),
        },
        "fit_time": fit_time,
        "ridge_correction": selected_ridge_summary,
        "dev_result": dev_result,
        "final_test_result": final_test_result,
        "rollout_result": rollout_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved windowed TBMD QR/CS forecasting summary to {args.output}")


if __name__ == "__main__":
    main()
