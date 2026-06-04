#!/usr/bin/env python3
"""Cached multi-dev residual sweep for Fast TBMD+QR+CS t+1 forecasting.

This script keeps the expensive TBMD dictionary, QR sensors, and sensor decoder
fixed per train-only dev split. It then sweeps residual correction heads from
cached base predictions and coefficient features. The official test split is
not loaded or evaluated.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import load_navier_stokes_trajectory_dataset
from TBMD.experiments.navier_stokes_fast_tplus1 import (
    apply_coefficient_calibrator,
    apply_ridge_residual_corrector,
    attach_coefficient_gate,
    build_forecast_segment_tensor,
    build_correction_feature_matrix,
    fit_coefficient_calibrator,
    fit_sensor_innovation_encoder,
    fit_patch_residual_svd_corrector,
    fit_ridge_residual_corrector,
    fit_segment_dictionary,
    fit_sensor_coefficient_decoder,
    history_and_target,
    history_sensor_matrix,
    place_fixed_spatial_sensors,
    predict_next_sensor_decoder_with_measurements,
    reconstruct_target_from_coefficients,
    target_frames_from_segments,
)
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics
from TBMD.experiments.navier_stokes_model_registry import DEFAULT_N_TRAIN_TRAJECTORIES


DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "stage5_fast_tplus1_cached_residual.json"
)


@dataclass(frozen=True)
class CachedBaseConfig:
    history_length: int
    ranks: tuple[int, int, int, int]
    n_spatial_sensors: int
    max_train_segments: int | None
    sensor_decoder: str = "ridge"
    decoder_ridge_lambda: float = 1e-8
    sensor_rcond: float = 1e-6
    random_state: int = 0
    dtype: str = "float32"


@dataclass(frozen=True)
class ResidualCandidate:
    name: str
    residual_rank: int | None
    scale: float
    alpha: float = 1e-8
    residual_weighting: str = "uniform"
    residual_weight_floor: float = 0.1
    head_type: str = "global_residual_svd"
    patch_size: int | None = None
    patch_residual_rank: int | None = None
    gate_type: str = "none"
    gate_threshold: float = 1.25
    gate_strength: float = 1.0
    gate_min: float = 0.5
    innovation_rank: int = 0
    innovation_include_norms: bool = False
    coefficient_calibration_type: str = "none"
    coefficient_calibration_alpha: float = 1e-6
    coefficient_calibration_blend: float = 1.0


def build_base_config(mode: str, args: argparse.Namespace) -> CachedBaseConfig:
    if mode == "smoke":
        ranks = (6, 16, 16, 48)
        sensors = 96
        max_segments = 96
    else:
        ranks = (8, 32, 32, 300)
        sensors = 1000
        max_segments = 2048
    return CachedBaseConfig(
        history_length=args.history_length,
        ranks=tuple(args.ranks) if args.ranks else ranks,
        n_spatial_sensors=args.n_spatial_sensors or sensors,
        max_train_segments=args.max_train_segments
        if args.max_train_segments is not None
        else max_segments,
        sensor_decoder=args.sensor_decoder,
        decoder_ridge_lambda=args.decoder_ridge_lambda,
        sensor_rcond=args.sensor_rcond,
        random_state=args.random_state,
    )


def build_residual_candidates(mode: str) -> list[ResidualCandidate]:
    if mode == "smoke":
        return [
            ResidualCandidate("smoke_svd16_scale1.0", residual_rank=16, scale=1.0),
            ResidualCandidate("smoke_svd32_scale1.1", residual_rank=32, scale=1.1),
            ResidualCandidate(
                "smoke_energy_svd32_scale1.1",
                residual_rank=32,
                scale=1.1,
                residual_weighting="residual_energy",
            ),
            ResidualCandidate(
                "smoke_patch16_rank4_scale1.0",
                residual_rank=None,
                scale=1.0,
                head_type="patch_residual_svd",
                patch_size=16,
                patch_residual_rank=4,
            ),
            ResidualCandidate(
                "smoke_innovation_svd16_rank4_scale1.0",
                residual_rank=16,
                scale=1.0,
                innovation_rank=4,
                innovation_include_norms=True,
            ),
            ResidualCandidate(
                "smoke_coeffcal_svd16_blend1.0",
                residual_rank=16,
                scale=1.0,
                coefficient_calibration_type="ridge",
                coefficient_calibration_blend=1.0,
            ),
        ]
    ranks = [128, 192, 256, 320]
    scales = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    candidates: list[ResidualCandidate] = []
    for rank in ranks:
        for scale in scales:
            candidates.append(
                ResidualCandidate(
                    f"uniform_svd{rank}_scale{scale:g}",
                    residual_rank=rank,
                    scale=scale,
                    residual_weighting="uniform",
                )
            )
            if rank in {192, 256, 320} and scale in {1.0, 1.1, 1.2}:
                candidates.append(
                    ResidualCandidate(
                        f"energy_svd{rank}_scale{scale:g}",
                        residual_rank=rank,
                        scale=scale,
                        residual_weighting="residual_energy",
                    )
                )
    for rank in [192, 256]:
        candidates.append(
            ResidualCandidate(
                f"uniform_svd{rank}_scale1.1_alpha1e-6",
                residual_rank=rank,
                scale=1.1,
                alpha=1e-6,
            )
        )
    for scale in [1.3, 1.5]:
        for threshold in [0.85, 1.0, 1.15]:
            for strength in [0.5, 1.0, 2.0]:
                candidates.append(
                    ResidualCandidate(
                        f"gated_svd256_scale{scale:g}_thr{threshold:g}_str{strength:g}",
                        residual_rank=256,
                        scale=scale,
                        gate_type="coefficient_rms",
                        gate_threshold=threshold,
                        gate_strength=strength,
                        gate_min=0.5,
                    )
                )
    for innovation_rank in [8, 16, 32, 64]:
        for scale in [1.0, 1.1, 1.2, 1.3]:
            candidates.append(
                ResidualCandidate(
                    f"innovation_svd256_ir{innovation_rank}_scale{scale:g}",
                    residual_rank=256,
                    scale=scale,
                    innovation_rank=innovation_rank,
                    innovation_include_norms=True,
                )
            )
    for blend in [0.25, 0.5, 0.75, 1.0]:
        for scale in [1.0, 1.1, 1.2]:
            candidates.append(
                ResidualCandidate(
                    f"coeffcal_svd256_blend{blend:g}_scale{scale:g}",
                    residual_rank=256,
                    scale=scale,
                    coefficient_calibration_type="ridge",
                    coefficient_calibration_blend=blend,
                )
            )
    for patch_rank in [24, 32]:
        for blend in [0.5, 1.0]:
            for scale in [1.0, 1.1, 1.2]:
                candidates.append(
                    ResidualCandidate(
                        f"coeffcal_patch16_rank{patch_rank}_blend{blend:g}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        coefficient_calibration_type="ridge",
                        coefficient_calibration_blend=blend,
                    )
                )
    for patch_rank in [24, 32]:
        for innovation_rank in [8, 16, 32]:
            for scale in [1.1, 1.2, 1.3]:
                candidates.append(
                    ResidualCandidate(
                        f"patch16_rank{patch_rank}_ir{innovation_rank}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        innovation_rank=innovation_rank,
                        innovation_include_norms=True,
                    )
                )
    for patch_size, patch_ranks, scales_for_patch in [
        (16, [8, 12, 16, 24, 32, 48], [1.0, 1.1, 1.2, 1.3]),
        (8, [4, 8, 12], [1.0, 1.1, 1.2]),
    ]:
        for patch_rank in patch_ranks:
            for scale in scales_for_patch:
                candidates.append(
                    ResidualCandidate(
                        f"patch{patch_size}_rank{patch_rank}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=patch_size,
                        patch_residual_rank=patch_rank,
                    )
                )
    return candidates


def build_dev_blocks(
    n_trajectories: int,
    *,
    dev_count: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_trajectories < 2:
        raise ValueError("Need at least two trajectories")
    if dev_count <= 0 or n_splits <= 0:
        raise ValueError("dev_count and n_splits must be positive")
    if dev_count >= n_trajectories:
        raise ValueError("dev_count leaves no training trajectories")
    max_start = n_trajectories - dev_count
    starts = np.linspace(0, max_start, num=n_splits, dtype=int)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(n_trajectories)
    for split_idx, start in enumerate(starts):
        dev_idx = np.arange(start, start + dev_count)
        train_idx = np.setdiff1d(all_indices, dev_idx, assume_unique=True)
        if train_idx.size == 0:
            raise ValueError(f"Split {split_idx} has no train trajectories")
        blocks.append((train_idx, dev_idx))
    return blocks


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _metrics_with_mae(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics = _compute_regression_metrics(target, pred)
    metrics["mae"] = float(np.mean(np.abs(np.asarray(target) - np.asarray(pred))))
    return metrics


def _sensing_diagnostics(dictionary: np.ndarray, sensor_indices: np.ndarray) -> dict[str, float]:
    sensing_matrix = history_sensor_matrix(dictionary, sensor_indices)
    singular_values = np.linalg.svd(sensing_matrix, compute_uv=False)
    positive = singular_values[singular_values > 1e-12]
    condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
    return {
        "sensing_rows": int(sensing_matrix.shape[0]),
        "sensing_cols": int(sensing_matrix.shape[1]),
        "sensing_rank": int(np.linalg.matrix_rank(sensing_matrix, tol=1e-10)),
        "sensing_condition_proxy": condition,
        "sensing_min_singular_value": float(positive[-1]) if positive.size else 0.0,
        "sensing_max_singular_value": float(positive[0]) if positive.size else 0.0,
    }


def build_residual_basis_caches(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    candidates: list[ResidualCandidate],
    *,
    random_state: int,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Precompute residual SVD bases once per weighting/floor pair."""
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    residual = target_flat - base_flat
    grouped: dict[tuple[str, float], int] = {}
    for candidate in candidates:
        if candidate.head_type != "global_residual_svd":
            continue
        if candidate.residual_rank is None:
            continue
        key = (candidate.residual_weighting, float(candidate.residual_weight_floor))
        grouped[key] = max(grouped.get(key, 0), int(candidate.residual_rank))

    caches: dict[tuple[str, float], dict[str, Any]] = {}
    for (weighting, floor), max_rank in grouped.items():
        residual_mean = residual.mean(axis=0)
        centered_residual = residual - residual_mean
        if weighting == "uniform":
            residual_weights = np.ones(centered_residual.shape[1], dtype=np.float64)
        elif weighting == "residual_energy":
            if floor <= 0:
                raise ValueError("residual_weight_floor must be positive")
            residual_weights = np.sqrt(np.mean(centered_residual**2, axis=0) + 1e-12)
            residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
            residual_weights = np.maximum(residual_weights, floor)
            residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
        else:
            raise ValueError(f"Unknown residual_weighting: {weighting}")
        weighted_residual = centered_residual * residual_weights
        actual_rank = min(max_rank, weighted_residual.shape[0], weighted_residual.shape[1])
        if actual_rank <= 0:
            raise ValueError("residual_rank must be positive")
        _, _, vt = randomized_svd(
            weighted_residual,
            n_components=actual_rank,
            n_iter=4,
            random_state=random_state,
        )
        caches[(weighting, floor)] = {
            "residual_mean": residual_mean,
            "residual_weights": residual_weights,
            "weighted_residual": weighted_residual,
            "residual_basis": vt,
            "max_rank": int(actual_rank),
        }
    return caches


def fit_cached_ridge_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    candidate: ResidualCandidate,
    residual_basis_caches: dict[tuple[str, float], dict[str, Any]],
    *,
    feature_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    model_features = coeffs if feature_matrix is None else np.asarray(feature_matrix, dtype=np.float64)
    features = np.concatenate(
        [model_features, np.ones((model_features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    if candidate.alpha < 0:
        raise ValueError("alpha must be non-negative")

    mode = "full"
    residual_basis = None
    residual_mean = None
    residual_weights = None
    actual_residual_rank = None
    regression_target = target_flat - base_flat
    if candidate.head_type != "global_residual_svd":
        raise ValueError("fit_cached_ridge_residual_corrector only supports global residual SVD")
    if candidate.residual_rank is not None:
        key = (candidate.residual_weighting, float(candidate.residual_weight_floor))
        basis_cache = residual_basis_caches[key]
        actual_residual_rank = min(int(candidate.residual_rank), int(basis_cache["max_rank"]))
        residual_basis = np.asarray(basis_cache["residual_basis"], dtype=np.float64)[
            :actual_residual_rank
        ]
        residual_mean = np.asarray(basis_cache["residual_mean"], dtype=np.float64)
        residual_weights = np.asarray(basis_cache["residual_weights"], dtype=np.float64)
        regression_target = (
            np.asarray(basis_cache["weighted_residual"], dtype=np.float64) @ residual_basis.T
        )
        mode = "residual_svd"

    gram = features.T @ features
    penalty = candidate.alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = features.T @ regression_target
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]

    corrector = {
        "alpha": float(candidate.alpha),
        "weights": weights,
        "feature_dim": int(model_features.shape[1]),
        "coefficient_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
        "mode": mode,
        "residual_rank": actual_residual_rank,
        "residual_weighting": candidate.residual_weighting if mode == "residual_svd" else "uniform",
    }
    if mode == "residual_svd":
        corrector["residual_basis"] = residual_basis
        corrector["residual_mean"] = residual_mean
        corrector["residual_weights"] = residual_weights
    return corrector


def build_split_cache(
    states: np.ndarray,
    *,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    base_config: CachedBaseConfig,
) -> dict[str, Any]:
    split_start = time.perf_counter()
    train_states = np.asarray(states[train_idx], dtype=np.float64)
    dev_states = np.asarray(states[dev_idx], dtype=np.float64)
    spatial_mean = np.mean(train_states, axis=(0, 1))
    train_centered = train_states - spatial_mean
    dev_centered = dev_states - spatial_mean
    train_segments = build_forecast_segment_tensor(
        train_centered,
        history_length=base_config.history_length,
        stride=1,
        max_segments=base_config.max_train_segments,
    )
    dev_segments = build_forecast_segment_tensor(
        dev_centered,
        history_length=base_config.history_length,
        stride=1,
        max_segments=None,
    )
    dictionary, tbmd_summary = fit_segment_dictionary(
        train_segments,
        ranks=list(base_config.ranks),
        random_state=base_config.random_state,
        dtype=base_config.dtype,
    )
    history_dictionary, _ = history_and_target(dictionary)
    spatial_mask, sensor_indices = place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=base_config.n_spatial_sensors,
        random_state=base_config.random_state,
    )
    decoder_payload = fit_sensor_coefficient_decoder(
        dictionary,
        sensor_indices,
        decoder=base_config.sensor_decoder,
        rcond=base_config.sensor_rcond,
        ridge_lambda=base_config.decoder_ridge_lambda,
    )
    train_base, train_coeffs, train_measurements = predict_next_sensor_decoder_with_measurements(
        train_segments,
        dictionary,
        sensor_indices,
        decoder_payload,
    )
    dev_base, dev_coeffs, dev_measurements = predict_next_sensor_decoder_with_measurements(
        dev_segments,
        dictionary,
        sensor_indices,
        decoder_payload,
    )
    train_targets = target_frames_from_segments(train_segments)
    dev_targets = target_frames_from_segments(dev_segments)
    elapsed = time.perf_counter() - split_start
    return {
        "train_targets": train_targets,
        "train_base": train_base,
        "train_coeffs": train_coeffs,
        "train_measurements": train_measurements,
        "dev_targets": dev_targets,
        "dev_base": dev_base,
        "dev_coeffs": dev_coeffs,
        "dev_measurements": dev_measurements,
        "decoder_payload": decoder_payload,
        "dictionary": dictionary,
        "base_train_metrics": _metrics_with_mae(train_targets, train_base),
        "base_dev_metrics": _metrics_with_mae(dev_targets, dev_base),
        "tbmd_summary": tbmd_summary,
        "dictionary_shape": list(dictionary.shape),
        "actual_spatial_sensors": int(spatial_mask.sum()),
        "sensing_diagnostics": _sensing_diagnostics(dictionary, sensor_indices),
        "cache_time_seconds": float(elapsed),
        "train_shape": list(train_states.shape),
        "dev_shape": list(dev_states.shape),
        "train_segments_shape": list(train_segments.shape),
        "dev_segments_shape": list(dev_segments.shape),
    }


def evaluate_residual_candidate(
    cache: dict[str, Any],
    candidate: ResidualCandidate,
    residual_basis_caches: dict[tuple[str, float], dict[str, Any]],
) -> dict[str, Any]:
    start = time.perf_counter()
    coefficient_calibrator = fit_coefficient_calibrator(
        cache["train_coeffs"],
        cache["train_targets"],
        cache["dictionary"],
        calibration_type=candidate.coefficient_calibration_type,
        target="target",
        alpha=candidate.coefficient_calibration_alpha,
        blend=candidate.coefficient_calibration_blend,
        rcond=1e-6,
    )
    train_coeffs = apply_coefficient_calibrator(
        cache["train_coeffs"],
        coefficient_calibrator,
    )
    dev_coeffs = apply_coefficient_calibrator(
        cache["dev_coeffs"],
        coefficient_calibrator,
    )
    train_base = (
        cache["train_base"]
        if coefficient_calibrator.get("type", "none") == "none"
        else reconstruct_target_from_coefficients(train_coeffs, cache["dictionary"])
    )
    dev_base = (
        cache["dev_base"]
        if coefficient_calibrator.get("type", "none") == "none"
        else reconstruct_target_from_coefficients(dev_coeffs, cache["dictionary"])
    )
    innovation_encoder = fit_sensor_innovation_encoder(
        cache["train_measurements"],
        train_coeffs,
        cache["decoder_payload"],
        rank=candidate.innovation_rank,
        include_norms=candidate.innovation_include_norms,
        random_state=int(cache.get("split_index", 0)),
    )
    feature_probe = {"innovation_encoder": innovation_encoder}
    train_feature_matrix = build_correction_feature_matrix(
        train_coeffs,
        feature_probe,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    if candidate.head_type == "patch_residual_svd":
        if candidate.patch_size is None or candidate.patch_residual_rank is None:
            raise ValueError("patch_residual_svd candidates require patch_size and patch_residual_rank")
        corrector = fit_patch_residual_svd_corrector(
            cache["train_targets"],
            train_base,
            train_coeffs,
            alpha=candidate.alpha,
            patch_size=candidate.patch_size,
            patch_residual_rank=candidate.patch_residual_rank,
            residual_weighting=candidate.residual_weighting,
            residual_weight_floor=candidate.residual_weight_floor,
            feature_matrix=train_feature_matrix,
        )
    else:
        if coefficient_calibrator.get("type", "none") == "none":
            corrector = fit_cached_ridge_residual_corrector(
                cache["train_targets"],
                train_base,
                train_coeffs,
                candidate,
                residual_basis_caches,
                feature_matrix=train_feature_matrix,
            )
        else:
            corrector = fit_ridge_residual_corrector(
                cache["train_targets"],
                train_base,
                train_coeffs,
                alpha=candidate.alpha,
                residual_rank=candidate.residual_rank,
                residual_weighting=candidate.residual_weighting,
                residual_weight_floor=candidate.residual_weight_floor,
                feature_matrix=train_feature_matrix,
            )
    corrector["innovation_encoder"] = innovation_encoder
    corrector = attach_coefficient_gate(
        corrector,
        train_coeffs,
        gate_type=candidate.gate_type,
        threshold=candidate.gate_threshold,
        strength=candidate.gate_strength,
        gate_min=candidate.gate_min,
    )
    train_pred = apply_ridge_residual_corrector(
        train_base,
        train_coeffs,
        corrector,
        scale=candidate.scale,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    dev_pred = apply_ridge_residual_corrector(
        dev_base,
        dev_coeffs,
        corrector,
        scale=candidate.scale,
        measurements=cache["dev_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    elapsed = time.perf_counter() - start
    train_metrics = _metrics_with_mae(cache["train_targets"], train_pred)
    dev_metrics = _metrics_with_mae(cache["dev_targets"], dev_pred)
    base_train_metrics = _metrics_with_mae(cache["train_targets"], train_base)
    base_dev_metrics = _metrics_with_mae(cache["dev_targets"], dev_base)
    return {
        "candidate": candidate.name,
        "residual_rank": candidate.residual_rank,
        "actual_residual_rank": corrector.get("residual_rank"),
        "head_type": candidate.head_type,
        "patch_size": candidate.patch_size,
        "patch_residual_rank": candidate.patch_residual_rank,
        "actual_patch_residual_rank": corrector.get("patch_residual_rank"),
        "gate_type": candidate.gate_type,
        "gate_threshold": candidate.gate_threshold,
        "gate_strength": candidate.gate_strength,
        "gate_min": candidate.gate_min,
        "innovation_rank": candidate.innovation_rank,
        "actual_innovation_rank": innovation_encoder.get("innovation_rank", 0),
        "innovation_include_norms": candidate.innovation_include_norms,
        "coefficient_calibration_type": candidate.coefficient_calibration_type,
        "coefficient_calibration_blend": candidate.coefficient_calibration_blend,
        "coefficient_calibration_alpha": candidate.coefficient_calibration_alpha,
        "base_train_r2": base_train_metrics["r2"],
        "base_dev_r2": base_dev_metrics["r2"],
        "base_dev_rmse": base_dev_metrics["rmse"],
        "scale": candidate.scale,
        "alpha": candidate.alpha,
        "residual_weighting": candidate.residual_weighting,
        "residual_weight_floor": candidate.residual_weight_floor,
        "train_r2": train_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_rel_frob_err": train_metrics["rel_frob_err"],
        "dev_r2": dev_metrics["r2"],
        "dev_rmse": dev_metrics["rmse"],
        "dev_mae": dev_metrics["mae"],
        "dev_rel_frob_err": dev_metrics["rel_frob_err"],
        "correction_fit_eval_time_seconds": float(elapsed),
    }


def aggregate_candidate_results(split_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in split_results:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    aggregated = []
    for name, rows in by_candidate.items():
        dev_r2 = np.array([row["dev_r2"] for row in rows], dtype=np.float64)
        dev_rmse = np.array([row["dev_rmse"] for row in rows], dtype=np.float64)
        dev_mae = np.array([row["dev_mae"] for row in rows], dtype=np.float64)
        train_r2 = np.array([row["train_r2"] for row in rows], dtype=np.float64)
        first = rows[0]
        aggregated.append(
            {
                "candidate": name,
                "residual_rank": first["residual_rank"],
                "actual_residual_rank": first["actual_residual_rank"],
                "head_type": first["head_type"],
                "patch_size": first["patch_size"],
                "patch_residual_rank": first["patch_residual_rank"],
                "actual_patch_residual_rank": first["actual_patch_residual_rank"],
                "gate_type": first["gate_type"],
                "gate_threshold": first["gate_threshold"],
                "gate_strength": first["gate_strength"],
                "gate_min": first["gate_min"],
                "innovation_rank": first["innovation_rank"],
                "actual_innovation_rank": first["actual_innovation_rank"],
                "innovation_include_norms": first["innovation_include_norms"],
                "coefficient_calibration_type": first["coefficient_calibration_type"],
                "coefficient_calibration_blend": first["coefficient_calibration_blend"],
                "coefficient_calibration_alpha": first["coefficient_calibration_alpha"],
                "mean_base_dev_r2": float(np.mean([row["base_dev_r2"] for row in rows])),
                "mean_base_dev_rmse": float(np.mean([row["base_dev_rmse"] for row in rows])),
                "scale": first["scale"],
                "alpha": first["alpha"],
                "residual_weighting": first["residual_weighting"],
                "n_splits": len(rows),
                "mean_dev_r2": float(np.mean(dev_r2)),
                "std_dev_r2": float(np.std(dev_r2)),
                "worst_dev_r2": float(np.min(dev_r2)),
                "best_dev_r2": float(np.max(dev_r2)),
                "mean_dev_rmse": float(np.mean(dev_rmse)),
                "mean_dev_mae": float(np.mean(dev_mae)),
                "mean_train_r2": float(np.mean(train_r2)),
                "mean_correction_time_seconds": float(
                    np.mean([row["correction_fit_eval_time_seconds"] for row in rows])
                ),
            }
        )
    return sorted(
        aggregated,
        key=lambda item: (
            item["mean_dev_r2"],
            item["worst_dev_r2"],
            -item["mean_dev_rmse"],
        ),
        reverse=True,
    )


def select_best_robust_result(aggregated: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregated:
        raise ValueError("aggregated results must not be empty")
    return max(
        aggregated,
        key=lambda item: (
            item["mean_dev_r2"],
            item["worst_dev_r2"],
            -item["mean_dev_rmse"],
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 5 Cached Residual Sweep",
        "",
        "Protocol: train-only multi-dev sweep. Official test is not loaded or evaluated.",
        "",
        "## Selected",
        "",
    ]
    selected = payload["selected_by_multi_dev"]
    lines.append(
        "| candidate | head | mean dev R2 | worst dev R2 | std dev R2 | mean RMSE | scale | rank | patch | calib | innov | gate | weighting |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|")
    lines.append(
        "| {candidate} | {head_type} | {mean_dev_r2:.4f} | {worst_dev_r2:.4f} | "
        "{std_dev_r2:.4f} | {mean_dev_rmse:.4f} | {scale:g} | {rank} | {patch} | "
        "{calib} | {innov} | {gate} | {residual_weighting} |".format(
            rank=selected.get("actual_residual_rank") or selected.get("actual_patch_residual_rank"),
            patch=(
                ""
                if selected.get("patch_size") is None
                else f"{selected.get('patch_size')}/{selected.get('actual_patch_residual_rank')}"
            ),
            gate=(
                "none"
                if selected.get("gate_type") == "none"
                else f"{selected.get('gate_threshold')}/{selected.get('gate_strength')}"
            ),
            innov=(
                ""
                if selected.get("actual_innovation_rank", 0) == 0
                and not selected.get("innovation_include_norms")
                else f"{selected.get('actual_innovation_rank')}/norms={selected.get('innovation_include_norms')}"
            ),
            calib=(
                "none"
                if selected.get("coefficient_calibration_type") == "none"
                else f"{selected.get('coefficient_calibration_type')}/{selected.get('coefficient_calibration_blend')}"
            ),
            **selected,
        )
    )
    lines.extend(["", "## Top Candidates", ""])
    lines.append(
        "| candidate | head | mean dev R2 | worst dev R2 | std dev R2 | mean RMSE | scale | rank | patch | calib | innov | gate | weighting |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for row in payload["aggregated_results"][:12]:
        lines.append(
            "| {candidate} | {head_type} | {mean_dev_r2:.4f} | {worst_dev_r2:.4f} | "
            "{std_dev_r2:.4f} | {mean_dev_rmse:.4f} | {scale:g} | {rank} | {patch} | "
            "{calib} | {innov} | {gate} | {residual_weighting} |".format(
                rank=row.get("actual_residual_rank") or row.get("actual_patch_residual_rank"),
                patch=(
                    ""
                    if row.get("patch_size") is None
                    else f"{row.get('patch_size')}/{row.get('actual_patch_residual_rank')}"
                ),
                gate=(
                    "none"
                    if row.get("gate_type") == "none"
                    else f"{row.get('gate_threshold')}/{row.get('gate_strength')}"
                ),
                innov=(
                    ""
                    if row.get("actual_innovation_rank", 0) == 0
                    and not row.get("innovation_include_norms")
                    else f"{row.get('actual_innovation_rank')}/norms={row.get('innovation_include_norms')}"
                ),
                calib=(
                    "none"
                    if row.get("coefficient_calibration_type") == "none"
                    else f"{row.get('coefficient_calibration_type')}/{row.get('coefficient_calibration_blend')}"
                ),
                **row,
            )
        )
    lines.extend(["", "## Base Diagnostics", ""])
    lines.append(
        "| split | base dev R2 | base dev RMSE | sensors | condition proxy | cache time sec | residual SVD sec |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["split_diagnostics"]:
        lines.append(
            "| {split_index} | {base_dev_r2:.4f} | {base_dev_rmse:.4f} | {actual_spatial_sensors} | "
            "{sensing_condition_proxy:.2f} | {cache_time_seconds:.2f} | "
            "{residual_basis_cache_time_seconds:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "fast"], default="smoke")
    parser.add_argument("--n-train-trajectories", type=int, default=None)
    parser.add_argument("--n-dev-trajectories", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--history-length", type=int, default=7)
    parser.add_argument("--ranks", type=int, nargs=4, default=None)
    parser.add_argument("--n-spatial-sensors", type=int, default=None)
    parser.add_argument("--max-train-segments", type=int, default=None)
    parser.add_argument("--sensor-decoder", choices=["lstsq", "ridge"], default="ridge")
    parser.add_argument("--decoder-ridge-lambda", type=float, default=1e-8)
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def resolve_limits(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.mode == "smoke":
        default_train = 96
        default_dev = 16
        default_splits = 2
    else:
        default_train = 800
        default_dev = 80
        default_splits = 2
    n_train = args.n_train_trajectories or default_train
    n_dev = args.n_dev_trajectories or default_dev
    n_splits = args.n_splits or default_splits
    if n_train > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(f"n-train-trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}")
    return n_train, n_dev, n_splits


def main() -> None:
    args = parse_args()
    n_train, n_dev, n_splits = resolve_limits(args)
    base_config = build_base_config(args.mode, args)
    candidates = build_residual_candidates(args.mode)

    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_pool = dataset.train_states[:n_train]
    blocks = build_dev_blocks(
        train_pool.shape[0],
        dev_count=n_dev,
        n_splits=n_splits,
    )

    split_results: list[dict[str, Any]] = []
    split_diagnostics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for split_index, (train_idx, dev_idx) in enumerate(blocks):
        cache = build_split_cache(
            train_pool,
            train_idx=train_idx,
            dev_idx=dev_idx,
            base_config=base_config,
        )
        base_dev = cache["base_dev_metrics"]
        sensing = cache["sensing_diagnostics"]
        residual_cache_start = time.perf_counter()
        residual_basis_caches = build_residual_basis_caches(
            cache["train_targets"],
            cache["train_base"],
            candidates,
            random_state=base_config.random_state + split_index,
        )
        residual_cache_time = time.perf_counter() - residual_cache_start
        split_diagnostics.append(
            {
                "split_index": split_index,
                "train_indices": train_idx.tolist(),
                "dev_indices": dev_idx.tolist(),
                "train_shape": cache["train_shape"],
                "dev_shape": cache["dev_shape"],
                "train_segments_shape": cache["train_segments_shape"],
                "dev_segments_shape": cache["dev_segments_shape"],
                "dictionary_shape": cache["dictionary_shape"],
                "actual_spatial_sensors": cache["actual_spatial_sensors"],
                "base_dev_r2": base_dev["r2"],
                "base_dev_rmse": base_dev["rmse"],
                "base_dev_mae": base_dev["mae"],
                "base_train_r2": cache["base_train_metrics"]["r2"],
                "cache_time_seconds": cache["cache_time_seconds"],
                "residual_basis_cache_time_seconds": float(residual_cache_time),
                **sensing,
            }
        )
        for candidate in candidates:
            row = evaluate_residual_candidate(cache, candidate, residual_basis_caches)
            row["split_index"] = split_index
            split_results.append(row)
        details.append(
            {
                "split_index": split_index,
                "tbmd_summary": cache["tbmd_summary"],
                "base_train_metrics": cache["base_train_metrics"],
                "base_dev_metrics": cache["base_dev_metrics"],
                "sensing_diagnostics": sensing,
                "residual_basis_cache_shapes": {
                    f"{key[0]}:{key[1]:g}": list(value["residual_basis"].shape)
                    for key, value in residual_basis_caches.items()
                },
                "residual_basis_cache_time_seconds": float(residual_cache_time),
            }
        )

    aggregated = aggregate_candidate_results(split_results)
    selected = select_best_robust_result(aggregated)
    payload = {
        "stage": "stage5_fast_tplus1_cached_residual",
        "protocol": (
            "Train-only multi-dev residual sweep. The official test split is not evaluated "
            "and must not be used for candidate selection."
        ),
        "mode": args.mode,
        "base_config": asdict(base_config),
        "limits": {
            "n_train_trajectories": n_train,
            "n_dev_trajectories_per_split": n_dev,
            "n_splits": n_splits,
            "official_test_evaluated": False,
        },
        "candidate_count": len(candidates),
        "split_diagnostics": split_diagnostics,
        "split_results": split_results,
        "aggregated_results": aggregated,
        "selected_by_multi_dev": selected,
        "details": details,
        "run_time_seconds": float(time.perf_counter() - run_start),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    _write_csv(args.output.with_suffix(".csv"), aggregated)
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(f"Saved cached residual sweep to {args.output}")
    print(
        "Selected {candidate}: mean_dev_r2={mean_dev_r2:.4f}, worst_dev_r2={worst_dev_r2:.4f}".format(
            **selected
        )
    )


if __name__ == "__main__":
    main()
