#!/usr/bin/env python3
"""Staged TBMD+QR+CS next-step forecasting diagnostics and fast decoder sweep.

The script is intentionally narrower than the rollout-oriented CS scripts.  It
targets one-step-ahead quality and separates three questions:

1. How good is the causal full-history TBMD projection upper bound?
2. How much quality is lost by QR-sensor coefficient recovery?
3. Can a lightweight sensor-conditioned latent forecaster improve over a
   sensor-free latent ridge baseline without iterative CS at inference?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.config import CompressiveSensingConfig
from TBMD.core.reconstruction.tensor_compressive_sensing import (
    ExtensionCompressiveSensingConfig,
    TensorCompressiveSensing,
)
from TBMD.experiments import (
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_fast_tplus1 import (
    build_forecast_segment_tensor,
    fit_segment_dictionary,
    history_and_target,
    place_fixed_spatial_sensors,
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
    / "tbmd_qr_cs_next_step_forecasting_summary.json"
)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


class RidgeSensorDecoder:
    """Batched sensor-to-coefficient ridge decoder with a precomputed map."""

    def __init__(self, ridge_lambda: float):
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        self.ridge_lambda = float(ridge_lambda)
        self.sensing_matrix_: np.ndarray | None = None
        self.decoder_matrix_: np.ndarray | None = None
        self.fit_time_sec_: float = 0.0

    def fit(self, sensing_matrix: np.ndarray) -> "RidgeSensorDecoder":
        start = time.perf_counter()
        sensing = np.asarray(sensing_matrix, dtype=np.float64)
        gram = sensing.T @ sensing
        penalty = self.ridge_lambda * np.eye(gram.shape[0], dtype=np.float64)
        rhs = sensing.T
        try:
            self.decoder_matrix_ = np.linalg.solve(gram + penalty, rhs)
        except np.linalg.LinAlgError:
            self.decoder_matrix_ = np.linalg.pinv(gram + penalty) @ rhs
        self.sensing_matrix_ = sensing
        self.fit_time_sec_ = time.perf_counter() - start
        return self

    def decode(self, measurements: np.ndarray) -> np.ndarray:
        if self.decoder_matrix_ is None:
            raise RuntimeError("Call fit() before decode().")
        batch = np.asarray(measurements, dtype=np.float64)
        return batch @ self.decoder_matrix_.T


class FistaSensorDecoder:
    """Batched FISTA decoder for ||Ax-y||_2^2 / 2 + lambda ||x||_1."""

    def __init__(self, l1_lambda: float, max_iter: int, tol: float = 1e-6):
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.l1_lambda = float(l1_lambda)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.sensing_matrix_: np.ndarray | None = None
        self.lipschitz_: float = 1.0
        self.fit_time_sec_: float = 0.0
        self.last_iterations_: int = 0

    def fit(self, sensing_matrix: np.ndarray) -> "FistaSensorDecoder":
        start = time.perf_counter()
        sensing = np.asarray(sensing_matrix, dtype=np.float64)
        self.sensing_matrix_ = sensing
        spectral_norm = float(np.linalg.norm(sensing, ord=2))
        self.lipschitz_ = max(spectral_norm * spectral_norm, 1e-12)
        self.fit_time_sec_ = time.perf_counter() - start
        return self

    def decode(self, measurements: np.ndarray) -> np.ndarray:
        if self.sensing_matrix_ is None:
            raise RuntimeError("Call fit() before decode().")
        sensing = self.sensing_matrix_
        batch = np.asarray(measurements, dtype=np.float64)
        coeffs = np.zeros((batch.shape[0], sensing.shape[1]), dtype=np.float64)
        momentum_state = coeffs.copy()
        momentum = 1.0
        threshold = self.l1_lambda / self.lipschitz_
        for iteration in range(1, self.max_iter + 1):
            residual = momentum_state @ sensing.T - batch
            gradient = residual @ sensing
            next_coeffs = _soft_threshold(
                momentum_state - gradient / self.lipschitz_,
                threshold,
            )
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
            self.last_iterations_ = iteration
            if self.tol > 0.0 and rel_change < self.tol:
                break
        return coeffs


def build_experiment_plan(mode: str) -> dict[str, Any]:
    """Return staged configs without touching official test outside full mode."""
    if mode not in {"smoke", "fast", "full"}:
        raise ValueError("mode must be one of: smoke, fast, full")

    limits_by_mode = {
        "smoke": {
            "n_train_trajectories": 48,
            "dev_split": 0.25,
            "n_test_trajectories": 0,
            "max_train_segments": 96,
            "max_eval_segments": 24,
            "evaluate_official_test": False,
        },
        "fast": {
            "n_train_trajectories": 240,
            "dev_split": 0.2,
            "n_test_trajectories": 0,
            "max_train_segments": 384,
            "max_eval_segments": 160,
            "evaluate_official_test": False,
        },
        "full": {
            "n_train_trajectories": DEFAULT_N_TRAIN_TRAJECTORIES,
            "dev_split": 0.2,
            "n_test_trajectories": 200,
            "max_train_segments": 2048,
            "max_eval_segments": None,
            "evaluate_official_test": True,
        },
    }
    limits = limits_by_mode[mode]

    if mode == "smoke":
        decoder_candidates = [
            {
                "candidate": "smoke_r30_s30_admm_i10_l1_1e-3_h5",
                "family": "decoder_audit",
                "decoder": "current_admm",
                "rank": 30,
                "n_sensors": 30,
                "history_length": 5,
                "ranks": [6, 32, 32, 30],
                "cs_max_iter": 10,
                "cs_tol": 1e-4,
                "l1_lambda": 1e-3,
            },
            {
                "candidate": "smoke_r30_s30_ridge_lam_1e-4_h5",
                "family": "decoder_audit",
                "decoder": "ridge_sensor_decoder",
                "rank": 30,
                "n_sensors": 30,
                "history_length": 5,
                "ranks": [6, 32, 32, 30],
                "ridge_lambda": 1e-4,
            },
            {
                "candidate": "smoke_r30_s30_fista_l1_1e-3_i15_h5",
                "family": "decoder_audit",
                "decoder": "fista_decoder",
                "rank": 30,
                "n_sensors": 30,
                "history_length": 5,
                "ranks": [6, 32, 32, 30],
                "l1_lambda": 1e-3,
                "fista_max_iter": 15,
            },
        ]
        hybrid_candidates = [
            {
                "candidate": "smoke_sensor_conditioned_svd_r10_s30_h5",
                "family": "hybrid_sensor_conditioned",
                "history_length": 5,
                "latent_rank": 10,
                "n_sensors": 30,
                "tbmd_ranks": [6, 32, 32, 30],
                "ridge_lambda": 1e-4,
            }
        ]
    else:
        rank_grid = [30, 45] if mode == "fast" else [30, 45, 60]
        history_grid = [5, 7] if mode == "fast" else [5, 7, 10]
        ridge_grid = [1e-6, 1e-4, 1e-2] if mode == "fast" else [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        l1_grid = [1e-4, 1e-3] if mode == "fast" else [1e-4, 1e-3, 1e-2]
        decoder_candidates = []
        for rank in rank_grid:
            sensor_grid = sorted({max(rank - 2, 1), rank, rank + 10})
            for history_length in history_grid:
                ranks = [min(history_length + 1, 8), 32, 32, rank]
                for n_sensors in sensor_grid:
                    for ridge_lambda in ridge_grid:
                        decoder_candidates.append(
                            {
                                "candidate": (
                                    f"r{rank}_s{n_sensors}_ridge_lam_{ridge_lambda:g}"
                                    f"_h{history_length}"
                                ),
                                "family": "decoder_audit",
                                "decoder": "ridge_sensor_decoder",
                                "rank": rank,
                                "n_sensors": n_sensors,
                                "history_length": history_length,
                                "ranks": ranks,
                                "ridge_lambda": ridge_lambda,
                            }
                        )
                    for l1_lambda in l1_grid:
                        decoder_candidates.append(
                            {
                                "candidate": (
                                    f"r{rank}_s{n_sensors}_fista_l1_{l1_lambda:g}"
                                    f"_i25_h{history_length}"
                                ),
                                "family": "decoder_audit",
                                "decoder": "fista_decoder",
                                "rank": rank,
                                "n_sensors": n_sensors,
                                "history_length": history_length,
                                "ranks": ranks,
                                "l1_lambda": l1_lambda,
                                "fista_max_iter": 25,
                            }
                        )
                    if mode == "full" and n_sensors == rank and history_length == 7:
                        decoder_candidates.append(
                            {
                                "candidate": f"r{rank}_s{n_sensors}_admm_i100_l1_1e-3_h7",
                                "family": "decoder_audit",
                                "decoder": "current_admm",
                                "rank": rank,
                                "n_sensors": n_sensors,
                                "history_length": history_length,
                                "ranks": ranks,
                                "cs_max_iter": 100,
                                "cs_tol": 1e-4,
                                "l1_lambda": 1e-3,
                            }
                        )
        hybrid_candidates = [
            {
                "candidate": f"sensor_conditioned_svd_r{latent_rank}_s{sensors}_h{history_length}",
                "family": "hybrid_sensor_conditioned",
                "history_length": history_length,
                "latent_rank": latent_rank,
                "n_sensors": sensors,
                "tbmd_ranks": [min(history_length + 1, 8), 32, 32, max(30, sensors)],
                "ridge_lambda": 1e-4,
            }
            for history_length in ([5, 7] if mode == "fast" else [5, 7, 10])
            for latent_rank in ([10, 15] if mode == "fast" else [10, 15, 20])
            for sensors in ([30, 45] if mode == "fast" else [30, 45, 60])
        ]

    return {
        "mode": mode,
        "limits": limits,
        "selection_metric": "dev_one_step_r2",
        "selection_tie_breakers": [
            "lower dev_rmse",
            "lower inference_ms_per_step",
            "lower reconstruction rel_frob",
            "fewer sensors",
        ],
        "decoder_candidates": decoder_candidates,
        "hybrid_candidates": hybrid_candidates,
    }


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No results to select from")

    def key(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(result.get("dev_one_step_r2", -np.inf)),
            -float(result.get("dev_rmse", np.inf)),
            -float(result.get("inference_ms_per_step", np.inf)),
            -float(result.get("dev_rel_frob", np.inf)),
            -float(result.get("n_sensors", np.inf)),
        )

    return max(results, key=key)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_markdown_report(output_path: Path, payload: dict[str, Any]) -> Path:
    report_path = output_path.with_suffix(".md")
    selected = payload["selected_by_dev"]
    final = payload.get("final_test_result")
    best_decoder = max(
        (item for item in payload["results"] if item["family"] == "decoder_audit"),
        key=lambda item: item["dev_one_step_r2"],
        default=None,
    )
    best_hybrid = max(
        (item for item in payload["results"] if item["family"] == "hybrid_sensor_conditioned"),
        key=lambda item: item["dev_one_step_r2"],
        default=None,
    )
    lines = [
        "# TBMD+QR+CS Next-Step Forecasting Sweep",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Train split: `{payload['train_shape']}`",
        f"- Dev split: `{payload['dev_shape']}`",
        f"- Official test touched: `{payload['official_test_shape'] is not None}`",
        f"- Selected by dev: `{selected['candidate']}`",
        f"- Selected dev R2: `{selected['dev_one_step_r2']:.4f}`",
        f"- Selected dev RMSE: `{selected['dev_rmse']:.4f}`",
        f"- Selected inference ms/step: `{selected['inference_ms_per_step']:.4f}`",
        "",
        "## Best Decoder",
        "",
    ]
    if best_decoder is None:
        lines.append("No decoder candidate was evaluated.")
    else:
        lines.extend(
            [
                f"- Candidate: `{best_decoder['candidate']}`",
                f"- Decoder: `{best_decoder['decoder']}`",
                f"- Rank/sensors/history: `{best_decoder['rank']}/{best_decoder['n_sensors']}/{best_decoder['history_length']}`",
                f"- Dev R2/RMSE: `{best_decoder['dev_one_step_r2']:.4f}` / `{best_decoder['dev_rmse']:.4f}`",
                f"- Oracle full-history R2: `{best_decoder['oracle_projection_forecast']['r2']:.4f}`",
                f"- Coeff R2 vs full-history: `{best_decoder['sensor_recovery_only']['coeff_r2_vs_full_history']:.4f}`",
                f"- Sensor residual rel-Frob: `{best_decoder['sensor_recovery_only']['sensor_residual_rel_frob']:.4f}`",
                f"- Condition number: `{best_decoder['condition_number_sensor_matrix']:.4f}`",
            ]
        )
    lines.extend(["", "## Best Hybrid", ""])
    if best_hybrid is None:
        lines.append("No hybrid candidate was evaluated.")
    else:
        lines.extend(
            [
                f"- Candidate: `{best_hybrid['candidate']}`",
                f"- Dev R2/RMSE: `{best_hybrid['dev_one_step_r2']:.4f}` / `{best_hybrid['dev_rmse']:.4f}`",
                f"- Sensor gain R2: `{best_hybrid['sensor_gain_r2']:.4f}`",
                "- Interpretation: positive gain means QR history measurements helped the latent ridge baseline; negative gain means sensor features overfit or destabilized the small split.",
            ]
        )
    lines.extend(["", "## Official Test", ""])
    if final is None:
        lines.append("Not evaluated. This is expected for `smoke` and `fast` modes.")
    else:
        lines.extend(
            [
                f"- Candidate: `{final['candidate']}`",
                f"- Test R2/RMSE: `{final['dev_one_step_r2']:.4f}` / `{final['dev_rmse']:.4f}`",
                "- Note: in full mode this is the only official-test evaluation after dev selection.",
            ]
        )
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- `ridge_sensor_decoder` and `fista_decoder` precompute/vectorize coefficient recovery and avoid a per-step ADMM loop.",
            "- `current_admm` is retained as a faithful QR+CS reference.",
            "- `hybrid_sensor_conditioned` tests whether QR history measurements improve a lightweight SVD-latent ridge forecaster.",
            "- Smoke/fast results are train/dev diagnostics and must not be reported as official-test metrics.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _metric_payload(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics = _compute_regression_metrics(target, pred)
    return {
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "mae": float(np.mean(np.abs(np.asarray(target) - np.asarray(pred)))),
        "rel_frob": metrics["rel_frob_err"],
    }


def _sensor_sensing_matrix(dictionary: np.ndarray, sensor_indices: np.ndarray) -> np.ndarray:
    history_dictionary, _ = history_and_target(dictionary)
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history_dictionary = history_dictionary.reshape(
        history_dictionary.shape[0],
        height_width,
        history_dictionary.shape[-1],
    )
    return flat_history_dictionary[:, sensor_indices, :].reshape(
        -1,
        history_dictionary.shape[-1],
    )


def _full_history_matrix(dictionary: np.ndarray) -> np.ndarray:
    history_dictionary, _ = history_and_target(dictionary)
    return history_dictionary.reshape(-1, history_dictionary.shape[-1])


def _history_sensor_measurements(
    segments: np.ndarray,
    dictionary: np.ndarray,
    sensor_indices: np.ndarray,
) -> np.ndarray:
    history_dictionary, _ = history_and_target(dictionary)
    history = segments[:-1]
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history = history.reshape(history_dictionary.shape[0], height_width, history.shape[-1])
    return flat_history[:, sensor_indices, :].reshape(-1, history.shape[-1]).T


def _full_history_measurements(segments: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    history_dictionary, _ = history_and_target(dictionary)
    history = segments[:-1]
    return history.reshape(-1, history.shape[-1]).T.reshape(
        history.shape[-1],
        history_dictionary.shape[0] * history_dictionary.shape[1] * history_dictionary.shape[2],
    )


def _predict_target_from_coeffs(coeffs: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    _, target_dictionary = history_and_target(dictionary)
    target_flat = target_dictionary.reshape(-1, target_dictionary.shape[-1])
    pred = coeffs @ target_flat.T
    return pred.reshape(coeffs.shape[0], *target_dictionary.shape[:-1])


def _lstsq_coefficients(
    measurements: np.ndarray, matrix: np.ndarray, rcond: float = 1e-8
) -> np.ndarray:
    return measurements @ np.linalg.pinv(matrix, rcond=rcond).T


def _coefficient_diagnostics(
    coeffs: np.ndarray,
    reference_coeffs: np.ndarray,
    measurements: np.ndarray,
    sensing_matrix: np.ndarray,
) -> dict[str, float]:
    coeff_metrics = _compute_regression_metrics(reference_coeffs, coeffs)
    residual = measurements - coeffs @ sensing_matrix.T
    coeff_abs = np.abs(coeffs)
    row_scale = np.maximum(np.max(coeff_abs, axis=1, keepdims=True), 1e-12)
    active = coeff_abs > (1e-3 * row_scale)
    return {
        "coeff_rmse_vs_full_history": coeff_metrics["rmse"],
        "coeff_rel_frob_vs_full_history": coeff_metrics["rel_frob_err"],
        "coeff_r2_vs_full_history": coeff_metrics["r2"],
        "sensor_residual_rel_frob": float(
            np.linalg.norm(residual) / max(np.linalg.norm(measurements), 1e-12)
        ),
        "mean_active_coefficients_rel_1e_3": float(np.mean(np.sum(active, axis=1))),
        "mean_l1_l2_ratio": float(
            np.mean(np.sum(coeff_abs, axis=1) / np.maximum(np.linalg.norm(coeffs, axis=1), 1e-12))
        ),
    }


def _condition_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")


def _decode_admm_coefficients(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_mask: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    l1_lambda: float,
) -> tuple[np.ndarray, dict[str, float]]:
    history_dictionary, _ = history_and_target(dictionary)
    history_mask = np.broadcast_to(spatial_mask, history_dictionary.shape[:-1]).copy()
    core_cfg = CompressiveSensingConfig(
        max_iter=max_iter,
        tol=tol,
        epsilon_l1=l1_lambda,
        device="cpu",
        dtype=torch.float32,
    )
    ext_cfg = ExtensionCompressiveSensingConfig(solver="cholesky", collect_history=False)
    coeffs = np.zeros((segments.shape[-1], dictionary.shape[-1]), dtype=np.float64)
    iterations = []
    converged = []
    objectives = []
    for idx in range(segments.shape[-1]):
        solver = TensorCompressiveSensing(
            history_dictionary.astype(np.float32),
            history_mask,
            segments[:-1, :, :, idx].astype(np.float32),
            core_cfg=core_cfg,
            ext_cfg=ext_cfg,
        )
        coeff, metrics = solver.solve()
        coeffs[idx] = coeff.detach().cpu().numpy().astype(np.float64, copy=False)
        iterations.append(int(metrics.iterations))
        converged.append(bool(metrics.converged))
        objectives.append(float(metrics.objective))
    return coeffs, {
        "mean_iterations": float(np.mean(iterations)),
        "convergence_rate": float(np.mean(converged)),
        "mean_objective": float(np.mean(objectives)),
    }


def _prepare_dictionary_and_segments(
    train_states: np.ndarray,
    eval_states: np.ndarray,
    candidate: dict[str, Any],
    limits: dict[str, Any],
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    spatial_mean = np.mean(train_states, axis=(0, 1))
    centered_train = train_states - spatial_mean
    centered_eval = eval_states - spatial_mean
    train_segments = build_forecast_segment_tensor(
        centered_train,
        history_length=int(candidate["history_length"]),
        stride=1,
        max_segments=limits["max_train_segments"],
    )
    eval_segments = build_forecast_segment_tensor(
        centered_eval,
        history_length=int(candidate["history_length"]),
        stride=1,
        max_segments=limits["max_eval_segments"],
    )
    dictionary, tbmd_summary = fit_segment_dictionary(
        train_segments,
        ranks=list(candidate["ranks"]),
        random_state=random_state,
        dtype="float32",
    )
    history_dictionary, _ = history_and_target(dictionary)
    spatial_mask, sensor_indices = place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=int(candidate["n_sensors"]),
        random_state=random_state,
    )
    return (
        spatial_mean,
        train_segments,
        eval_segments,
        dictionary,
        spatial_mask,
        tbmd_summary,
        sensor_indices,
    )


def evaluate_decoder_candidate(
    candidate: dict[str, Any],
    train_states: np.ndarray,
    eval_states: np.ndarray,
    limits: dict[str, Any],
    *,
    random_state: int,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    (
        spatial_mean,
        _train_segments,
        eval_segments,
        dictionary,
        spatial_mask,
        tbmd_summary,
        sensor_indices,
    ) = _prepare_dictionary_and_segments(
        train_states,
        eval_states,
        candidate,
        limits,
        random_state=random_state,
    )
    sensor_matrix = _sensor_sensing_matrix(dictionary, sensor_indices)
    full_matrix = _full_history_matrix(dictionary)
    sensor_measurements = _history_sensor_measurements(eval_segments, dictionary, sensor_indices)
    full_measurements = _full_history_measurements(eval_segments, dictionary)
    target = target_frames_from_segments(eval_segments) + spatial_mean

    oracle_coeffs = _lstsq_coefficients(full_measurements, full_matrix, rcond=1e-8)
    oracle_pred = _predict_target_from_coeffs(oracle_coeffs, dictionary) + spatial_mean
    oracle_metrics = _metric_payload(target, oracle_pred)

    decoder_fit_time = 0.0
    decode_start = time.perf_counter()
    decoder_extra: dict[str, Any] = {}
    if candidate["decoder"] == "ridge_sensor_decoder":
        decoder = RidgeSensorDecoder(float(candidate["ridge_lambda"])).fit(sensor_matrix)
        decoder_fit_time = decoder.fit_time_sec_
        coeffs = decoder.decode(sensor_measurements)
    elif candidate["decoder"] == "fista_decoder":
        decoder = FistaSensorDecoder(
            float(candidate["l1_lambda"]),
            max_iter=int(candidate["fista_max_iter"]),
            tol=float(candidate.get("fista_tol", 1e-6)),
        ).fit(sensor_matrix)
        decoder_fit_time = decoder.fit_time_sec_
        coeffs = decoder.decode(sensor_measurements)
        decoder_extra["fista_iterations"] = decoder.last_iterations_
    elif candidate["decoder"] == "current_admm":
        coeffs, decoder_extra = _decode_admm_coefficients(
            eval_segments,
            dictionary,
            spatial_mask,
            max_iter=int(candidate.get("cs_max_iter", 100)),
            tol=float(candidate.get("cs_tol", 1e-4)),
            l1_lambda=float(candidate.get("l1_lambda", 1e-3)),
        )
    else:
        raise ValueError(f"Unknown decoder: {candidate['decoder']}")
    decode_time = time.perf_counter() - decode_start

    pred = _predict_target_from_coeffs(coeffs, dictionary) + spatial_mean
    field_metrics = _metric_payload(target, pred)
    coeff_diag = _coefficient_diagnostics(
        coeffs,
        oracle_coeffs,
        sensor_measurements,
        sensor_matrix,
    )
    n_samples = int(target.shape[0])
    total_time = time.perf_counter() - total_start
    return {
        **candidate,
        "n_eval_samples": n_samples,
        "actual_sensors": int(spatial_mask.sum()),
        "total_history_measurements": int(spatial_mask.sum() * candidate["history_length"]),
        "condition_number_sensor_matrix": _condition_number(sensor_matrix),
        "tbmd_summary": tbmd_summary,
        "oracle_projection_forecast": oracle_metrics,
        "sensor_recovery_only": coeff_diag,
        "forecast_given_recovered_coeffs": field_metrics,
        "full_rollout": {
            "status": "not_run",
            "reason": "This staged script is scoped to t+1; rollout remains in evaluate_windowed_tbmd_qr_cs_forecasting.py.",
        },
        "decoder_extra": decoder_extra,
        "decoder_fit_time_sec": decoder_fit_time,
        "decoder_time_sec": decode_time,
        "fit_and_eval_time_sec": total_time,
        "inference_ms_per_step": 1000.0 * decode_time / max(n_samples, 1),
        "dev_one_step_r2": field_metrics["r2"],
        "dev_rmse": field_metrics["rmse"],
        "dev_rel_frob": field_metrics["rel_frob"],
        "n_sensors": int(candidate["n_sensors"]),
    }


def _build_history_target_examples(
    states: np.ndarray,
    *,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    targets = []
    for start in range(0, states.shape[1] - history_length):
        histories.append(states[:, start : start + history_length])
        targets.append(states[:, start + history_length])
    return np.concatenate(histories, axis=0), np.concatenate(targets, axis=0)


def _standardize_with_train(
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_features - mean) / std, (eval_features - mean) / std, mean, std


def _fit_ridge_map(features: np.ndarray, target: np.ndarray, ridge_lambda: float) -> np.ndarray:
    augmented = np.concatenate(
        [features, np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    gram = augmented.T @ augmented
    penalty = ridge_lambda * np.eye(gram.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = augmented.T @ target
    try:
        return np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + penalty) @ rhs


def _apply_ridge_map(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate(
        [features, np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return augmented @ weights


def _history_sensor_features(history: np.ndarray, sensor_indices: np.ndarray) -> np.ndarray:
    flat = history.reshape(history.shape[0], history.shape[1], -1)
    return flat[:, :, sensor_indices].reshape(history.shape[0], -1)


def evaluate_sensor_conditioned_latent_candidate(
    candidate: dict[str, Any],
    train_states: np.ndarray,
    eval_states: np.ndarray,
    limits: dict[str, Any],
    *,
    random_state: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    history_length = int(candidate["history_length"])
    spatial_mean = np.mean(train_states, axis=(0, 1))
    centered_train = train_states - spatial_mean
    centered_eval = eval_states - spatial_mean

    train_segments = build_forecast_segment_tensor(
        centered_train,
        history_length=history_length,
        stride=1,
        max_segments=limits["max_train_segments"],
    )
    dictionary, tbmd_summary = fit_segment_dictionary(
        train_segments,
        ranks=list(candidate["tbmd_ranks"]),
        random_state=random_state,
        dtype="float32",
    )
    history_dictionary, _ = history_and_target(dictionary)
    spatial_mask, sensor_indices = place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=int(candidate["n_sensors"]),
        random_state=random_state,
    )

    train_history, train_target = _build_history_target_examples(
        centered_train,
        history_length=history_length,
    )
    eval_history, eval_target = _build_history_target_examples(
        centered_eval,
        history_length=history_length,
    )
    if (
        limits["max_train_segments"] is not None
        and train_history.shape[0] > limits["max_train_segments"]
    ):
        selected = np.linspace(
            0,
            train_history.shape[0] - 1,
            num=limits["max_train_segments"],
            dtype=int,
        )
        train_history = train_history[selected]
        train_target = train_target[selected]
    if (
        limits["max_eval_segments"] is not None
        and eval_history.shape[0] > limits["max_eval_segments"]
    ):
        selected = np.linspace(
            0,
            eval_history.shape[0] - 1,
            num=limits["max_eval_segments"],
            dtype=int,
        )
        eval_history = eval_history[selected]
        eval_target = eval_target[selected]

    flat_train_snapshots = centered_train.reshape(
        -1, centered_train.shape[2] * centered_train.shape[3]
    )
    latent_rank = min(
        int(candidate["latent_rank"]),
        flat_train_snapshots.shape[0],
        flat_train_snapshots.shape[1],
    )
    _, _, basis = randomized_svd(
        flat_train_snapshots,
        n_components=latent_rank,
        random_state=random_state,
    )

    train_latent_history = (train_history.reshape(-1, basis.shape[1]) @ basis.T).reshape(
        train_history.shape[0], history_length, latent_rank
    )
    eval_latent_history = (eval_history.reshape(-1, basis.shape[1]) @ basis.T).reshape(
        eval_history.shape[0], history_length, latent_rank
    )
    train_target_latent = train_target.reshape(train_target.shape[0], -1) @ basis.T

    train_latent_features = train_latent_history.reshape(train_latent_history.shape[0], -1)
    eval_latent_features = eval_latent_history.reshape(eval_latent_history.shape[0], -1)
    train_sensor_features = _history_sensor_features(train_history, sensor_indices)
    eval_sensor_features = _history_sensor_features(eval_history, sensor_indices)

    train_base_std, eval_base_std, _, _ = _standardize_with_train(
        train_latent_features,
        eval_latent_features,
    )
    train_hybrid_features = np.concatenate([train_latent_features, train_sensor_features], axis=1)
    eval_hybrid_features = np.concatenate([eval_latent_features, eval_sensor_features], axis=1)
    train_hybrid_std, eval_hybrid_std, _, _ = _standardize_with_train(
        train_hybrid_features,
        eval_hybrid_features,
    )

    target_mean = train_target_latent.mean(axis=0)
    target_std = train_target_latent.std(axis=0)
    target_std[target_std < 1e-8] = 1.0
    train_target_std = (train_target_latent - target_mean) / target_std

    ridge_lambda = float(candidate["ridge_lambda"])
    base_weights = _fit_ridge_map(train_base_std, train_target_std, ridge_lambda)
    hybrid_weights = _fit_ridge_map(train_hybrid_std, train_target_std, ridge_lambda)

    forecast_start = time.perf_counter()
    base_latent = _apply_ridge_map(eval_base_std, base_weights) * target_std + target_mean
    hybrid_latent = _apply_ridge_map(eval_hybrid_std, hybrid_weights) * target_std + target_mean
    forecast_time = time.perf_counter() - forecast_start

    base_pred = (base_latent @ basis).reshape(eval_target.shape) + spatial_mean
    hybrid_pred = (hybrid_latent @ basis).reshape(eval_target.shape) + spatial_mean
    target = eval_target + spatial_mean
    base_metrics = _metric_payload(target, base_pred)
    hybrid_metrics = _metric_payload(target, hybrid_pred)
    n_samples = int(target.shape[0])
    total_time = time.perf_counter() - start
    return {
        **candidate,
        "n_eval_samples": n_samples,
        "actual_sensors": int(spatial_mask.sum()),
        "total_history_measurements": int(spatial_mask.sum() * history_length),
        "tbmd_summary": tbmd_summary,
        "base_no_sensor_latent_ridge": base_metrics,
        "sensor_conditioned_latent_ridge": hybrid_metrics,
        "sensor_gain_r2": hybrid_metrics["r2"] - base_metrics["r2"],
        "sensor_gain_rmse": base_metrics["rmse"] - hybrid_metrics["rmse"],
        "fit_and_eval_time_sec": total_time,
        "forecast_time_sec": forecast_time,
        "inference_ms_per_step": 1000.0 * forecast_time / max(n_samples, 1),
        "dev_one_step_r2": hybrid_metrics["r2"],
        "dev_rmse": hybrid_metrics["rmse"],
        "dev_rel_frob": hybrid_metrics["rel_frob"],
        "n_sensors": int(candidate["n_sensors"]),
    }


def _load_protocol_data(limits: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    n_train = int(limits["n_train_trajectories"])
    all_train = dataset.train_states[:n_train]
    tuning_train, tuning_dev = split_train_dev_trajectories(
        all_train,
        dev_split=float(limits["dev_split"]),
    )
    official_test = None
    if limits["evaluate_official_test"]:
        n_test = limits["n_test_trajectories"]
        official_test = (
            dataset.test_states if n_test is None else dataset.test_states[: int(n_test)]
        )
    return tuning_train, tuning_dev, official_test


def _evaluate_candidate(
    candidate: dict[str, Any],
    train_states: np.ndarray,
    eval_states: np.ndarray,
    limits: dict[str, Any],
    *,
    random_state: int,
) -> dict[str, Any]:
    if candidate["family"] == "decoder_audit":
        return evaluate_decoder_candidate(
            candidate,
            train_states,
            eval_states,
            limits,
            random_state=random_state,
        )
    if candidate["family"] == "hybrid_sensor_conditioned":
        return evaluate_sensor_conditioned_latent_candidate(
            candidate,
            train_states,
            eval_states,
            limits,
            random_state=random_state,
        )
    raise ValueError(f"Unknown candidate family: {candidate['family']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "fast", "full"), default="smoke")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional debugging cap. Do not use for final selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_experiment_plan(args.mode)
    limits = plan["limits"]
    tuning_train, tuning_dev, official_test = _load_protocol_data(limits)

    candidates = plan["decoder_candidates"] + plan["hybrid_candidates"]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    results = []
    for candidate in candidates:
        print(f"Running {candidate['candidate']} ...", flush=True)
        result = _evaluate_candidate(
            candidate,
            tuning_train,
            tuning_dev,
            limits,
            random_state=args.random_state,
        )
        results.append(result)
        print(
            (
                f"  dev_one_step_r2={result['dev_one_step_r2']:.4f} "
                f"rmse={result['dev_rmse']:.4f} "
                f"inference_ms={result['inference_ms_per_step']:.3f}"
            ),
            flush=True,
        )

    selected = select_best_result(results)
    final_test_result = None
    if (
        limits["evaluate_official_test"]
        and official_test is not None
        and args.max_candidates is None
    ):
        dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
        all_train = dataset.train_states[: int(limits["n_train_trajectories"])]
        final_limits = dict(limits)
        final_limits["max_eval_segments"] = None
        final_test_result = _evaluate_candidate(
            selected,
            all_train,
            official_test,
            final_limits,
            random_state=args.random_state,
        )

    payload = {
        "stage": "tbmd_qr_cs_next_step_decoder_and_hybrid_sweep",
        "mode": args.mode,
        "protocol": (
            "Train/dev selection uses only train trajectories. Official test is evaluated "
            "only in full mode for the dev-selected candidate; smoke/fast do not touch official test."
        ),
        "train_shape": list(tuning_train.shape),
        "dev_shape": list(tuning_dev.shape),
        "official_test_shape": None if official_test is None else list(official_test.shape),
        "plan": plan,
        "results": results,
        "selected_by_dev": selected,
        "final_test_result": final_test_result,
        "notes": {
            "decoder_audit": (
                "oracle_projection_forecast is causal full-history least squares on the "
                "window dictionary; sensor_recovery_only compares QR-sensor coefficients "
                "against that causal upper bound."
            ),
            "hybrid_sensor_conditioned": (
                "SVD latent ridge forecast is compared with the same latent forecast "
                "conditioned on QR history measurements. No ADMM loop is used at inference."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2)
    report_path = _write_markdown_report(args.output, payload)
    print(f"Saved summary to {args.output}", flush=True)
    print(f"Saved report to {report_path}", flush=True)
    print(f"Selected by dev: {selected['candidate']}", flush=True)


if __name__ == "__main__":
    main()
