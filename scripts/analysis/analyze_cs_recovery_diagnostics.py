#!/usr/bin/env python3
"""Diagnose QR/CS coefficient recovery independently from temporal forecasting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import load_navier_stokes_trajectory_dataset
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareCSForecaster,
    _compute_regression_metrics,
)

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "cs_recovery_diagnostics_summary.json"
)


def _compute_sparsity_diagnostics(coeffs: np.ndarray, thresholds=(1e-3, 1e-2)) -> dict[str, object]:
    vectors = np.asarray(coeffs, dtype=np.float64).reshape(-1, coeffs.shape[-1])
    abs_vectors = np.abs(vectors)
    l1 = np.sum(abs_vectors, axis=1)
    l2 = np.linalg.norm(vectors, axis=1)
    safe_l2 = np.where(l2 < 1e-12, 1.0, l2)
    sorted_energy = np.sort(vectors * vectors, axis=1)[:, ::-1]
    total_energy = np.sum(sorted_energy, axis=1)
    safe_energy = np.where(total_energy < 1e-12, 1.0, total_energy)
    topk = {}
    for k in (3, 5, 10, 15):
        if vectors.shape[1] >= k:
            topk[f"top_{k}_energy_fraction_mean"] = float(
                np.mean(np.sum(sorted_energy[:, :k], axis=1) / safe_energy)
            )

    return {
        "mean_l1_over_l2": float(np.mean(l1 / safe_l2)),
        "mean_participation_ratio": float(np.mean((l1 * l1) / (safe_l2 * safe_l2))),
        "mean_abs_coeff": float(np.mean(abs_vectors)),
        "median_abs_coeff": float(np.median(abs_vectors)),
        **{
            f"mean_count_abs_gt_{threshold:g}": float(
                np.mean(np.sum(abs_vectors > threshold, axis=1))
            )
            for threshold in thresholds
        },
        **topk,
    }


def _sensor_matrix_diagnostics(sensor_dictionary: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(sensor_dictionary, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.maximum(column_norms, 1e-12)
    gram = np.abs(normalized.T @ normalized)
    if gram.shape[0] > 1:
        coherence = float(np.max(gram - np.eye(gram.shape[0])))
    else:
        coherence = 0.0
    return {
        "shape": list(matrix.shape),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": float(np.linalg.cond(matrix)),
        "min_singular_value": float(np.min(singular_values)),
        "max_singular_value": float(np.max(singular_values)),
        "column_coherence": coherence,
    }


def _first_difference_gram(n_steps: int) -> np.ndarray:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_steps == 1:
        return np.zeros((1, 1), dtype=np.float64)

    difference = np.zeros((n_steps - 1, n_steps), dtype=np.float64)
    rows = np.arange(n_steps - 1)
    difference[rows, rows] = -1.0
    difference[rows, rows + 1] = 1.0
    return difference.T @ difference


def _solve_temporal_regularized_lstsq(
    measurements: np.ndarray,
    sensor_dictionary: np.ndarray,
    *,
    temporal_weight: float,
    ridge_weight: float,
) -> np.ndarray:
    """
    Solve `||A Theta.T - Y||_F^2 + lt ||D A||_F^2 + lr ||A||_F^2`.

    `Y` has shape `(T, p)`, `Theta` has shape `(p, r)`, and the returned
    coefficient trajectory `A` has shape `(T, r)`.
    """
    data = np.asarray(measurements, dtype=np.float64)
    theta = np.asarray(sensor_dictionary, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("measurements must have shape `(T, n_sensors)`")
    if theta.ndim != 2:
        raise ValueError("sensor_dictionary must have shape `(n_sensors, rank)`")
    if data.shape[1] != theta.shape[0]:
        raise ValueError("measurement and sensor dictionary dimensions do not match")
    if temporal_weight < 0.0 or ridge_weight < 0.0:
        raise ValueError("regularization weights must be non-negative")

    gram_space = theta.T @ theta
    rhs = data @ theta
    gram_time = _first_difference_gram(data.shape[0])

    time_evals, time_vecs = np.linalg.eigh(gram_time)
    space_evals, space_vecs = np.linalg.eigh(gram_space)
    transformed_rhs = time_vecs.T @ rhs @ space_vecs
    denominator = (
        temporal_weight * time_evals[:, np.newaxis] + space_evals[np.newaxis, :] + ridge_weight
    )
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    transformed_solution = transformed_rhs / denominator
    return time_vecs @ transformed_solution @ space_vecs.T


def _recover_causal_temporal_regularized_coefficients(
    centered_states: np.ndarray,
    sensor_indices: np.ndarray,
    sensor_dictionary: np.ndarray,
    *,
    window_length: int,
    temporal_weight: float,
    ridge_weight: float,
) -> np.ndarray:
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    states = np.asarray(centered_states, dtype=np.float64)
    flat = states.reshape(states.shape[0], states.shape[1], -1)
    measurements = flat[:, :, sensor_indices]
    rank = sensor_dictionary.shape[1]
    coeffs = np.zeros((states.shape[0], states.shape[1], rank), dtype=np.float64)

    for traj_idx in range(measurements.shape[0]):
        for step_idx in range(measurements.shape[1]):
            start = max(0, step_idx - window_length + 1)
            window_coeffs = _solve_temporal_regularized_lstsq(
                measurements[traj_idx, start : step_idx + 1],
                sensor_dictionary,
                temporal_weight=temporal_weight,
                ridge_weight=ridge_weight,
            )
            coeffs[traj_idx, step_idx] = window_coeffs[-1]

    return coeffs


def _coefficient_error_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    metrics = _compute_regression_metrics(reference, estimate)
    return {
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "rel_frob_err": metrics["rel_frob_err"],
        "r2": metrics["r2"],
    }


def _centered_reconstruction_metrics(
    centered_states: np.ndarray,
    coeffs: np.ndarray,
    basis_vectors: np.ndarray,
) -> dict[str, float]:
    target = centered_states.reshape(-1, *centered_states.shape[2:])
    vectors = coeffs.reshape(-1, coeffs.shape[-1])
    recon = (vectors @ basis_vectors).reshape(target.shape)
    metrics = _compute_regression_metrics(target, recon)
    return {
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "rel_frob_err": metrics["rel_frob_err"],
        "r2": metrics["r2"],
    }


def _sensor_equation_residual(
    centered_states: np.ndarray,
    coeffs: np.ndarray,
    sensor_indices: np.ndarray,
    sensor_dictionary: np.ndarray,
) -> dict[str, float]:
    measurements = centered_states.reshape(-1, int(np.prod(centered_states.shape[2:])))[
        :, sensor_indices
    ]
    predicted = coeffs.reshape(-1, coeffs.shape[-1]) @ sensor_dictionary.T
    residual = measurements - predicted
    denom = max(float(np.linalg.norm(measurements)), 1e-12)
    return {
        "sensor_rmse": float(np.sqrt(np.mean(residual * residual))),
        "sensor_rel_frob_err": float(np.linalg.norm(residual) / denom),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=45)
    parser.add_argument("--n-sensors", type=int, default=45)
    parser.add_argument("--n-dictionary-trajectories", type=int, default=160)
    parser.add_argument("--n-probe-trajectories", type=int, default=40)
    parser.add_argument("--cs-max-iter", type=int, default=100)
    parser.add_argument("--cs-tol", type=float, default=1e-4)
    parser.add_argument("--cs-epsilon-l1", type=float, default=1e-3)
    parser.add_argument("--temporal-window-length", type=int, default=7)
    parser.add_argument(
        "--temporal-weights",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 5.0, 10.0],
    )
    parser.add_argument("--temporal-ridge-weight", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    states = np.asarray(dataset.train_states[: args.n_dictionary_trajectories], dtype=np.float64)
    probe_states = states[: args.n_probe_trajectories]

    model = TrajectoryAwareCSForecaster(
        rank=args.rank,
        n_sensors=args.n_sensors,
        coefficient_source="sensor_cs",
        feature_mode="coeff",
        cs_max_iter=args.cs_max_iter,
        cs_tol=args.cs_tol,
        cs_epsilon_l1=args.cs_epsilon_l1,
        lstm_num_epochs=1,
        correction_num_epochs=0,
    )
    model._spatial_shape = states.shape[2:]
    model._spatial_mean = np.mean(states, axis=(0, 1))
    centered = states - model._spatial_mean
    centered_probe = probe_states - model._spatial_mean
    model._fit_dictionary(centered)
    model._place_sensors()
    model._prepare_sensor_lstsq()

    flat_probe = centered_probe.reshape(-1, int(np.prod(centered_probe.shape[2:])))
    oracle_coeffs = (flat_probe @ model._basis_vectors.T).reshape(
        centered_probe.shape[0],
        centered_probe.shape[1],
        model.rank,
    )
    sensor_dictionary = model._dictionary_tensor.reshape(-1, model.rank)[model._sensor_indices]

    source_results = {
        "full_projection": {
            "coefficient_error_vs_projection": _coefficient_error_metrics(
                oracle_coeffs,
                oracle_coeffs,
            ),
            "centered_reconstruction": _centered_reconstruction_metrics(
                centered_probe,
                oracle_coeffs,
                model._basis_vectors,
            ),
            "sparsity": _compute_sparsity_diagnostics(oracle_coeffs),
            "sensor_equation": _sensor_equation_residual(
                centered_probe,
                oracle_coeffs,
                model._sensor_indices,
                sensor_dictionary,
            ),
        }
    }

    for source in ("sensor_lstsq", "sensor_cs"):
        model.coefficient_source = source
        recovered = model._states_to_coefficients(centered_probe)
        result = {
            "coefficient_error_vs_projection": _coefficient_error_metrics(
                oracle_coeffs,
                recovered,
            ),
            "centered_reconstruction": _centered_reconstruction_metrics(
                centered_probe,
                recovered,
                model._basis_vectors,
            ),
            "sparsity": _compute_sparsity_diagnostics(recovered),
            "sensor_equation": _sensor_equation_residual(
                centered_probe,
                recovered,
                model._sensor_indices,
                sensor_dictionary,
            ),
        }
        if source == "sensor_cs":
            result["cs_mean_iterations"] = (
                float(np.mean([m["iterations"] for m in model._last_projection_metrics]))
                if model._last_projection_metrics
                else None
            )
            result["cs_convergence_rate"] = (
                float(np.mean([m["converged"] for m in model._last_projection_metrics]))
                if model._last_projection_metrics
                else None
            )
            result["cs_mean_objective"] = (
                float(np.mean([m["objective"] for m in model._last_projection_metrics]))
                if model._last_projection_metrics
                else None
            )
        source_results[source] = result

    for temporal_weight in args.temporal_weights:
        recovered = _recover_causal_temporal_regularized_coefficients(
            centered_probe,
            model._sensor_indices,
            sensor_dictionary,
            window_length=args.temporal_window_length,
            temporal_weight=float(temporal_weight),
            ridge_weight=args.temporal_ridge_weight,
        )
        source_name = f"sensor_temporal_ridge_lam_t_{temporal_weight:g}"
        source_results[source_name] = {
            "coefficient_error_vs_projection": _coefficient_error_metrics(
                oracle_coeffs,
                recovered,
            ),
            "centered_reconstruction": _centered_reconstruction_metrics(
                centered_probe,
                recovered,
                model._basis_vectors,
            ),
            "sparsity": _compute_sparsity_diagnostics(recovered),
            "sensor_equation": _sensor_equation_residual(
                centered_probe,
                recovered,
                model._sensor_indices,
                sensor_dictionary,
            ),
            "temporal_regularization": {
                "causal": True,
                "window_length": args.temporal_window_length,
                "temporal_weight": float(temporal_weight),
                "ridge_weight": args.temporal_ridge_weight,
                "objective": ("||A Theta.T - Y||_F^2 + lambda_t ||D A||_F^2 + lambda_r ||A||_F^2"),
            },
        }

    payload = {
        "protocol": "Recovery-only diagnostics on train trajectories; no temporal forecaster selection.",
        "config": vars(args) | {"output": str(args.output)},
        "dictionary_shape": list(model._dictionary_tensor.shape),
        "basis_shape": list(model._basis_vectors.shape),
        "sensor_summary": {
            "requested_sensors": model.requested_n_sensors,
            "actual_sensors": model.n_sensors,
            "selection_method": model._sensor_selection_method,
            "indices": model._sensor_indices.astype(int).tolist(),
        },
        "sensor_matrix": _sensor_matrix_diagnostics(sensor_dictionary),
        "sources": source_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved CS recovery diagnostics to {args.output}")


if __name__ == "__main__":
    main()
