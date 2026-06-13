#!/usr/bin/env python3
"""Diagnose windowed TBMD + QR + CS recovery for Navier-Stokes trajectories."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
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
from TBMD.experiments import load_navier_stokes_trajectory_dataset
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "windowed_tbmd_qr_cs_diagnostics_summary.json"
)


def _build_window_tensor(
    states: np.ndarray,
    *,
    window_length: int,
    stride: int,
    max_windows: int | None,
) -> np.ndarray:
    """Build causal trajectory windows as `(L, H, W, N_windows)`."""
    series = np.asarray(states, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("states must have shape `(B, T, H, W)`")
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if series.shape[1] < window_length:
        raise ValueError("window_length exceeds trajectory length")

    windows = []
    for trajectory in series:
        for start in range(0, series.shape[1] - window_length + 1, stride):
            windows.append(trajectory[start : start + window_length])
            if max_windows is not None and len(windows) >= max_windows:
                return np.stack(windows, axis=-1)

    return np.stack(windows, axis=-1)


def _compute_window_dictionary_from_tucker(
    core: np.ndarray | torch.Tensor,
    factors: list[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """Convert Tucker decomposition of `(L,H,W,N)` windows to `(L,H,W,Rn)` modes."""
    core_np = _as_numpy(core)
    factor_np = [_as_numpy(factor) for factor in factors]
    if core_np.ndim != 4 or len(factor_np) != 4:
        raise ValueError("Expected a 4D core and four Tucker factors for `(L,H,W,N)`")

    u_tau, u_x, u_y, _ = factor_np
    return np.einsum("ta,xb,yc,abcq->txyq", u_tau, u_x, u_y, core_np, optimize=True)


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fit_windowed_tbmd_dictionary(
    window_tensor: np.ndarray,
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
    decomposer = TuckerDecomposerInterface(window_tensor.astype(np.float32), config=config)
    decomposer.decompose()
    decomposer.reconstruct()
    dictionary = _compute_window_dictionary_from_tucker(
        decomposer.core_tensor,
        decomposer.factors,
    )
    return dictionary.astype(np.float64), {
        "core_shape": list(_as_numpy(decomposer.core_tensor).shape),
        "factor_shapes": [list(_as_numpy(factor).shape) for factor in decomposer.factors],
        "reconstruction_rel_frob": float(decomposer.reconstruction_errors),
    }


def _place_window_sensors(
    dictionary: np.ndarray,
    *,
    n_sensors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    config = SensorPlacementConfig(
        n_sensors=min(n_sensors, dictionary.shape[-1]),
        random_state=random_state,
        verbose=False,
        dtype="float64",
    )
    qr = TensorTubeQRDecomposition(
        dictionary,
        N=min(n_sensors, dictionary.shape[-1]),
        config=config,
        dtype=torch.float64,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        mask, _, _ = qr.factorize()
    sensor_mask = mask.detach().cpu().numpy().astype(bool)
    if n_sensors > int(sensor_mask.sum()):
        sensor_mask = _augment_window_sensor_mask_by_leverage(
            dictionary,
            sensor_mask,
            target_sensors=n_sensors,
        )
    return sensor_mask, np.flatnonzero(sensor_mask.reshape(-1))


def _augment_window_sensor_mask_by_leverage(
    dictionary: np.ndarray,
    sensor_mask: np.ndarray,
    *,
    target_sensors: int,
) -> np.ndarray:
    flat_mask = sensor_mask.reshape(-1).copy()
    n_extra = min(target_sensors - int(flat_mask.sum()), flat_mask.size - int(flat_mask.sum()))
    if n_extra <= 0:
        return flat_mask.reshape(sensor_mask.shape)

    leverage = np.sum(dictionary.reshape(-1, dictionary.shape[-1]) ** 2, axis=1)
    leverage[flat_mask] = -np.inf
    extra_indices = np.argpartition(-leverage, n_extra - 1)[:n_extra]
    extra_indices = extra_indices[np.argsort(-leverage[extra_indices])]
    flat_mask[extra_indices] = True
    return flat_mask.reshape(sensor_mask.shape)


def _coefficient_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    metrics = _compute_regression_metrics(reference, estimate)
    return {
        "rmse": metrics["rmse"],
        "rel_frob_err": metrics["rel_frob_err"],
        "r2": metrics["r2"],
    }


def _window_reconstruction_metrics(
    windows: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, object]:
    full = _compute_regression_metrics(
        np.moveaxis(windows, -1, 0),
        np.moveaxis(reconstructed, -1, 0),
    )
    last = _compute_regression_metrics(
        windows[-1].transpose(2, 0, 1),
        reconstructed[-1].transpose(2, 0, 1),
    )
    return {
        "window_rel_frob_err": full["rel_frob_err"],
        "window_rmse": full["rmse"],
        "window_r2": full["r2"],
        "last_frame_rel_frob_err": last["rel_frob_err"],
        "last_frame_rmse": last["rmse"],
        "last_frame_r2": last["r2"],
    }


def _recover_window_lstsq(
    windows: np.ndarray,
    dictionary: np.ndarray,
    sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> np.ndarray:
    sensor_dictionary = dictionary.reshape(-1, dictionary.shape[-1])[sensor_indices]
    measurements = windows.reshape(-1, windows.shape[-1])[sensor_indices].T
    pinv = np.linalg.pinv(sensor_dictionary, rcond=rcond)
    return measurements @ pinv.T


def _recover_window_cs(
    windows: np.ndarray,
    dictionary: np.ndarray,
    sensor_mask: np.ndarray,
    *,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    coeffs = np.zeros((windows.shape[-1], dictionary.shape[-1]), dtype=np.float64)
    metrics_out = []
    core_cfg = CompressiveSensingConfig(
        max_iter=cs_max_iter,
        tol=cs_tol,
        epsilon_l1=cs_epsilon_l1,
        device="cpu",
        dtype=torch.float32,
    )
    ext_cfg = ExtensionCompressiveSensingConfig(solver="cholesky", collect_history=False)
    for idx in range(windows.shape[-1]):
        solver = TensorCompressiveSensing(
            dictionary.astype(np.float32),
            sensor_mask,
            windows[..., idx].astype(np.float32),
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
    return coeffs, metrics_out


def _reconstruct_windows(dictionary: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    flat = coeffs @ dictionary.reshape(-1, dictionary.shape[-1]).T
    return flat.T.reshape(*dictionary.shape[:-1], coeffs.shape[0])


def _sensor_equation_metrics(
    windows: np.ndarray,
    dictionary: np.ndarray,
    coeffs: np.ndarray,
    sensor_indices: np.ndarray,
) -> dict[str, float]:
    measurements = windows.reshape(-1, windows.shape[-1])[sensor_indices].T
    sensor_dictionary = dictionary.reshape(-1, dictionary.shape[-1])[sensor_indices]
    predicted = coeffs @ sensor_dictionary.T
    residual = measurements - predicted
    return {
        "sensor_rmse": float(np.sqrt(np.mean(residual * residual))),
        "sensor_rel_frob_err": float(
            np.linalg.norm(residual) / max(np.linalg.norm(measurements), 1e-12)
        ),
    }


def _source_result(
    windows: np.ndarray,
    dictionary: np.ndarray,
    coeffs: np.ndarray,
    reference_coeffs: np.ndarray,
    sensor_indices: np.ndarray,
) -> dict[str, object]:
    reconstructed = _reconstruct_windows(dictionary, coeffs)
    return {
        "coefficient_error_vs_projection": _coefficient_metrics(reference_coeffs, coeffs),
        "reconstruction": _window_reconstruction_metrics(windows, reconstructed),
        "sensor_equation": _sensor_equation_metrics(
            windows,
            dictionary,
            coeffs,
            sensor_indices,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dictionary-trajectories", type=int, default=80)
    parser.add_argument("--n-probe-trajectories", type=int, default=20)
    parser.add_argument("--window-length", type=int, default=7)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-dictionary-windows", type=int, default=512)
    parser.add_argument("--max-probe-windows", type=int, default=256)
    parser.add_argument("--r-tau", type=int, default=5)
    parser.add_argument("--r-x", type=int, default=32)
    parser.add_argument("--r-y", type=int, default=32)
    parser.add_argument("--r-window", type=int, default=45)
    parser.add_argument("--n-sensors", type=int, default=60)
    parser.add_argument("--cs-max-iter", type=int, default=100)
    parser.add_argument("--cs-tol", type=float, default=1e-4)
    parser.add_argument("--cs-epsilon-l1", type=float, default=1e-3)
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_states = np.asarray(
        dataset.train_states[: args.n_dictionary_trajectories],
        dtype=np.float64,
    )
    spatial_mean = np.mean(train_states, axis=(0, 1))
    centered_train = train_states - spatial_mean
    centered_probe_states = (
        np.asarray(dataset.train_states[: args.n_probe_trajectories], dtype=np.float64)
        - spatial_mean
    )

    dictionary_windows = _build_window_tensor(
        centered_train,
        window_length=args.window_length,
        stride=args.window_stride,
        max_windows=args.max_dictionary_windows,
    )
    probe_windows = _build_window_tensor(
        centered_probe_states,
        window_length=args.window_length,
        stride=args.window_stride,
        max_windows=args.max_probe_windows,
    )

    ranks = [args.r_tau, args.r_x, args.r_y, args.r_window]
    dictionary, tbmd_summary = _fit_windowed_tbmd_dictionary(
        dictionary_windows,
        ranks=ranks,
        random_state=args.random_state,
    )
    sensor_mask, sensor_indices = _place_window_sensors(
        dictionary,
        n_sensors=args.n_sensors,
        random_state=args.random_state,
    )

    flat_dictionary = dictionary.reshape(-1, dictionary.shape[-1])
    reference_coeffs = (
        probe_windows.reshape(-1, probe_windows.shape[-1]).T
        @ np.linalg.pinv(flat_dictionary, rcond=args.sensor_rcond).T
    )

    source_results = {
        "full_window_projection": _source_result(
            probe_windows,
            dictionary,
            reference_coeffs,
            reference_coeffs,
            sensor_indices,
        )
    }

    lstsq_coeffs = _recover_window_lstsq(
        probe_windows,
        dictionary,
        sensor_indices,
        rcond=args.sensor_rcond,
    )
    source_results["window_sensor_lstsq"] = _source_result(
        probe_windows,
        dictionary,
        lstsq_coeffs,
        reference_coeffs,
        sensor_indices,
    )

    cs_coeffs, cs_metrics = _recover_window_cs(
        probe_windows,
        dictionary,
        sensor_mask,
        cs_max_iter=args.cs_max_iter,
        cs_tol=args.cs_tol,
        cs_epsilon_l1=args.cs_epsilon_l1,
    )
    source_results["window_sensor_cs"] = _source_result(
        probe_windows,
        dictionary,
        cs_coeffs,
        reference_coeffs,
        sensor_indices,
    )
    source_results["window_sensor_cs"]["cs_mean_iterations"] = float(
        np.mean([m["iterations"] for m in cs_metrics])
    )
    source_results["window_sensor_cs"]["cs_convergence_rate"] = float(
        np.mean([m["converged"] for m in cs_metrics])
    )
    source_results["window_sensor_cs"]["cs_mean_objective"] = float(
        np.mean([m["objective"] for m in cs_metrics])
    )

    payload = {
        "protocol": (
            "Windowed TBMD recovery diagnostics. Tucker/HOSVD dictionary is fit "
            "only on train trajectory windows; probe windows are causal fixed-length "
            "history windows. No forecasting model is trained in this script."
        ),
        "config": vars(args) | {"output": str(args.output)},
        "train_shape": list(train_states.shape),
        "dictionary_window_shape": list(dictionary_windows.shape),
        "probe_window_shape": list(probe_windows.shape),
        "tbmd_summary": tbmd_summary,
        "dictionary_shape": list(dictionary.shape),
        "sensor_summary": {
            "requested_sensors": args.n_sensors,
            "actual_sensors": int(sensor_mask.sum()),
            "selection_method": "qr" if args.n_sensors <= args.r_window else "qr_plus_leverage",
            "sensor_indices": sensor_indices.astype(int).tolist(),
        },
        "sources": source_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved windowed TBMD QR/CS diagnostics to {args.output}")


if __name__ == "__main__":
    main()
