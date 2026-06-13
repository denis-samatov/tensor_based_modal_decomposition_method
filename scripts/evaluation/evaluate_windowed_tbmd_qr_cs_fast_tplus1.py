#!/usr/bin/env python3
"""Fast causal t+1 forecasting with windowed TBMD + QR sparse sensing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import (
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics
from TBMD.experiments.navier_stokes_model_registry import DEFAULT_N_TRAIN_TRAJECTORIES

WINDOWED_SCRIPT = Path(__file__).with_name("evaluate_windowed_tbmd_qr_cs_forecasting.py")
_WINDOWED_SPEC = importlib.util.spec_from_file_location(
    "evaluate_windowed_tbmd_qr_cs_forecasting",
    WINDOWED_SCRIPT,
)
windowed = importlib.util.module_from_spec(_WINDOWED_SPEC)
_WINDOWED_SPEC.loader.exec_module(windowed)

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "windowed_tbmd_qr_cs_fast_tplus1_summary.json"
)
TUNING_DEV_SPLIT = 0.2


def _history_sensor_measurements_from_segments(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
) -> np.ndarray:
    """Return fixed-spatial history measurements shaped `(N_segments, L*n_sensors)`."""
    history_dictionary, _ = windowed._history_and_target(dictionary)
    history = np.asarray(segments, dtype=np.float64)[:-1]
    if history.shape[1:3] != history_dictionary.shape[1:3]:
        raise ValueError("segments and dictionary spatial shapes must match")
    if history.shape[0] != history_dictionary.shape[0]:
        raise ValueError("segments history length must match dictionary")

    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history = history.reshape(history.shape[0], height_width, history.shape[-1])
    measurements = flat_history[:, spatial_sensor_indices, :].reshape(
        -1,
        history.shape[-1],
    )
    return measurements.T


def _fit_standardized_ridge_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    features: np.ndarray,
    *,
    alpha: float,
) -> dict[str, object]:
    """Fit a standardized ridge residual map from arbitrary features to frame error."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    features = np.asarray(features, dtype=np.float64)
    if target_flat.shape != base_flat.shape:
        raise ValueError("target_frames and base_predictions must have matching sample/output shapes")
    if features.shape[0] != target_flat.shape[0]:
        raise ValueError("features and predictions must contain the same number of samples")

    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std = np.where(feature_std < 1e-12, 1.0, feature_std)
    standardized = (features - feature_mean) / feature_std
    design = np.concatenate(
        [standardized, np.ones((standardized.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    residual = target_flat - base_flat
    gram = design.T @ design
    penalty = alpha * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = design.T @ residual
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    return {
        "alpha": float(alpha),
        "weights": weights,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_dim": int(features.shape[1]),
        "output_dim": int(target_flat.shape[1]),
    }


def _apply_standardized_ridge_residual_corrector(
    base_predictions: np.ndarray,
    features: np.ndarray,
    corrector: dict[str, object],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    features = np.asarray(features, dtype=np.float64)
    standardized = (
        features - np.asarray(corrector["feature_mean"], dtype=np.float64)
    ) / np.asarray(corrector["feature_std"], dtype=np.float64)
    design = np.concatenate(
        [standardized, np.ones((standardized.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    correction = design @ np.asarray(corrector["weights"], dtype=np.float64)
    return (base_flat + scale * correction).reshape(base_predictions.shape)


def _predict_fast_next_from_history(
    history_states: np.ndarray,
    model: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict raw `t+1` frames from raw history frames using a fitted fast model."""
    dictionary = np.asarray(model["dictionary"], dtype=np.float64)
    spatial_mean = np.asarray(model["spatial_mean"], dtype=np.float64)
    spatial_sensor_indices = np.asarray(model["spatial_sensor_indices"], dtype=int)
    history = np.asarray(history_states, dtype=np.float64)
    history_length = dictionary.shape[0] - 1
    if history.ndim != 4:
        raise ValueError("history_states must have shape `(B,L,H,W)`")
    if history.shape[1:] != dictionary[:-1].shape[:-1]:
        raise ValueError("history_states must match dictionary history shape")

    centered_history = history - spatial_mean
    pred_centered, coeffs = windowed._predict_from_history_sensor_lstsq(
        centered_history,
        dictionary,
        spatial_sensor_indices,
        rcond=float(model.get("sensor_rcond", 1e-6)),
    )
    corrector = model.get("coefficient_corrector")
    if corrector is not None:
        pred_centered = windowed._apply_ridge_residual_corrector(
            pred_centered,
            coeffs,
            corrector,
            scale=float(model.get("correction_scale", 1.0)),
        )
    return pred_centered + spatial_mean, coeffs


def _save_fast_predictor_npz(
    path: Path,
    *,
    dictionary: np.ndarray,
    spatial_mean: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    sensor_rcond: float,
    correction_scale: float,
    coefficient_corrector: dict[str, object] | None,
) -> None:
    """Save the fitted fast t+1 predictor state for direct inference reuse."""
    arrays = {
        "dictionary": np.asarray(dictionary, dtype=np.float32),
        "spatial_mean": np.asarray(spatial_mean, dtype=np.float32),
        "spatial_sensor_indices": np.asarray(spatial_sensor_indices, dtype=np.int64),
        "sensor_rcond": np.asarray(sensor_rcond, dtype=np.float64),
        "correction_scale": np.asarray(correction_scale, dtype=np.float64),
        "has_coefficient_corrector": np.asarray(coefficient_corrector is not None),
    }
    if coefficient_corrector is not None:
        arrays["coefficient_corrector_weights"] = np.asarray(
            coefficient_corrector["weights"],
            dtype=np.float32,
        )
        arrays["coefficient_corrector_alpha"] = np.asarray(
            coefficient_corrector["alpha"],
            dtype=np.float64,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_fast_predictor_npz(path: Path) -> dict[str, object]:
    """Load a predictor saved by `_save_fast_predictor_npz`."""
    data = np.load(path, allow_pickle=False)
    corrector = None
    if bool(data["has_coefficient_corrector"]):
        corrector = {
            "alpha": float(data["coefficient_corrector_alpha"]),
            "weights": data["coefficient_corrector_weights"].astype(np.float64),
            "feature_dim": int(data["coefficient_corrector_weights"].shape[0] - 1),
            "output_dim": int(data["coefficient_corrector_weights"].shape[1]),
        }
    return {
        "dictionary": data["dictionary"].astype(np.float64),
        "spatial_mean": data["spatial_mean"].astype(np.float64),
        "spatial_sensor_indices": data["spatial_sensor_indices"].astype(int),
        "sensor_rcond": float(data["sensor_rcond"]),
        "correction_scale": float(data["correction_scale"]),
        "coefficient_corrector": corrector,
    }


def _compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "r2": float(metrics["r2"]),
        "rmse": float(metrics["rmse"]),
        "rel_frob_err": float(metrics["rel_frob_err"]),
        "mse": float(metrics["mse"]),
    }


def _evaluate_fast_tplus1(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    sensor_rcond: float,
    coefficient_correctors: dict[str, dict[str, object]],
    measurement_correctors: dict[str, dict[str, object]],
    correction_scale: float,
) -> dict[str, object]:
    target = windowed._target_frames_from_segments(segments)
    start = time.perf_counter()
    base_pred, coeffs = windowed._predict_next_sensor_lstsq(
        segments,
        dictionary,
        spatial_sensor_indices,
        rcond=sensor_rcond,
    )
    base_time = time.perf_counter() - start
    measurement_start = time.perf_counter()
    measurements = _history_sensor_measurements_from_segments(
        segments,
        dictionary,
        spatial_sensor_indices,
    )
    measurement_time = time.perf_counter() - measurement_start
    base_metrics = _compact_metrics(_compute_regression_metrics(target, base_pred))

    result = {
        "base_fixed_sensor_lstsq": {
            **base_metrics,
            "prediction_time": float(base_time),
            "samples_per_second": float(target.shape[0] / max(base_time, 1e-12)),
        },
        "measurement_extraction_time": float(measurement_time),
        "coefficient_ridge": {},
        "measurement_ridge": {},
        "n_eval_samples": int(target.shape[0]),
    }
    for label, corrector in coefficient_correctors.items():
        pred_start = time.perf_counter()
        pred = windowed._apply_ridge_residual_corrector(
            base_pred,
            coeffs,
            corrector,
            scale=correction_scale,
        )
        elapsed = time.perf_counter() - pred_start
        result["coefficient_ridge"][label] = {
            **_compact_metrics(_compute_regression_metrics(target, pred)),
            "prediction_time": float(base_time + elapsed),
            "samples_per_second": float(target.shape[0] / max(base_time + elapsed, 1e-12)),
        }
    for label, corrector in measurement_correctors.items():
        pred_start = time.perf_counter()
        pred = _apply_standardized_ridge_residual_corrector(
            base_pred,
            measurements,
            corrector,
            scale=correction_scale,
        )
        elapsed = time.perf_counter() - pred_start
        result["measurement_ridge"][label] = {
            **_compact_metrics(_compute_regression_metrics(target, pred)),
            "prediction_time": float(base_time + measurement_time + elapsed),
            "samples_per_second": float(
                target.shape[0] / max(base_time + measurement_time + elapsed, 1e-12)
            ),
        }
    return result


def _select_best_fast_candidate(result: dict[str, object]) -> dict[str, object]:
    candidates = [
        {
            "family": "base_fixed_sensor_lstsq",
            "label": "base",
            **result["base_fixed_sensor_lstsq"],
        }
    ]
    for family in ("coefficient_ridge", "measurement_ridge"):
        for label, metrics in result[family].items():
            candidates.append({"family": family, "label": label, **metrics})
    return max(candidates, key=lambda item: item["r2"])


def _load_data(n_train_trajectories: int, n_test_trajectories: int | None):
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_states = dataset.train_states[:n_train_trajectories]
    test_states = dataset.test_states
    if n_test_trajectories is not None:
        test_states = test_states[:n_test_trajectories]
    return train_states, test_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train-trajectories", type=int, default=640)
    parser.add_argument("--n-test-trajectories", type=int, default=200)
    parser.add_argument("--history-length", type=int, default=7)
    parser.add_argument("--segment-stride", type=int, default=1)
    parser.add_argument("--max-train-segments", type=int, default=4096)
    parser.add_argument("--max-dev-segments", type=int, default=None)
    parser.add_argument("--max-test-segments", type=int, default=None)
    parser.add_argument("--r-tau", type=int, default=8)
    parser.add_argument("--r-x", type=int, default=32)
    parser.add_argument("--r-y", type=int, default=32)
    parser.add_argument("--r-segment", type=int, default=300)
    parser.add_argument("--n-spatial-sensors", type=int, default=300)
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument("--coefficient-ridge-alphas", type=float, nargs="*", default=[1e-8])
    parser.add_argument(
        "--measurement-ridge-alphas",
        type=float,
        nargs="*",
        default=[1e-2, 1.0, 100.0],
    )
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional .npz path for the fitted fast t+1 predictor.",
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

    start = time.perf_counter()
    train_segments = windowed._build_forecast_segment_tensor(
        centered_train,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_train_segments,
    )
    dictionary, tbmd_summary = windowed._fit_segment_dictionary(
        train_segments,
        ranks=[args.r_tau, args.r_x, args.r_y, args.r_segment],
        random_state=args.random_state,
    )
    history_dictionary, _ = windowed._history_and_target(dictionary)
    spatial_mask, spatial_sensor_indices = windowed._place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=args.n_spatial_sensors,
        random_state=args.random_state,
    )
    fit_dictionary_time = time.perf_counter() - start

    train_pred, train_coeffs = windowed._predict_next_sensor_lstsq(
        train_segments,
        dictionary,
        spatial_sensor_indices,
        rcond=args.sensor_rcond,
    )
    train_targets = windowed._target_frames_from_segments(train_segments)
    train_measurements = _history_sensor_measurements_from_segments(
        train_segments,
        dictionary,
        spatial_sensor_indices,
    )

    correction_start = time.perf_counter()
    coefficient_correctors = {}
    coefficient_train_metrics = {}
    for alpha in args.coefficient_ridge_alphas:
        label = f"alpha_{alpha:g}"
        corrector = windowed._fit_ridge_residual_corrector(
            train_targets,
            train_pred,
            train_coeffs,
            alpha=alpha,
        )
        coefficient_correctors[label] = corrector
        corrected = windowed._apply_ridge_residual_corrector(
            train_pred,
            train_coeffs,
            corrector,
            scale=args.correction_scale,
        )
        coefficient_train_metrics[label] = _compact_metrics(
            _compute_regression_metrics(train_targets, corrected)
        )

    measurement_correctors = {}
    measurement_train_metrics = {}
    for alpha in args.measurement_ridge_alphas:
        label = f"alpha_{alpha:g}"
        corrector = _fit_standardized_ridge_residual_corrector(
            train_targets,
            train_pred,
            train_measurements,
            alpha=alpha,
        )
        measurement_correctors[label] = corrector
        corrected = _apply_standardized_ridge_residual_corrector(
            train_pred,
            train_measurements,
            corrector,
            scale=args.correction_scale,
        )
        measurement_train_metrics[label] = _compact_metrics(
            _compute_regression_metrics(train_targets, corrected)
        )
    correction_fit_time = time.perf_counter() - correction_start

    dev_segments = windowed._build_forecast_segment_tensor(
        centered_dev,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_dev_segments,
    )
    dev_result = _evaluate_fast_tplus1(
        dev_segments,
        dictionary,
        spatial_sensor_indices,
        sensor_rcond=args.sensor_rcond,
        coefficient_correctors=coefficient_correctors,
        measurement_correctors=measurement_correctors,
        correction_scale=args.correction_scale,
    )
    selected = _select_best_fast_candidate(dev_result)

    test_segments = windowed._build_forecast_segment_tensor(
        centered_test,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_test_segments,
    )
    selected_coefficient_correctors = {}
    selected_measurement_correctors = {}
    if selected["family"] == "coefficient_ridge":
        selected_coefficient_correctors[selected["label"]] = coefficient_correctors[selected["label"]]
    elif selected["family"] == "measurement_ridge":
        selected_measurement_correctors[selected["label"]] = measurement_correctors[selected["label"]]
    test_result = _evaluate_fast_tplus1(
        test_segments,
        dictionary,
        spatial_sensor_indices,
        sensor_rcond=args.sensor_rcond,
        coefficient_correctors=selected_coefficient_correctors,
        measurement_correctors=selected_measurement_correctors,
        correction_scale=args.correction_scale,
    )

    if selected["family"] == "base_fixed_sensor_lstsq":
        selected_test = {
            "family": selected["family"],
            "label": selected["label"],
            **test_result["base_fixed_sensor_lstsq"],
        }
    else:
        selected_test = {
            "family": selected["family"],
            "label": selected["label"],
            **test_result[selected["family"]][selected["label"]],
        }

    saved_model_path = None
    if args.save_model is not None:
        selected_coefficient_corrector = (
            coefficient_correctors[selected["label"]]
            if selected["family"] == "coefficient_ridge"
            else None
        )
        _save_fast_predictor_npz(
            args.save_model,
            dictionary=dictionary,
            spatial_mean=spatial_mean,
            spatial_sensor_indices=spatial_sensor_indices,
            sensor_rcond=args.sensor_rcond,
            correction_scale=args.correction_scale,
            coefficient_corrector=selected_coefficient_corrector,
        )
        saved_model_path = str(args.save_model)

    config_payload = vars(args).copy()
    config_payload["output"] = str(config_payload["output"])
    if config_payload["save_model"] is not None:
        config_payload["save_model"] = str(config_payload["save_model"])
    payload = {
        "protocol": (
            "Fast causal t+1 only. The windowed TBMD/HOSVD dictionary and QR fixed "
            "spatial sensors are fit on train windows. Online prediction uses only "
            "history-frame sparse sensor measurements and a precomputed gappy/linear "
            "recovery map; no ADMM iterations are required for the fast path."
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
            "total_history_measurements_per_prediction": int(spatial_mask.sum() * args.history_length),
            "sensor_indices": spatial_sensor_indices.astype(int).tolist(),
        },
        "fast_predictor_summary": {
            "measurement_dim": int(spatial_sensor_indices.shape[0] * args.history_length),
            "latent_dim": int(dictionary.shape[-1]),
            "selected_family": selected["family"],
            "selected_label": selected["label"],
            "selected_head_parameters": (
                int((dictionary.shape[-1] + 1) * np.prod(dictionary.shape[1:3]))
                if selected["family"] == "coefficient_ridge"
                else None
            ),
            "saved_model_path": saved_model_path,
        },
        "fit_dictionary_time": float(fit_dictionary_time),
        "correction_fit_time": float(correction_fit_time),
        "coefficient_train_metrics": coefficient_train_metrics,
        "measurement_train_metrics": measurement_train_metrics,
        "dev_result": dev_result,
        "selected_by_dev": selected,
        "final_test_result": test_result,
        "selected_test_result": selected_test,
        "total_time": float(time.perf_counter() - start),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved fast windowed TBMD QR/CS t+1 summary to {args.output}")


if __name__ == "__main__":
    main()
