#!/usr/bin/env python3
"""Evaluate hybrid SVD-latent + windowed TBMD/QR/CS rollout forecasting."""

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

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments import (
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareResidualCorrectedForecaster,
    _compute_regression_metrics,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_N_TRAIN_TRAJECTORIES,
    DEFAULT_NAVIER_STOKES_RANKS,
)

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
    / "hybrid_tbmd_qr_cs_forecasting_summary.json"
)
TUNING_DEV_SPLIT = 0.2


def _blend_predictions(
    backbone_predictions: np.ndarray,
    sensor_predictions: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Blend stable latent rollout with TBMD/QR/CS sensor forecast."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    backbone = np.asarray(backbone_predictions, dtype=np.float64)
    sensor = np.asarray(sensor_predictions, dtype=np.float64)
    if backbone.shape != sensor.shape:
        raise ValueError("backbone_predictions and sensor_predictions must match")
    return backbone + beta * (sensor - backbone)


def _select_beta_by_dev_rollout(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Select hybrid beta by dev rollout R2 only."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    return max(candidates, key=lambda item: item["dev"]["spatial_r2"])


def _json_safe(value):
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


def _compact_rollout_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"target_spatial", "pred_spatial", "backbone_spatial", "sensor_spatial"}
    }


def _fit_backbone(args: argparse.Namespace) -> TrajectoryAwareResidualCorrectedForecaster:
    config = LatentModalForecasterConfig(
        ranks=list(args.backbone_ranks),
        forecaster_type="lstm",
        verbose=False,
        delta_forecast=False,
        lstm_hidden_size=args.backbone_hidden_size,
        lstm_num_layers=args.backbone_num_layers,
        lstm_seq_length=args.backbone_seq_length,
        lstm_num_epochs=args.backbone_epochs,
        lstm_use_scheduled_sampling=True,
        lstm_ss_unroll_steps=args.backbone_ss_unroll_steps,
        lstm_ss_decay_rate=args.backbone_ss_decay_rate,
    )
    return TrajectoryAwareResidualCorrectedForecaster(
        config=config,
        feature_mode="latent_plus_delta",
        correction_hidden_size=args.correction_hidden_size,
        correction_num_layers=args.correction_num_layers,
        correction_dropout=0.0,
        correction_learning_rate=1e-3,
        correction_weight_decay=1e-5,
        correction_num_epochs=args.correction_epochs,
        correction_batch_size=32,
        correction_val_split=0.2,
        correction_early_stopping_patience=20,
        correction_latent_loss_weight=1.0,
        correction_spatial_loss_weight=0.0,
        correction_rel_frob_loss_weight=0.0,
    )


def _predict_backbone_next_frames(
    backbone: TrajectoryAwareResidualCorrectedForecaster,
    history_states: np.ndarray,
) -> np.ndarray:
    if not backbone._fitted:
        raise RuntimeError("backbone must be fitted")
    states = np.asarray(history_states, dtype=np.float64)
    if states.ndim != 4:
        raise ValueError("history_states must have shape `(B,T,H,W)`")

    seq_length = backbone.config.lstm_seq_length if backbone.config.forecaster_type == "lstm" else 1
    if states.shape[1] < seq_length:
        raise ValueError("history_states has fewer frames than the backbone sequence length")

    base = backbone._base_forecaster
    spatial_shape = states.shape[2:]
    raw_latent = base._project_states_to_latent(states[:, -seq_length:])
    model_latent = base._to_model_latent(raw_latent)

    corrected_raw = []
    for traj_idx in range(model_latent.shape[0]):
        if backbone.config.forecaster_type == "lstm":
            current_input = model_latent[traj_idx]
        else:
            current_input = model_latent[traj_idx, -1]
        _, _, pred_raw = backbone._predict_corrected_next_raw(current_input)
        corrected_raw.append(pred_raw)

    return base._reconstruct_latent_batch(np.asarray(corrected_raw), spatial_shape)


def _predict_sensor_next_frames(
    history_centered: np.ndarray,
    dictionary: np.ndarray,
    spatial_mask: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    recovery_source: str,
    sensor_rcond: float,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
    ridge_corrector: dict[str, object] | None,
    ridge_correction_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if recovery_source == "sensor_lstsq":
        pred, coeffs = windowed._predict_from_history_sensor_lstsq(
            history_centered,
            dictionary,
            spatial_sensor_indices,
            rcond=sensor_rcond,
        )
        cs_metrics = []
    elif recovery_source == "sensor_cs":
        pred, coeffs, cs_metrics = windowed._predict_from_history_sensor_cs(
            history_centered,
            dictionary,
            spatial_mask,
            cs_max_iter=cs_max_iter,
            cs_tol=cs_tol,
            cs_epsilon_l1=cs_epsilon_l1,
        )
    else:
        raise ValueError("recovery_source must be `sensor_lstsq` or `sensor_cs`")

    if ridge_corrector is not None:
        pred = windowed._apply_ridge_residual_corrector(
            pred,
            coeffs,
            ridge_corrector,
            scale=ridge_correction_scale,
        )
    return pred, coeffs, cs_metrics


def _fit_sensor_module(
    train_states: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    spatial_mean = np.mean(train_states, axis=(0, 1))
    centered_train = train_states - spatial_mean
    train_segments = windowed._build_forecast_segment_tensor(
        centered_train,
        history_length=args.sensor_history_length,
        stride=args.segment_stride,
        max_segments=args.max_train_segments,
    )
    dictionary, tbmd_summary = windowed._fit_segment_dictionary(
        train_segments,
        ranks=[
            args.sensor_r_tau,
            args.sensor_r_x,
            args.sensor_r_y,
            args.sensor_r_segment,
        ],
        random_state=args.random_state,
    )
    history_dictionary, _ = windowed._history_and_target(dictionary)
    spatial_mask, spatial_sensor_indices = windowed._place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=args.n_spatial_sensors,
        random_state=args.random_state,
    )

    ridge_corrector = None
    ridge_train_metrics = None
    if args.sensor_ridge_alpha >= 0:
        train_sensor_pred, train_sensor_coeffs = windowed._predict_next_sensor_lstsq(
            train_segments,
            dictionary,
            spatial_sensor_indices,
            rcond=args.sensor_rcond,
        )
        train_targets = windowed._target_frames_from_segments(train_segments)
        ridge_corrector = windowed._fit_ridge_residual_corrector(
            train_targets,
            train_sensor_pred,
            train_sensor_coeffs,
            alpha=args.sensor_ridge_alpha,
        )
        corrected_train = windowed._apply_ridge_residual_corrector(
            train_sensor_pred,
            train_sensor_coeffs,
            ridge_corrector,
            scale=args.sensor_ridge_scale,
        )
        ridge_train_metrics = _compute_regression_metrics(train_targets, corrected_train)

    return {
        "spatial_mean": spatial_mean,
        "train_segments": train_segments,
        "dictionary": dictionary,
        "tbmd_summary": tbmd_summary,
        "spatial_mask": spatial_mask,
        "spatial_sensor_indices": spatial_sensor_indices,
        "ridge_corrector": ridge_corrector,
        "ridge_train_metrics": ridge_train_metrics,
    }


def _evaluate_hybrid_rollout(
    trajectories: np.ndarray,
    *,
    backbone: TrajectoryAwareResidualCorrectedForecaster,
    sensor_module: dict[str, object],
    beta: float,
    recovery_source: str,
    sensor_rcond: float,
    cs_max_iter: int,
    cs_tol: float,
    cs_epsilon_l1: float,
    ridge_correction_scale: float,
) -> dict[str, object]:
    series = np.asarray(trajectories, dtype=np.float64)
    dictionary = np.asarray(sensor_module["dictionary"], dtype=np.float64)
    sensor_mean = np.asarray(sensor_module["spatial_mean"], dtype=np.float64)
    history_length = dictionary.shape[0] - 1
    backbone_warmup = (
        backbone.config.lstm_seq_length if backbone.config.forecaster_type == "lstm" else 1
    )
    warmup = max(history_length, backbone_warmup)
    if series.ndim != 4:
        raise ValueError("trajectories must have shape `(B,T,H,W)`")
    if series.shape[1] <= warmup:
        raise ValueError("trajectory length must exceed rollout warmup")

    history = series[:, :warmup].copy()
    predictions = []
    backbone_predictions = []
    sensor_predictions = []
    targets = []
    cs_metrics = []
    per_step = []

    for step_idx in range(warmup, series.shape[1]):
        backbone_pred = _predict_backbone_next_frames(backbone, history)
        history_centered = history[:, -history_length:] - sensor_mean
        sensor_pred_centered, _, step_cs_metrics = _predict_sensor_next_frames(
            history_centered,
            dictionary,
            np.asarray(sensor_module["spatial_mask"]),
            np.asarray(sensor_module["spatial_sensor_indices"]),
            recovery_source=recovery_source,
            sensor_rcond=sensor_rcond,
            cs_max_iter=cs_max_iter,
            cs_tol=cs_tol,
            cs_epsilon_l1=cs_epsilon_l1,
            ridge_corrector=sensor_module["ridge_corrector"],
            ridge_correction_scale=ridge_correction_scale,
        )
        sensor_pred = sensor_pred_centered + sensor_mean
        hybrid_pred = _blend_predictions(backbone_pred, sensor_pred, beta=beta)
        target = series[:, step_idx]

        predictions.append(hybrid_pred)
        backbone_predictions.append(backbone_pred)
        sensor_predictions.append(sensor_pred)
        targets.append(target)
        cs_metrics.extend(step_cs_metrics)
        per_step.append(_compute_regression_metrics(target, hybrid_pred))
        history = np.concatenate([history, hybrid_pred[:, None]], axis=1)

    pred_spatial = np.concatenate(predictions, axis=0)
    target_spatial = np.concatenate(targets, axis=0)
    backbone_spatial = np.concatenate(backbone_predictions, axis=0)
    sensor_spatial = np.concatenate(sensor_predictions, axis=0)
    spatial_metrics = _compute_regression_metrics(target_spatial, pred_spatial)
    backbone_metrics = _compute_regression_metrics(target_spatial, backbone_spatial)
    sensor_metrics = _compute_regression_metrics(target_spatial, sensor_spatial)
    result = {
        "spatial_mse": spatial_metrics["mse"],
        "spatial_rmse": spatial_metrics["rmse"],
        "spatial_rel_frob_err": spatial_metrics["rel_frob_err"],
        "spatial_r2": spatial_metrics["r2"],
        "backbone_spatial_r2": backbone_metrics["r2"],
        "backbone_spatial_rmse": backbone_metrics["rmse"],
        "sensor_spatial_r2": sensor_metrics["r2"],
        "sensor_spatial_rmse": sensor_metrics["rmse"],
        "beta": float(beta),
        "n_eval_samples": int(target_spatial.shape[0]),
        "n_trajectories": int(series.shape[0]),
        "n_rollout_steps": int(series.shape[1] - warmup),
        "warmup": int(warmup),
        "backbone_warmup": int(backbone_warmup),
        "sensor_history_length": int(history_length),
        "per_step_spatial_r2": [float(item["r2"]) for item in per_step],
        "per_step_spatial_rmse": [float(item["rmse"]) for item in per_step],
        "target_spatial": target_spatial,
        "pred_spatial": pred_spatial,
        "backbone_spatial": backbone_spatial,
        "sensor_spatial": sensor_spatial,
    }
    if cs_metrics:
        result["cs_mean_iterations"] = float(np.mean([m["iterations"] for m in cs_metrics]))
        result["cs_convergence_rate"] = float(np.mean([m["converged"] for m in cs_metrics]))
        result["cs_mean_objective"] = float(np.mean([m["objective"] for m in cs_metrics]))
    return result


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
    parser.add_argument("--n-dev-eval-trajectories", type=int, default=None)
    parser.add_argument("--backbone-ranks", type=int, nargs=3, default=DEFAULT_NAVIER_STOKES_RANKS)
    parser.add_argument("--backbone-seq-length", type=int, default=10)
    parser.add_argument("--backbone-hidden-size", type=int, default=128)
    parser.add_argument("--backbone-num-layers", type=int, default=2)
    parser.add_argument("--backbone-epochs", type=int, default=80)
    parser.add_argument("--backbone-ss-unroll-steps", type=int, default=5)
    parser.add_argument("--backbone-ss-decay-rate", type=float, default=0.01)
    parser.add_argument("--correction-hidden-size", type=int, default=64)
    parser.add_argument("--correction-num-layers", type=int, default=2)
    parser.add_argument("--correction-epochs", type=int, default=60)
    parser.add_argument("--sensor-history-length", type=int, default=7)
    parser.add_argument("--segment-stride", type=int, default=1)
    parser.add_argument("--max-train-segments", type=int, default=1024)
    parser.add_argument("--sensor-r-tau", type=int, default=8)
    parser.add_argument("--sensor-r-x", type=int, default=32)
    parser.add_argument("--sensor-r-y", type=int, default=32)
    parser.add_argument("--sensor-r-segment", type=int, default=120)
    parser.add_argument("--n-spatial-sensors", type=int, default=160)
    parser.add_argument(
        "--sensor-recovery-source", choices=("sensor_cs", "sensor_lstsq"), default="sensor_cs"
    )
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument("--cs-max-iter", type=int, default=100)
    parser.add_argument("--cs-tol", type=float, default=1e-4)
    parser.add_argument("--cs-epsilon-l1", type=float, default=1e-3)
    parser.add_argument(
        "--sensor-ridge-alpha",
        type=float,
        default=1e-8,
        help="Set a negative value to disable the TBMD/QR/CS ridge residual head.",
    )
    parser.add_argument("--sensor-ridge-scale", type=float, default=1.5)
    parser.add_argument(
        "--beta-grid", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train_trajectories > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(f"n_train_trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}")

    all_train_states, official_test_states = _load_data(
        args.n_train_trajectories,
        args.n_test_trajectories,
    )
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=TUNING_DEV_SPLIT,
    )
    dev_eval_states = tuning_dev_states
    if args.n_dev_eval_trajectories is not None:
        dev_eval_states = dev_eval_states[: args.n_dev_eval_trajectories]

    start = time.time()
    backbone = _fit_backbone(args)
    backbone.fit(tuning_train_states)
    backbone_fit_time = time.time() - start

    sensor_start = time.time()
    sensor_module = _fit_sensor_module(tuning_train_states, args)
    sensor_fit_time = time.time() - sensor_start

    dev_candidates = []
    for beta in args.beta_grid:
        dev_result = _evaluate_hybrid_rollout(
            dev_eval_states,
            backbone=backbone,
            sensor_module=sensor_module,
            beta=beta,
            recovery_source=args.sensor_recovery_source,
            sensor_rcond=args.sensor_rcond,
            cs_max_iter=args.cs_max_iter,
            cs_tol=args.cs_tol,
            cs_epsilon_l1=args.cs_epsilon_l1,
            ridge_correction_scale=args.sensor_ridge_scale,
        )
        dev_candidates.append(
            {
                "beta": float(beta),
                "dev": _compact_rollout_result(dev_result),
                "test": None,
            }
        )

    selected = _select_beta_by_dev_rollout(dev_candidates)
    final_test = _evaluate_hybrid_rollout(
        official_test_states,
        backbone=backbone,
        sensor_module=sensor_module,
        beta=selected["beta"],
        recovery_source=args.sensor_recovery_source,
        sensor_rcond=args.sensor_rcond,
        cs_max_iter=args.cs_max_iter,
        cs_tol=args.cs_tol,
        cs_epsilon_l1=args.cs_epsilon_l1,
        ridge_correction_scale=args.sensor_ridge_scale,
    )

    config_payload = vars(args).copy()
    config_payload["output"] = str(config_payload["output"])
    payload = {
        "protocol": (
            "Hybrid strict autoregressive rollout. The SVD/TBMD latent residual-corrected "
            "LSTM is the stable backbone. A windowed TBMD dictionary is fit only on "
            "training history+target segments; QR spatial sensors and CS/ADMM recover "
            "history coefficients, reconstruct a next-step sensor forecast, and beta "
            "blends that forecast into the backbone. Beta is selected on dev rollout "
            "only; official test is evaluated once with the selected beta."
        ),
        "config": config_payload,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "dev_eval_shape": list(dev_eval_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "backbone_fit_time": backbone_fit_time,
        "sensor_fit_time": sensor_fit_time,
        "sensor_summary": {
            "dictionary_shape": list(sensor_module["dictionary"].shape),
            "train_segment_shape": list(sensor_module["train_segments"].shape),
            "tbmd_summary": sensor_module["tbmd_summary"],
            "requested_spatial_sensors": args.n_spatial_sensors,
            "actual_spatial_sensors": int(sensor_module["spatial_mask"].sum()),
            "total_history_measurements_per_prediction": int(
                sensor_module["spatial_mask"].sum() * args.sensor_history_length
            ),
            "sensor_recovery_source": args.sensor_recovery_source,
            "ridge_train_metrics": sensor_module["ridge_train_metrics"],
        },
        "dev_beta_candidates": dev_candidates,
        "selected_beta": selected["beta"],
        "selection_metric": "dev strict rollout spatial_r2",
        "final_test": _compact_rollout_result(final_test),
        "total_time": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2)
    print(f"Saved hybrid TBMD QR/CS forecasting summary to {args.output}")


if __name__ == "__main__":
    main()
