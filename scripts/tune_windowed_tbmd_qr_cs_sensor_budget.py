#!/usr/bin/env python3
"""Sweep fixed-spatial sensor budgets for windowed TBMD + QR + CS next-step forecasting."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "windowed_tbmd_qr_cs_sensor_budget_sweep_summary.json"
)

_EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_windowed_tbmd_qr_cs_forecasting",
    SCRIPT_DIR / "evaluate_windowed_tbmd_qr_cs_forecasting.py",
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError("Could not load evaluate_windowed_tbmd_qr_cs_forecasting.py")
eval_windowed = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(eval_windowed)


def _selected_ridge_metrics(
    result: dict[str, object],
    selected_label: str,
) -> dict[str, float]:
    corrected = result["ridge_corrected"][selected_label]["fixed_sensor_cs"]
    base = result["fixed_sensor_cs"]
    return {
        "base_fixed_sensor_cs_r2": float(base["spatial_r2"]),
        "base_fixed_sensor_cs_rmse": float(base["spatial_rmse"]),
        "base_fixed_sensor_cs_rel_frob": float(base["spatial_rel_frob_err"]),
        "corrected_fixed_sensor_cs_r2": float(corrected["r2"]),
        "corrected_fixed_sensor_cs_rmse": float(corrected["rmse"]),
        "corrected_fixed_sensor_cs_rel_frob": float(corrected["rel_frob_err"]),
    }


def _select_best_budget_result(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("No sensor-budget results to select from")
    return max(
        results,
        key=lambda result: result["selected_ridge"]["dev_fixed_sensor_cs_r2"],
    )


def _select_practical_budget_result(
    results: list[dict[str, object]],
    *,
    tolerance: float,
) -> dict[str, object]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    best = _select_best_budget_result(results)
    threshold = best["selected_ridge"]["dev_fixed_sensor_cs_r2"] - tolerance
    candidates = [
        result
        for result in results
        if result["selected_ridge"]["dev_fixed_sensor_cs_r2"] >= threshold
    ]
    return min(candidates, key=lambda result: result["n_spatial_sensors"])


def _fit_ridge_correctors(
    train_segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    sensor_rcond: float,
    alphas: list[float],
) -> dict[str, dict[str, object]]:
    train_sensor_pred, train_sensor_coeffs = eval_windowed._predict_next_sensor_lstsq(
        train_segments,
        dictionary,
        spatial_sensor_indices,
        rcond=sensor_rcond,
    )
    train_targets = eval_windowed._target_frames_from_segments(train_segments)
    correctors = {}
    for alpha in alphas:
        label = f"alpha_{alpha:g}"
        correctors[label] = eval_windowed._fit_ridge_residual_corrector(
            train_targets,
            train_sensor_pred,
            train_sensor_coeffs,
            alpha=alpha,
        )
    return correctors


def _evaluate_sensor_budget(
    *,
    n_spatial_sensors: int,
    train_segments: np.ndarray,
    dev_segments: np.ndarray,
    test_segments: np.ndarray | None,
    dictionary: np.ndarray,
    history_dictionary: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    spatial_mask, spatial_sensor_indices = eval_windowed._place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=n_spatial_sensors,
        random_state=args.random_state,
    )
    ridge_correctors = _fit_ridge_correctors(
        train_segments,
        dictionary,
        spatial_sensor_indices,
        sensor_rcond=args.sensor_rcond,
        alphas=args.ridge_correction_alphas,
    )
    dev_result = eval_windowed._evaluate_segments(
        dev_segments,
        dictionary,
        spatial_mask,
        spatial_sensor_indices,
        sensor_rcond=args.sensor_rcond,
        cs_max_iter=args.cs_max_iter,
        cs_tol=args.cs_tol,
        cs_epsilon_l1=args.cs_epsilon_l1,
        ridge_correctors=ridge_correctors,
    )
    selected_label = max(
        dev_result["ridge_corrected"],
        key=lambda label: dev_result["ridge_corrected"][label]["fixed_sensor_cs"]["r2"],
    )
    selected_alpha = ridge_correctors[selected_label]["alpha"]

    test_result = None
    selected_test_metrics = {}
    if test_segments is not None:
        test_result = eval_windowed._evaluate_segments(
            test_segments,
            dictionary,
            spatial_mask,
            spatial_sensor_indices,
            sensor_rcond=args.sensor_rcond,
            cs_max_iter=args.cs_max_iter,
            cs_tol=args.cs_tol,
            cs_epsilon_l1=args.cs_epsilon_l1,
            ridge_correctors={selected_label: ridge_correctors[selected_label]},
        )
        test_metrics = _selected_ridge_metrics(test_result, selected_label)
        selected_test_metrics = {
            "test_fixed_sensor_cs_r2": test_metrics["corrected_fixed_sensor_cs_r2"],
            "test_fixed_sensor_cs_rmse": test_metrics["corrected_fixed_sensor_cs_rmse"],
            "test_fixed_sensor_cs_rel_frob": test_metrics[
                "corrected_fixed_sensor_cs_rel_frob"
            ],
            "test_base_fixed_sensor_cs_r2": test_metrics["base_fixed_sensor_cs_r2"],
        }

    dev_metrics = _selected_ridge_metrics(dev_result, selected_label)
    return {
        "n_spatial_sensors": int(n_spatial_sensors),
        "actual_spatial_sensors": int(spatial_mask.sum()),
        "total_history_measurements_per_prediction": int(
            spatial_mask.sum() * args.history_length
        ),
        "selected_ridge": {
            "selected_label": selected_label,
            "selected_alpha": float(selected_alpha),
            "dev_fixed_sensor_cs_r2": dev_metrics["corrected_fixed_sensor_cs_r2"],
            "dev_fixed_sensor_cs_rmse": dev_metrics["corrected_fixed_sensor_cs_rmse"],
            "dev_fixed_sensor_cs_rel_frob": dev_metrics[
                "corrected_fixed_sensor_cs_rel_frob"
            ],
            "dev_base_fixed_sensor_cs_r2": dev_metrics["base_fixed_sensor_cs_r2"],
            **selected_test_metrics,
        },
        "dev_result": dev_result,
        "final_test_result": test_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train-trajectories", type=int, default=1000)
    parser.add_argument("--n-test-trajectories", type=int, default=200)
    parser.add_argument("--history-length", type=int, default=7)
    parser.add_argument("--segment-stride", type=int, default=1)
    parser.add_argument("--max-train-segments", type=int, default=6144)
    parser.add_argument("--max-dev-segments", type=int, default=None)
    parser.add_argument("--max-test-segments", type=int, default=None)
    parser.add_argument("--r-tau", type=int, default=8)
    parser.add_argument("--r-x", type=int, default=32)
    parser.add_argument("--r-y", type=int, default=32)
    parser.add_argument("--r-segment", type=int, default=250)
    parser.add_argument("--n-spatial-sensors-grid", type=int, nargs="+", required=True)
    parser.add_argument(
        "--ridge-correction-alphas",
        type=float,
        nargs="+",
        default=[1e-8, 1e-6, 1e-4],
    )
    parser.add_argument("--practical-tolerance", type=float, default=0.01)
    parser.add_argument("--cs-max-iter", type=int, default=50)
    parser.add_argument("--cs-tol", type=float, default=1e-4)
    parser.add_argument("--cs-epsilon-l1", type=float, default=1e-3)
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_train_states, official_test_states = eval_windowed._load_data(
        args.n_train_trajectories,
        args.n_test_trajectories,
    )
    tuning_train_states, tuning_dev_states = eval_windowed.split_train_dev_trajectories(
        all_train_states,
        dev_split=eval_windowed.TUNING_DEV_SPLIT,
    )
    spatial_mean = np.mean(tuning_train_states, axis=(0, 1))
    centered_train = tuning_train_states - spatial_mean
    centered_dev = tuning_dev_states - spatial_mean
    centered_test = official_test_states - spatial_mean

    train_segments = eval_windowed._build_forecast_segment_tensor(
        centered_train,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_train_segments,
    )
    dev_segments = eval_windowed._build_forecast_segment_tensor(
        centered_dev,
        history_length=args.history_length,
        stride=args.segment_stride,
        max_segments=args.max_dev_segments,
    )
    test_segments = None
    if not args.skip_official_test:
        test_segments = eval_windowed._build_forecast_segment_tensor(
            centered_test,
            history_length=args.history_length,
            stride=args.segment_stride,
            max_segments=args.max_test_segments,
        )

    dictionary, tbmd_summary = eval_windowed._fit_segment_dictionary(
        train_segments,
        ranks=[args.r_tau, args.r_x, args.r_y, args.r_segment],
        random_state=args.random_state,
    )
    history_dictionary, _ = eval_windowed._history_and_target(dictionary)

    results = []
    for n_spatial_sensors in args.n_spatial_sensors_grid:
        results.append(
            _evaluate_sensor_budget(
                n_spatial_sensors=n_spatial_sensors,
                train_segments=train_segments,
                dev_segments=dev_segments,
                test_segments=test_segments,
                dictionary=dictionary,
                history_dictionary=history_dictionary,
                args=args,
            )
        )

    best_result = _select_best_budget_result(results)
    practical_result = _select_practical_budget_result(
        results,
        tolerance=args.practical_tolerance,
    )
    config_payload = vars(args).copy()
    config_payload["output"] = str(config_payload["output"])
    payload = {
        "protocol": (
            "Fit one windowed TBMD dictionary, then sweep fixed spatial QR "
            "sensor budgets for causal next-step QR/CS recovery with ridge "
            "residual correction selected on the train/dev split."
        ),
        "config": config_payload,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "train_segment_shape": list(train_segments.shape),
        "dictionary_shape": list(dictionary.shape),
        "tbmd_summary": tbmd_summary,
        "results": results,
        "best_result": best_result,
        "practical_result": practical_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved windowed TBMD QR/CS sensor-budget sweep to {args.output}")


if __name__ == "__main__":
    main()
