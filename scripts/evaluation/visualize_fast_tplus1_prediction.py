#!/usr/bin/env python3
"""Visualize one Fast TBMD+QR+CS t+1 prediction against an official test frame."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import load_navier_stokes_trajectory_dataset
from TBMD.experiments.navier_stokes_fast_tplus1 import FastWindowedTBMDQRCSForecaster
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_N_TRAIN_TRAJECTORIES,
    get_fast_tplus1_model_specs,
)


DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "stage5_fast_tplus1_visuals"
)


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


def compute_frame_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute scalar metrics for one target/predicted frame pair."""
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if target.shape != pred.shape:
        raise ValueError("target and pred must have the same shape")
    diff = pred - target
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - np.sum(diff**2) / denom) if denom > 1e-12 else float("nan")
    rel_frob = float(np.linalg.norm(diff) / max(np.linalg.norm(target), 1e-12))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rel_frob_err": rel_frob,
    }


def load_or_fit_predictor(args: argparse.Namespace) -> tuple[FastWindowedTBMDQRCSForecaster, dict[str, Any]]:
    """Load a saved predictor, or explicitly fit one from a registry slug."""
    if args.predictor is not None:
        return FastWindowedTBMDQRCSForecaster.load(args.predictor), {
            "predictor_path": str(args.predictor),
            "fit_performed": False,
        }
    if args.fit_slug is None:
        raise ValueError("Provide --predictor or explicitly request --fit-slug")

    specs = {spec.slug: spec for spec in get_fast_tplus1_model_specs()}
    if args.fit_slug not in specs:
        raise ValueError(f"Unknown fast t+1 registry slug: {args.fit_slug}")
    if args.n_train_trajectories > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(f"--n-train-trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}")

    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_states = dataset.train_states[: args.n_train_trajectories]
    model = specs[args.fit_slug].factory()
    model.config.random_state = args.random_state
    fit_start = time.perf_counter()
    model.fit(train_states)
    fit_seconds = time.perf_counter() - fit_start
    predictor_path = args.output_dir / f"{args.fit_slug}_visual_refit_predictor.npz"
    model.save(predictor_path)
    return model, {
        "predictor_path": str(predictor_path),
        "fit_performed": True,
        "fit_slug": args.fit_slug,
        "n_train_trajectories": int(train_states.shape[0]),
        "fit_seconds": float(fit_seconds),
    }


def plot_tplus1_comparison(
    *,
    history_last: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    """Save a true/prediction/error comparison figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [history_last, target, pred]
    vmin = float(min(np.min(field) for field in fields))
    vmax = float(max(np.max(field) for field in fields))
    error = pred - target
    err_abs = float(max(np.max(np.abs(error)), 1e-12))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    panels = [
        ("History t", history_last, "viridis", vmin, vmax),
        ("True t+1", target, "viridis", vmin, vmax),
        ("Predicted t+1", pred, "viridis", vmin, vmax),
        ("Error pred-true", error, "coolwarm", -err_abs, err_abs),
    ]
    for axis, (panel_title, frame, cmap, frame_vmin, frame_vmax) in zip(axes, panels):
        image = axis.imshow(frame, cmap=cmap, vmin=frame_vmin, vmax=frame_vmax)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{title} | R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, relF={metrics['rel_frob_err']:.4f}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", type=Path, default=None, help="Saved .npz predictor path.")
    parser.add_argument(
        "--fit-slug",
        default=None,
        help="Explicitly fit a registry slug if --predictor is not supplied.",
    )
    parser.add_argument("--n-train-trajectories", type=int, default=64)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-arrays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, predictor_meta = load_or_fit_predictor(args)
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    test_states = np.asarray(dataset.test_states, dtype=np.float64)
    history_length = int(model.config.history_length)
    if not 0 <= args.trajectory_index < test_states.shape[0]:
        raise ValueError("--trajectory-index is outside the test split")
    if args.start_index < 0 or args.start_index + history_length >= test_states.shape[1]:
        raise ValueError("--start-index leaves no t+1 target for the fitted history length")

    history = test_states[
        args.trajectory_index : args.trajectory_index + 1,
        args.start_index : args.start_index + history_length,
    ]
    target = test_states[
        args.trajectory_index,
        args.start_index + history_length,
    ]
    infer_start = time.perf_counter()
    pred = model.predict_next(history)[0]
    infer_seconds = time.perf_counter() - infer_start
    metrics = compute_frame_metrics(target, pred)

    stem = f"traj{args.trajectory_index:03d}_start{args.start_index:02d}"
    png_path = args.output_dir / f"{stem}_tplus1_comparison.png"
    json_path = args.output_dir / f"{stem}_tplus1_metrics.json"
    plot_tplus1_comparison(
        history_last=history[0, -1],
        target=target,
        pred=pred,
        metrics=metrics,
        output_path=png_path,
        title="Fast TBMD+QR+CS t+1",
    )
    payload = {
        "stage": "stage5_fast_tplus1_visual_diagnostic",
        "protocol": "Prediction visualization only; no hyperparameter selection is performed.",
        "trajectory_index": int(args.trajectory_index),
        "start_index": int(args.start_index),
        "target_index": int(args.start_index + history_length),
        "history_length": history_length,
        "frame_metrics": metrics,
        "inference_seconds": float(infer_seconds),
        "figure_path": str(png_path),
        "predictor": predictor_meta,
        "config": model.get_config(),
    }
    if args.save_arrays:
        arrays_path = args.output_dir / f"{stem}_tplus1_arrays.npz"
        np.savez_compressed(
            arrays_path,
            history=history.astype(np.float32),
            target=target.astype(np.float32),
            prediction=pred.astype(np.float32),
            error=(pred - target).astype(np.float32),
        )
        payload["arrays_path"] = str(arrays_path)
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"Saved t+1 comparison figure to {png_path}")
    print(f"Saved t+1 metrics to {json_path}")
    print(
        "Frame metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, relF={rel_frob_err:.4f}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
