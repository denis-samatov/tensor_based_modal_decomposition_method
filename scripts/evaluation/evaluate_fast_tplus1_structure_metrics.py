#!/usr/bin/env python3
"""Evaluate structure-aware t+1 diagnostics for a saved Fast TBMD+QR+CS predictor."""

from __future__ import annotations

import argparse
import csv
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
from TBMD.experiments.navier_stokes_structure_metrics import (
    aggregate_structure_rows,
    compute_structure_metrics,
    radial_power_spectrum,
)

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage5_fast_tplus1_structure_metrics"
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


def select_representative_frames(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select best, median, and worst frames by structure score."""
    if not rows:
        raise ValueError("rows must not be empty")
    sorted_rows = sorted(rows, key=lambda row: float(row["structure_score"]))
    return {
        "best": dict(sorted_rows[0]),
        "median": dict(sorted_rows[len(sorted_rows) // 2]),
        "worst": dict(sorted_rows[-1]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_one_step_predictions(
    model: FastWindowedTBMDQRCSForecaster,
    states: np.ndarray,
    *,
    max_starts: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Collect batched one-step predictions and frame references."""
    states = np.asarray(states, dtype=np.float64)
    history_length = int(model.config.history_length)
    n_starts = states.shape[1] - history_length
    if n_starts <= 0:
        raise ValueError("states do not contain enough timesteps for this predictor")
    if max_starts is not None:
        n_starts = min(n_starts, int(max_starts))
    predictions = []
    targets = []
    refs: list[dict[str, int]] = []
    for start in range(n_starts):
        history = states[:, start : start + history_length]
        pred = model.predict_next(history)
        target = states[:, start + history_length]
        predictions.append(pred)
        targets.append(target)
        refs.extend(
            {
                "trajectory_index": int(traj_idx),
                "start_index": int(start),
                "target_index": int(start + history_length),
            }
            for traj_idx in range(states.shape[0])
        )
    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0), refs


def compute_per_frame_rows(
    target_frames: np.ndarray,
    pred_frames: np.ndarray,
    refs: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Compute structure metrics for each prediction frame."""
    if target_frames.shape != pred_frames.shape or target_frames.shape[0] != len(refs):
        raise ValueError("targets, predictions, and refs must have matching lengths")
    rows = []
    for frame_idx, ref in enumerate(refs):
        metrics = compute_structure_metrics(
            target_frames[frame_idx : frame_idx + 1],
            pred_frames[frame_idx : frame_idx + 1],
        )
        rows.append({"flat_index": int(frame_idx), **ref, **metrics})
    return rows


def plot_structure_frame(
    *,
    target: np.ndarray,
    pred: np.ndarray,
    row: dict[str, Any],
    label: str,
    output_path: Path,
) -> None:
    """Save a true/pred/error/spectrum diagnostic plot for one frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error = pred - target
    vmin = float(min(np.min(target), np.min(pred)))
    vmax = float(max(np.max(target), np.max(pred)))
    err_abs = float(max(np.max(np.abs(error)), 1e-12))
    target_spectrum = radial_power_spectrum(target[None])
    pred_spectrum = radial_power_spectrum(pred[None])

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    panels = [
        ("True t+1", target, "viridis", vmin, vmax),
        ("Predicted t+1", pred, "viridis", vmin, vmax),
        ("Error", error, "coolwarm", -err_abs, err_abs),
    ]
    for axis, (title, frame, cmap, frame_vmin, frame_vmax) in zip(axes[:3], panels):
        image = axis.imshow(frame, cmap=cmap, vmin=frame_vmin, vmax=frame_vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[3].plot(target_spectrum, label="true", linewidth=2)
    axes[3].plot(pred_spectrum, label="pred", linewidth=2)
    axes[3].set_yscale("log")
    axes[3].set_title("Radial spectrum")
    axes[3].set_xlabel("radius bin")
    axes[3].legend()
    fig.suptitle(
        "{label}: traj={traj}, start={start}, R2={r2:.4f}, score={score:.4f}, "
        "gradRel={grad:.4f}, specRel={spec:.4f}".format(
            label=label,
            traj=row["trajectory_index"],
            start=row["start_index"],
            r2=row["r2"],
            score=row["structure_score"],
            grad=row["gradient_rel_frob_err"],
            spec=row["radial_spectrum_rel_err"],
        )
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictor", type=Path, required=True, help="Saved Fast TBMD+QR+CS .npz predictor."
    )
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--max-starts", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = FastWindowedTBMDQRCSForecaster.load(args.predictor)
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    states = dataset.test_states if args.split == "test" else dataset.train_states
    states = np.asarray(states[: args.n_trajectories], dtype=np.float64)

    start_time = time.perf_counter()
    targets, predictions, refs = collect_one_step_predictions(
        model,
        states,
        max_starts=args.max_starts,
    )
    rows = compute_per_frame_rows(targets, predictions, refs)
    aggregate = aggregate_structure_rows(rows)
    global_metrics = compute_structure_metrics(targets, predictions)
    representatives = select_representative_frames(rows)
    elapsed = time.perf_counter() - start_time

    figure_paths: dict[str, str] = {}
    if not args.no_plots:
        for label, row in representatives.items():
            frame_idx = int(row["flat_index"])
            path = (
                args.output_dir
                / f"{label}_structure_traj{row['trajectory_index']:03d}_start{row['start_index']:02d}.png"
            )
            plot_structure_frame(
                target=targets[frame_idx],
                pred=predictions[frame_idx],
                row=row,
                label=label,
                output_path=path,
            )
            figure_paths[label] = str(path)

    if args.save_arrays:
        np.savez_compressed(
            args.output_dir / "structure_metric_arrays.npz",
            targets=targets.astype(np.float32),
            predictions=predictions.astype(np.float32),
            errors=(predictions - targets).astype(np.float32),
        )

    payload = {
        "stage": "stage5_fast_tplus1_structure_metrics",
        "protocol": "Structure diagnostics only; no hyperparameter selection is performed.",
        "predictor_path": str(args.predictor),
        "split": args.split,
        "states_shape": list(states.shape),
        "n_frames": len(rows),
        "elapsed_seconds": float(elapsed),
        "global_metrics": global_metrics,
        "aggregate_per_frame": aggregate,
        "representative_frames": representatives,
        "figure_paths": figure_paths,
        "config": model.get_config(),
    }
    json_path = args.output_dir / "structure_metrics_summary.json"
    csv_path = args.output_dir / "structure_metrics_per_frame.csv"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    print(f"Saved structure summary to {json_path}")
    print(f"Saved per-frame metrics to {csv_path}")
    print(
        "Global: R2={r2:.4f}, RMSE={rmse:.4f}, gradRel={grad:.4f}, "
        "specRel={spec:.4f}, score={score:.4f}".format(
            r2=global_metrics["r2"],
            rmse=global_metrics["rmse"],
            grad=global_metrics["gradient_rel_frob_err"],
            spec=global_metrics["radial_spectrum_rel_err"],
            score=global_metrics["structure_score"],
        )
    )


if __name__ == "__main__":
    main()
