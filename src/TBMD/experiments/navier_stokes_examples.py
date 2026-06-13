"""
Qualitative example helpers for Navier-Stokes experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def split_train_dev_trajectories(
    train_states: np.ndarray,
    *,
    dev_split: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic trajectory-level tuning split."""

    states = np.asarray(train_states)
    if states.ndim < 1:
        raise ValueError("train_states must include a trajectory axis")
    if not 0.0 < dev_split < 1.0:
        raise ValueError("dev_split must be in (0, 1)")
    if states.shape[0] < 2:
        raise ValueError("Need at least two trajectories to create a tuning holdout")

    dev_count = max(1, int(round(states.shape[0] * dev_split)))
    if dev_count >= states.shape[0]:
        raise ValueError("dev_split leaves no training trajectories")

    split_idx = states.shape[0] - dev_count
    return states[:split_idx], states[split_idx:]


def _select_evenly_spaced_indices(total_count: int, count: int) -> list[int]:
    if total_count <= 0:
        return []
    if count <= 0:
        raise ValueError("count must be positive")

    clamped = min(count, total_count)
    indices = np.linspace(0, total_count - 1, num=clamped, dtype=int).tolist()
    deduped = list(dict.fromkeys(indices))
    return deduped


def select_fixed_trajectory_indices(total_trajectories: int, count: int = 2) -> list[int]:
    """Pick deterministic official-test trajectories for qualitative examples."""

    return _select_evenly_spaced_indices(total_trajectories, count)


def select_fixed_rollout_steps(total_steps: int, count: int = 4) -> list[int]:
    """Pick deterministic rollout steps for contact sheets and comparisons."""

    return _select_evenly_spaced_indices(total_steps, count)


def make_frame_filename(step_index: int) -> str:
    """Return a deterministic frame filename for a rollout step."""

    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    return f"frame_t{step_index:02d}.png"


def build_examples_manifest(
    *,
    output_root: str,
    trajectory_indices: list[int],
    rollout_steps: list[int],
    image_settings: dict[str, Any],
    per_model: list[dict[str, Any]],
    comparison_artifacts: list[str],
) -> dict[str, Any]:
    """Build a machine-readable manifest for generated example artifacts."""

    return {
        "output_root": output_root,
        "trajectory_indices": list(trajectory_indices),
        "rollout_steps": list(rollout_steps),
        "image_settings": dict(image_settings),
        "models": list(per_model),
        "comparison_artifacts": list(comparison_artifacts),
    }


def compute_spatial_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute rollout metrics in spatial space."""

    target_arr = np.asarray(target, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    if target_arr.shape != pred_arr.shape:
        raise ValueError("target and pred must have identical shapes")

    diff = pred_arr - target_arr
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_diff))
    max_abs_err = float(np.max(abs_diff))
    bias = float(np.mean(diff))
    rel_frob = float(np.linalg.norm(diff) / max(np.linalg.norm(target_arr), 1e-12))

    target_2d = target_arr.reshape(target_arr.shape[0], -1)
    pred_2d = pred_arr.reshape(pred_arr.shape[0], -1)
    ss_res = float(np.sum((target_2d - pred_2d) ** 2))
    ss_tot = float(np.sum((target_2d - np.mean(target_2d, axis=0)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))

    diff_2d = diff.reshape(diff.shape[0], -1)
    target_norms = np.linalg.norm(target_2d, axis=1)
    diff_norms = np.linalg.norm(diff_2d, axis=1)
    per_sample_rmse = np.sqrt(np.mean(diff_2d**2, axis=1))
    per_sample_rel_frob = diff_norms / np.maximum(target_norms, 1e-12)

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "max_abs_err": max_abs_err,
        "bias": bias,
        "rel_frob": rel_frob,
        "per_sample_rmse": per_sample_rmse.tolist(),
        "per_sample_rel_frob": per_sample_rel_frob.tolist(),
    }


def extract_common_horizon_predictions(
    eval_result: dict[str, Any],
    test_states: np.ndarray,
    common_warmup_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align flattened rollout output to a shared common warmup horizon."""

    states = np.asarray(test_states, dtype=np.float64)
    n_trajectories, total_steps = states.shape[:2]
    spatial_shape = states.shape[2:]
    raw_eval_steps = int(eval_result["n_eval_samples"]) // n_trajectories
    raw_warmup_steps = total_steps - raw_eval_steps
    drop_steps = max(common_warmup_steps - raw_warmup_steps, 0)

    target = np.asarray(eval_result["target_spatial"], dtype=np.float64).reshape(
        n_trajectories, raw_eval_steps, *spatial_shape
    )
    pred = np.asarray(eval_result["pred_spatial"], dtype=np.float64).reshape(
        n_trajectories, raw_eval_steps, *spatial_shape
    )

    if drop_steps >= raw_eval_steps:
        raise ValueError(
            f"Cannot compare with common_warmup_steps={common_warmup_steps}; "
            f"model only provides {raw_eval_steps} evaluated steps"
        )

    return target[:, drop_steps:, :, :], pred[:, drop_steps:, :, :]


def compute_common_horizon_metrics(
    eval_result: dict[str, Any],
    test_states: np.ndarray,
    common_warmup_steps: int,
) -> dict[str, float | int]:
    """Compute spatial metrics on a fair common-warmup rollout horizon."""

    target_common, pred_common = extract_common_horizon_predictions(
        eval_result,
        test_states,
        common_warmup_steps,
    )
    metrics = compute_spatial_metrics(
        target_common.reshape(-1, *target_common.shape[2:]),
        pred_common.reshape(-1, *pred_common.shape[2:]),
    )
    metrics.update(
        {
            "warmup_steps": int(common_warmup_steps),
            "n_eval_steps_per_trajectory": int(target_common.shape[1]),
        }
    )
    return metrics


def compute_common_horizon_diagnostics(
    eval_result: dict[str, Any],
    test_states: np.ndarray,
    common_warmup_steps: int,
    *,
    worst_count: int = 5,
) -> dict[str, Any]:
    """Compute time- and trajectory-resolved errors on the common horizon."""

    if worst_count <= 0:
        raise ValueError("worst_count must be positive")

    target_common, pred_common = extract_common_horizon_predictions(
        eval_result,
        test_states,
        common_warmup_steps,
    )
    diff = np.asarray(pred_common - target_common, dtype=np.float64)
    abs_diff = np.abs(diff)
    n_trajectories, n_steps = diff.shape[:2]
    flat_per_step = diff.reshape(n_trajectories, n_steps, -1)
    abs_flat_per_step = abs_diff.reshape(n_trajectories, n_steps, -1)
    target_flat_per_traj = target_common.reshape(n_trajectories, -1)
    diff_flat_per_traj = diff.reshape(n_trajectories, -1)

    per_step_rmse = np.sqrt(np.mean(flat_per_step**2, axis=(0, 2)))
    per_step_mae = np.mean(abs_flat_per_step, axis=(0, 2))
    per_step_max_abs_err = np.max(abs_flat_per_step, axis=(0, 2))
    per_trajectory_rmse = np.sqrt(np.mean(diff_flat_per_traj**2, axis=1))
    per_trajectory_mae = np.mean(np.abs(diff_flat_per_traj), axis=1)
    per_trajectory_rel_frob = np.linalg.norm(diff_flat_per_traj, axis=1) / np.maximum(
        np.linalg.norm(target_flat_per_traj, axis=1), 1e-12
    )

    worst_order = np.argsort(per_trajectory_rmse)[::-1]
    worst_indices = worst_order[: min(worst_count, len(worst_order))]

    return {
        "warmup_steps": int(common_warmup_steps),
        "n_trajectories": int(n_trajectories),
        "n_eval_steps_per_trajectory": int(n_steps),
        "per_step_rmse": per_step_rmse.tolist(),
        "per_step_mae": per_step_mae.tolist(),
        "per_step_max_abs_err": per_step_max_abs_err.tolist(),
        "per_trajectory_rmse": per_trajectory_rmse.tolist(),
        "per_trajectory_mae": per_trajectory_mae.tolist(),
        "per_trajectory_rel_frob": per_trajectory_rel_frob.tolist(),
        "worst_trajectory_indices": worst_indices.astype(int).tolist(),
        "worst_trajectory_rmse": per_trajectory_rmse[worst_indices].tolist(),
    }


def _render_triptych_figure(
    *,
    target_frame: np.ndarray,
    pred_frame: np.ndarray,
    title: str,
    dpi: int = 150,
) -> np.ndarray:
    frame_target = np.asarray(target_frame, dtype=np.float64)
    frame_pred = np.asarray(pred_frame, dtype=np.float64)
    frame_err = np.abs(frame_target - frame_pred)
    vmin = min(np.min(frame_target), np.min(frame_pred))
    vmax = max(np.max(frame_target), np.max(frame_pred))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    fig.suptitle(title, fontsize=14)
    row = axes[0]

    im0 = row[0].imshow(frame_target, cmap="jet", vmin=vmin, vmax=vmax)
    row[0].set_title("Ground Truth")
    row[0].axis("off")
    fig.colorbar(im0, ax=row[0], fraction=0.046, pad=0.04)

    im1 = row[1].imshow(frame_pred, cmap="jet", vmin=vmin, vmax=vmax)
    row[1].set_title("Prediction")
    row[1].axis("off")
    fig.colorbar(im1, ax=row[1], fraction=0.046, pad=0.04)

    im2 = row[2].imshow(frame_err, cmap="hot")
    row[2].set_title(f"Absolute Error (Max: {np.max(frame_err):.4f})")
    row[2].axis("off")
    fig.colorbar(im2, ax=row[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def save_rollout_frame(
    *,
    target_frame: np.ndarray,
    pred_frame: np.ndarray,
    step_index: int,
    model_name: str,
    save_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Save a single triptych rollout frame."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _render_triptych_figure(
        target_frame=target_frame,
        pred_frame=pred_frame,
        title=f"{model_name} rollout t={step_index}",
        dpi=dpi,
    )
    imageio.imwrite(output_path, image)
    return output_path


def save_rollout_gif(
    *,
    target_frames: np.ndarray,
    pred_frames: np.ndarray,
    step_indices: list[int],
    model_name: str,
    save_path: str | Path,
    fps: int = 2,
    dpi: int = 120,
) -> Path:
    """Save a rollout GIF using triptych frames."""

    if fps <= 0:
        raise ValueError("fps must be positive")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for idx, step_idx in enumerate(step_indices):
        gif_frames.append(
            _render_triptych_figure(
                target_frame=target_frames[idx],
                pred_frame=pred_frames[idx],
                title=f"{model_name} rollout t={step_idx}",
                dpi=dpi,
            )
        )
    imageio.mimsave(output_path, gif_frames, fps=fps)
    return output_path


def save_contact_sheet(
    *,
    target_frames: np.ndarray,
    pred_frames: np.ndarray,
    step_indices: list[int],
    title: str,
    save_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Render a compact target/prediction/error contact sheet."""

    target = np.asarray(target_frames, dtype=np.float64)
    pred = np.asarray(pred_frames, dtype=np.float64)
    if target.shape != pred.shape:
        raise ValueError("target_frames and pred_frames must have identical shapes")
    if target.ndim != 3:
        raise ValueError("Expected frame arrays with shape `(N, H, W)`")
    if target.shape[0] != len(step_indices):
        raise ValueError("step_indices length must match the number of frames")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = target.shape[0]
    fig, axes = plt.subplots(3, frame_count, figsize=(4 * frame_count, 8), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for col_idx, step_idx in enumerate(step_indices):
        frame_target = target[col_idx]
        frame_pred = pred[col_idx]
        frame_err = np.abs(frame_target - frame_pred)
        vmin = min(np.min(frame_target), np.min(frame_pred))
        vmax = max(np.max(frame_target), np.max(frame_pred))

        im0 = axes[0, col_idx].imshow(frame_target, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0, col_idx].set_title(f"GT t={step_idx}")
        axes[0, col_idx].axis("off")
        fig.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

        im1 = axes[1, col_idx].imshow(frame_pred, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1, col_idx].set_title(f"Pred t={step_idx}")
        axes[1, col_idx].axis("off")
        fig.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

        im2 = axes[2, col_idx].imshow(frame_err, cmap="hot")
        axes[2, col_idx].set_title(f"|Err| t={step_idx}")
        axes[2, col_idx].axis("off")
        fig.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def save_t_plus_one_diagnostics_sheet(
    *,
    target_frames: np.ndarray,
    baseline_frames: np.ndarray,
    corrected_frames: np.ndarray,
    step_indices: list[int],
    title: str,
    save_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Render a fixed-step diagnostic sheet for one-step correction quality."""

    target = np.asarray(target_frames, dtype=np.float64)
    baseline = np.asarray(baseline_frames, dtype=np.float64)
    corrected = np.asarray(corrected_frames, dtype=np.float64)
    if target.shape != baseline.shape or target.shape != corrected.shape:
        raise ValueError(
            "target_frames, baseline_frames, and corrected_frames must have identical shapes"
        )
    if target.ndim != 3:
        raise ValueError("Expected frame arrays with shape `(N, H, W)`")
    if target.shape[0] != len(step_indices):
        raise ValueError("step_indices length must match the number of frames")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = target.shape[0]
    rows = [
        ("Ground Truth", target),
        ("Baseline", baseline),
        ("Corrected", corrected),
        ("|Err| Baseline", np.abs(target - baseline)),
        ("|Err| Corrected", np.abs(target - corrected)),
    ]
    signal_min = min(np.min(target), np.min(baseline), np.min(corrected))
    signal_max = max(np.max(target), np.max(baseline), np.max(corrected))
    error_max = max(np.max(rows[3][1]), np.max(rows[4][1]), 1e-12)

    fig, axes = plt.subplots(len(rows), frame_count, figsize=(4 * frame_count, 11), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for row_idx, (label, frames) in enumerate(rows):
        for col_idx, step_idx in enumerate(step_indices):
            ax = axes[row_idx, col_idx]
            if row_idx < 3:
                ax.imshow(frames[col_idx], cmap="jet", vmin=signal_min, vmax=signal_max)
            else:
                ax.imshow(frames[col_idx], cmap="hot", vmin=0.0, vmax=error_max)
            if row_idx == 0:
                ax.set_title(f"t={step_idx}")
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(label, rotation=90, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def save_comparison_sheet(
    *,
    target_frames: np.ndarray,
    model_frames: list[tuple[str, np.ndarray]],
    step_indices: list[int],
    trajectory_index: int,
    save_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Render a multi-model comparison sheet for one fixed trajectory."""

    target = np.asarray(target_frames, dtype=np.float64)
    if target.ndim != 3:
        raise ValueError("Expected `target_frames` with shape `(N, H, W)`")
    if target.shape[0] != len(step_indices):
        raise ValueError("step_indices length must match target frame count")
    if not model_frames:
        raise ValueError("model_frames must not be empty")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_labels = ["Ground Truth"] + [name for name, _ in model_frames]
    n_rows = len(row_labels)
    n_cols = len(step_indices)
    all_frames = [target] + [np.asarray(frames, dtype=np.float64) for _, frames in model_frames]
    global_min = min(np.min(frames) for frames in all_frames)
    global_max = max(np.max(frames) for frames in all_frames)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f"Fixed Test Trajectory {trajectory_index}", fontsize=16)

    for col_idx, step_idx in enumerate(step_indices):
        gt_ax = axes[0, col_idx]
        im = gt_ax.imshow(target[col_idx], cmap="jet", vmin=global_min, vmax=global_max)
        gt_ax.set_title(f"t={step_idx}")
        gt_ax.axis("off")
        if col_idx == n_cols - 1:
            fig.colorbar(im, ax=gt_ax, fraction=0.046, pad=0.04)

        for row_idx, (name, frames) in enumerate(model_frames, start=1):
            ax = axes[row_idx, col_idx]
            ax.imshow(frames[col_idx], cmap="jet", vmin=global_min, vmax=global_max)
            ax.axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, rotation=90, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


__all__ = [
    "compute_common_horizon_diagnostics",
    "compute_common_horizon_metrics",
    "compute_spatial_metrics",
    "extract_common_horizon_predictions",
    "split_train_dev_trajectories",
    "build_examples_manifest",
    "make_frame_filename",
    "save_comparison_sheet",
    "save_contact_sheet",
    "save_t_plus_one_diagnostics_sheet",
    "save_rollout_frame",
    "save_rollout_gif",
    "select_fixed_rollout_steps",
    "select_fixed_trajectory_indices",
]
