#!/usr/bin/env python3
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import (
    build_examples_manifest,
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    extract_common_horizon_predictions,
    get_navier_stokes_model_specs,
    load_navier_stokes_trajectory_dataset,
    make_frame_filename,
    save_comparison_sheet,
    save_contact_sheet,
    save_rollout_frame,
    save_rollout_gif,
    save_t_plus_one_diagnostics_sheet,
    select_fixed_rollout_steps,
    select_fixed_trajectory_indices,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_N_TRAIN_TRAJECTORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("generate_examples")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "examples"
CONTACT_SHEET_TRAJECTORY_COUNT = 2
CONTACT_SHEET_STEP_COUNT = 4
GIF_FPS = 2
IMAGE_DPI = 150


def load_data():
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:DEFAULT_N_TRAIN_TRAJECTORIES], dataset.test_states


def fit_forecaster(forecaster, train_states, test_states):
    forecaster.fit(train_states)
    return forecaster


def evaluate_spec(spec, train_states, test_states):
    logger.info("Evaluating %s for examples", spec.name)
    forecaster = fit_forecaster(spec.factory(), train_states, test_states)
    rollout = forecaster.evaluate_rollout(test_states)
    one_step = forecaster.evaluate_one_step(test_states)
    one_step_common = compute_common_horizon_metrics(
        one_step,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )
    rollout_common = compute_common_horizon_metrics(
        rollout,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )
    rollout_diagnostics = compute_common_horizon_diagnostics(
        rollout,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )
    target_common, pred_common = extract_common_horizon_predictions(
        rollout,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )

    return {
        "spec": spec,
        "rollout": rollout,
        "one_step": one_step,
        "target_common": target_common,
        "pred_common": pred_common,
        "n_trajectories": int(test_states.shape[0]),
        "total_steps": int(test_states.shape[1]),
        "spatial_shape": tuple(test_states.shape[2:]),
        "metrics": {
            "one_step_r2_raw": one_step["spatial_r2"],
            "one_step_r2_common": one_step_common["r2"],
            "one_step_rmse_common": one_step_common["rmse"],
            "one_step_mae_common": one_step_common["mae"],
            "one_step_rel_frob_common": one_step_common["rel_frob"],
            "rollout_r2_raw": rollout["spatial_r2"],
            "rollout_r2_common": rollout_common["r2"],
            "rollout_rmse_common": rollout_common["rmse"],
            "rollout_mae_common": rollout_common["mae"],
            "rollout_rel_frob_common": rollout_common["rel_frob"],
            "stability_gap_r2_common": one_step_common["r2"] - rollout_common["r2"],
            "rollout_worst_trajectory_indices": rollout_diagnostics["worst_trajectory_indices"],
            "rollout_worst_trajectory_rmse": rollout_diagnostics["worst_trajectory_rmse"],
        },
    }


def generate_per_model_artifacts(evaluation, trajectory_index, rollout_steps):
    spec = evaluation["spec"]
    model_dir = OUTPUT_ROOT / spec.slug
    if model_dir.exists():
        shutil.rmtree(model_dir)
    frames_dir = model_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    target = evaluation["target_common"][trajectory_index]
    pred = evaluation["pred_common"][trajectory_index]

    frame_paths = []
    for step_index in rollout_steps:
        frame_path = frames_dir / make_frame_filename(step_index)
        save_rollout_frame(
            target_frame=target[step_index],
            pred_frame=pred[step_index],
            step_index=step_index,
            model_name=spec.name,
            save_path=frame_path,
            dpi=IMAGE_DPI,
        )
        frame_paths.append(str(frame_path.relative_to(OUTPUT_ROOT)))

    contact_sheet_path = model_dir / "contact_sheet.png"
    save_contact_sheet(
        target_frames=target[rollout_steps],
        pred_frames=pred[rollout_steps],
        step_indices=rollout_steps,
        title=f"{spec.name} fixed trajectory {trajectory_index}",
        save_path=contact_sheet_path,
        dpi=IMAGE_DPI,
    )

    gif_path = model_dir / "rollout.gif"
    save_rollout_gif(
        target_frames=target,
        pred_frames=pred,
        step_indices=list(range(target.shape[0])),
        model_name=spec.name,
        save_path=gif_path,
        fps=GIF_FPS,
    )

    artifacts = {
        "contact_sheet": str(contact_sheet_path.relative_to(OUTPUT_ROOT)),
        "gif": str(gif_path.relative_to(OUTPUT_ROOT)),
        "frames": frame_paths,
    }

    one_step = evaluation["one_step"]
    if "baseline_pred_spatial" in one_step:
        n_trajectories = evaluation["n_trajectories"]
        total_steps = evaluation["total_steps"]
        spatial_shape = evaluation["spatial_shape"]
        one_step_steps = int(one_step["n_eval_samples"]) // n_trajectories
        one_step_warmup = total_steps - one_step_steps
        diagnostic_steps = select_fixed_rollout_steps(
            one_step_steps,
            count=min(CONTACT_SHEET_STEP_COUNT, one_step_steps),
        )
        absolute_step_indices = [one_step_warmup + step_idx for step_idx in diagnostic_steps]

        target_one_step = np.asarray(one_step["target_spatial"], dtype=np.float64).reshape(
            n_trajectories,
            one_step_steps,
            *spatial_shape,
        )
        baseline_one_step = np.asarray(one_step["baseline_pred_spatial"], dtype=np.float64).reshape(
            n_trajectories,
            one_step_steps,
            *spatial_shape,
        )
        corrected_one_step = np.asarray(one_step["pred_spatial"], dtype=np.float64).reshape(
            n_trajectories,
            one_step_steps,
            *spatial_shape,
        )

        diagnostics_path = model_dir / "t_plus_one_diagnostics.png"
        save_t_plus_one_diagnostics_sheet(
            target_frames=target_one_step[trajectory_index, diagnostic_steps],
            baseline_frames=baseline_one_step[trajectory_index, diagnostic_steps],
            corrected_frames=corrected_one_step[trajectory_index, diagnostic_steps],
            step_indices=absolute_step_indices,
            title=f"{spec.name} t+1 diagnostics on trajectory {trajectory_index}",
            save_path=diagnostics_path,
            dpi=IMAGE_DPI,
        )
        artifacts["t_plus_one_diagnostics"] = str(diagnostics_path.relative_to(OUTPUT_ROOT))

    return {
        "name": spec.name,
        "slug": spec.slug,
        "metrics": evaluation["metrics"],
        "artifacts": artifacts,
    }


def generate_comparison_artifacts(evaluations, trajectory_indices, rollout_steps):
    comparison_dir = OUTPUT_ROOT / "comparison"
    if comparison_dir.exists():
        shutil.rmtree(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_paths = []
    first_key = next(iter(evaluations))
    target_reference = evaluations[first_key]["target_common"]

    for trajectory_index in trajectory_indices:
        comparison_path = comparison_dir / f"fixed_trajectory_{trajectory_index}.png"
        save_comparison_sheet(
            target_frames=target_reference[trajectory_index, rollout_steps],
            model_frames=[
                (
                    evaluation["spec"].name,
                    evaluation["pred_common"][trajectory_index, rollout_steps],
                )
                for evaluation in evaluations.values()
            ],
            step_indices=rollout_steps,
            trajectory_index=trajectory_index,
            save_path=comparison_path,
            dpi=IMAGE_DPI,
        )
        comparison_paths.append(str(comparison_path.relative_to(OUTPUT_ROOT)))

    return comparison_paths


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_states, test_states = load_data()

    evaluations = {}
    for spec in get_navier_stokes_model_specs():
        evaluations[spec.slug] = evaluate_spec(spec, train_states, test_states)

    reference_eval = next(iter(evaluations.values()))
    trajectory_indices = select_fixed_trajectory_indices(
        reference_eval["target_common"].shape[0],
        count=CONTACT_SHEET_TRAJECTORY_COUNT,
    )
    rollout_steps = select_fixed_rollout_steps(
        reference_eval["target_common"].shape[1],
        count=CONTACT_SHEET_STEP_COUNT,
    )
    primary_trajectory = trajectory_indices[0]

    per_model_entries = []
    for evaluation in evaluations.values():
        per_model_entries.append(
            generate_per_model_artifacts(
                evaluation,
                primary_trajectory,
                rollout_steps,
            )
        )

    comparison_artifacts = generate_comparison_artifacts(
        evaluations,
        trajectory_indices,
        rollout_steps,
    )

    manifest = build_examples_manifest(
        output_root=str(OUTPUT_ROOT),
        trajectory_indices=trajectory_indices,
        rollout_steps=rollout_steps,
        image_settings={
            "dpi": IMAGE_DPI,
            "fps": GIF_FPS,
            "warmup_steps": DEFAULT_COMMON_WARMUP_STEPS,
        },
        per_model=per_model_entries,
        comparison_artifacts=comparison_artifacts,
    )

    manifest_path = OUTPUT_ROOT / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Example generation complete: %s", manifest_path)


if __name__ == "__main__":
    main()
