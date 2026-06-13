#!/usr/bin/env python3
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import (
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    extract_common_horizon_predictions,
    get_navier_stokes_model_specs,
    load_navier_stokes_trajectory_dataset,
    save_rollout_frame,
    select_fixed_rollout_steps,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_N_TRAIN_TRAJECTORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("evaluate_all")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"


def load_data():
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:DEFAULT_N_TRAIN_TRAJECTORIES], dataset.test_states


def fit_forecaster(forecaster, train_states, test_states):
    forecaster.fit(train_states)
    return forecaster


def save_benchmark_frames(
    model_name, model_slug, rollout, test_states, out_dir, common_warmup_steps
):
    model_dir = out_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    target_common, pred_common = extract_common_horizon_predictions(
        rollout,
        test_states,
        common_warmup_steps,
    )
    step_indices = select_fixed_rollout_steps(target_common.shape[1], count=4)
    trajectory_idx = 0

    for step_idx in step_indices:
        save_rollout_frame(
            target_frame=target_common[trajectory_idx, step_idx],
            pred_frame=pred_common[trajectory_idx, step_idx],
            step_index=step_idx,
            model_name=model_name,
            save_path=model_dir / f"pred_t{step_idx}.png",
        )
    return {
        "trajectory_index": trajectory_idx,
        "step_indices": step_indices,
        "output_dir": str(model_dir),
    }


def run_model(spec, train_states, test_states, out_dir, common_warmup_steps):
    logger.info("\n%s\nRunning %s\n%s", "=" * 60, spec.name, "=" * 60)
    t0 = time.time()

    forecaster = fit_forecaster(spec.factory(), train_states, test_states)
    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)
    elapsed = time.time() - t0

    one_step_common = compute_common_horizon_metrics(one_step, test_states, common_warmup_steps)
    rollout_common = compute_common_horizon_metrics(rollout, test_states, common_warmup_steps)
    rollout_diagnostics = compute_common_horizon_diagnostics(
        rollout,
        test_states,
        common_warmup_steps,
    )
    plot_info = save_benchmark_frames(
        spec.name,
        spec.slug,
        rollout,
        test_states,
        out_dir,
        common_warmup_steps,
    )

    logger.info("==> %s metrics:", spec.name)
    logger.info("  One-step R2 (raw):           %.4f", one_step["spatial_r2"])
    logger.info("  One-step R2 (warmup=%d):     %.4f", common_warmup_steps, one_step_common["r2"])
    logger.info("  Rollout R2 (raw):            %.4f", rollout["spatial_r2"])
    logger.info("  Rollout R2 (warmup=%d):      %.4f", common_warmup_steps, rollout_common["r2"])
    logger.info("  Rollout RMSE (warmup=%d):    %.4f", common_warmup_steps, rollout_common["rmse"])
    logger.info(
        "  Rollout Frob (warmup=%d):    %.4f", common_warmup_steps, rollout_common["rel_frob"]
    )
    logger.info("  Time:         %.1fs", elapsed)

    return {
        "name": spec.name,
        "slug": spec.slug,
        "family": spec.family,
        "notes": spec.notes,
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
        "rollout_diagnostics_common": rollout_diagnostics,
        "common_warmup_steps": common_warmup_steps,
        "time": elapsed,
        "benchmark_frames": plot_info,
    }


def save_summary_chart(results, out_dir, common_warmup_steps):
    names = [result["name"] for result in results]
    r2s = [result["rollout_r2_common"] for result in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, r2s, color="skyblue", edgecolor="black")
    ax.set_xlabel(f"Rollout Spatial R² (common warmup={common_warmup_steps})")
    ax.set_title("TBMD Forecasters Comparison (Navier-Stokes)")
    ax.axvline(x=0, color="gray", linestyle="--")
    for bar in bars:
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    chart_path = out_dir / "model_comparison_chart.png"
    plt.savefig(chart_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return chart_path


def main():
    train_states, test_states = load_data()
    logger.info("Train trajectories: %s", train_states.shape)
    logger.info("Test trajectories: %s", test_states.shape)

    out_dir = PROJECT_ROOT / "scripts" / "plots" / "models_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in get_navier_stokes_model_specs():
        results.append(
            run_model(
                spec,
                train_states,
                test_states,
                out_dir,
                DEFAULT_COMMON_WARMUP_STEPS,
            )
        )

    chart_path = save_summary_chart(results, out_dir, DEFAULT_COMMON_WARMUP_STEPS)
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "train_shape": list(train_states.shape),
                "test_shape": list(test_states.shape),
                "common_warmup_steps": DEFAULT_COMMON_WARMUP_STEPS,
                "results": results,
                "chart_path": str(chart_path),
            },
            fh,
            indent=2,
        )

    logger.info("Evaluation complete! Chart saved to: %s", chart_path)
    logger.info("Metrics summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
